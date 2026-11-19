"""ComplEx pretraining for the original-taxonomy boundary expansion, with exact full-state resume.

Scientific model, loss, N3, Adagrad update, seed-0 initialization, batch order and RNG use are those of the
pretrainer that produced the reused 633450-update reference state (train.py sha256 af824065...). Runtime-only
differences: the update log is a persistent buffered stream, flushed and fsynced before every checkpoint,
pause and completion, so committed log prefixes stay exact; the heartbeat is written every 10 seconds
instead of every update; the periodic checkpoint interval and the per-process time bound are configuration
values; and an exact terminal full-state import is admitted only for a parent named by the FROZEN protocol.
"""
import argparse, fcntl, hashlib, json, os, pathlib, random, re, signal, shutil, time
import numpy as np
import torch
from complex_model import ComplEx, n3
from state_compare import compare


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def atomic(path, data):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def dump(path, value):
    atomic(path, json_bytes(value))


def canonical(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def check_config(config, protocol):
    assert protocol['status'] == 'FROZEN'
    required = {'rank': 400, 'init_size': .001, 'batch_size': 1000, 'optimizer': 'Adagrad', 'seed': 0, 'dtype': 'float32'}
    for key, value in required.items():
        assert config[key] == value, (key, value)
    assert [config['learning_rate'], config['regularization']] in protocol['pretrain_grid']
    assert config['maximum_steps'] == protocol['pretrain_total_updates']
    assert config['milestone_updates'] == protocol['pretrain_milestone_updates']
    assert config['rolling_full_checkpoints'] == 2 and config['strict_deterministic_algorithms'] is True
    assert config['checkpoint_every_updates'] >= 1 and 60 <= config['checkpoint_every_seconds'] <= 1800
    assert 0 < config['maximum_seconds'] <= 86400
    assert protocol['admitted_graph_complete_sha256'][config['data_complete_sha256']] == config['taxonomy_mode']
    assert config['taxonomy_mode'] in protocol['taxonomy_modes']
    if 'import_parent' in config:
        assert protocol['allow_pretrain_import'] is True
        admitted = [p for p in protocol['admitted_import_parents'] if p['complete_sha256'] == config['import_parent']['complete_sha256']]
        assert len(admitted) == 1
        for key in ('checkpoint_sha256', 'identity_sha256', 'steps', 'learning_rate', 'regularization', 'data_complete_sha256'):
            expected = config['import_parent'][key] if key in config['import_parent'] else config[key]
            assert admitted[0][key] == expected, key


def run(config_path, output, allocation_stop_at=None):
    config_path = pathlib.Path(config_path)
    config = json.loads(config_path.read_text())
    output = pathlib.Path(output)
    owned = False
    protocol_path = pathlib.Path(config['protocol_path'])
    assert sha(protocol_path) == config['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    check_config(config, protocol)
    data = pathlib.Path(config['data_dir'])
    completion = json.loads((data / 'COMPLETE.json').read_text())
    assert sha(data / 'COMPLETE.json') == config['data_complete_sha256']
    for n, r in completion['files'].items():
        assert sha(data / n) == (r['sha256'] if isinstance(r, dict) else r)
    source_map = {n: sha(pathlib.Path(__file__).parent / n) for n in ['train.py', 'complex_model.py', 'state_compare.py']}
    assert source_map == config['source_hashes']
    scientific = {k: v for k, v in config.items() if k != 'data_dir'}
    identity = {'config': scientific, 'config_sha256': sha(config_path), 'data_complete_sha256': config['data_complete_sha256'], 'sources': source_map,
                'protocol_sha256': config['protocol_sha256'], 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda}
    identity_sha = canonical(identity)
    if (output / 'COMPLETE.json').exists():
        complete = json.loads((output / 'COMPLETE.json').read_text())
        assert complete['identity_sha256'] == identity_sha
        for n, h in complete['files'].items():
            assert sha(output / n) == h
        print('ALREADY_COMPLETE_NO_MUTATION', flush=True)
        return 0
    if output.exists():
        assert (output / 'identity.json').exists() and canonical(json.loads((output / 'identity.json').read_text())) == identity_sha
    else:
        output.mkdir(parents=True)
        dump(output / 'identity.json', identity)
    lock = (output / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owned = True
    stopping = []

    def terminate(sig, frame):
        stopping.append({'signal': sig, 'received_at': time.time()})
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        if (output / 'execution.json').exists():
            recovery = output / 'recovery'
            recovery.mkdir(exist_ok=True)
            os.replace(output / 'execution.json', recovery / ('execution-prior-%d.json' % time.time_ns()))
        dump(output / 'execution.json', {'job_id': os.environ.get('SLURM_JOB_ID'), 'slurm_step': os.environ.get('SLURM_STEP_ID'), 'time': time.time(), 'phase': 'startup_before_model_load', 'allocation_stop_at': allocation_stop_at})
        torch.use_deterministic_algorithms(True, warn_only=False)
        assert torch.cuda.is_available()
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        graph = np.load(data / 'triples.npy', mmap_mode='r')
        assert graph.shape == (config['training_triples'], 3)
        v = json.loads((data / 'vocabulary.json').read_text())
        assert len(v['entities']) == config['entity_count'] and len(v['relations']) == config['relation_count']
        assert config['steps_per_epoch'] == (len(graph) + 999) // 1000
        model = ComplEx(len(v['entities']), len(v['relations'])).cuda()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config['learning_rate'])
        step = 0
        epoch = 0
        position = 0
        order = None
        log = output / 'updates.jsonl'
        artifacts = {}
        last_commit_step = None
        started = time.time()
        last_save = started
        log_stream = None
        last_heartbeat = 0.
        last_log_flush = started
        phase_totals = {'batch_prepare': 0., 'compute_sync': 0., 'log_buffer_write': 0., 'heartbeat': 0., 'updates': 0}

        def flush_log():
            if log_stream is not None:
                log_stream.flush()
                os.fsync(log_stream.fileno())

        def state():
            return {'identity_sha256': identity_sha, 'step': step, 'epoch': epoch, 'position': position, 'order': order, 'model': to_cpu(model.state_dict()), 'optimizer': to_cpu(optimizer.state_dict()),
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'log_bytes': log.stat().st_size if log.exists() else 0, 'log_sha256': sha(log) if log.exists() else hashlib.sha256(b'').hexdigest()}

        def restore_from(saved):
            nonlocal step, epoch, position, order
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            step = saved['step']
            epoch = saved['epoch']
            position = saved['position']
            order = saved['order']
            random.setstate(saved['python_rng'])
            np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])

        def commit(reason):
            nonlocal last_commit_step, last_save, artifacts
            if step == last_commit_step:
                return json.loads((output / 'progress.json').read_text())
            begun = time.time()
            name = 'checkpoints/step-%06d.pt' % step
            path = output / name
            path.parent.mkdir(exist_ok=True)
            assert not path.exists(), 'Preserve unexpected existing checkpoint'
            flush_log()
            temp = path.with_suffix('.pt.tmp')
            assert not temp.exists()
            if log.exists():
                with log.open('rb') as f:
                    os.fsync(f.fileno())
            assert shutil.disk_usage(output).free >= 5 * 1024**3, 'Insufficient free storage for atomic checkpoint: require5GiB'
            st = state()
            copied = time.time()
            torch.save(st, temp)
            with temp.open('rb') as f:
                os.fsync(f.fileno())
            os.replace(temp, path)
            serialized = time.time()
            digest = sha(path)
            artifacts[name] = digest
            if step in config['milestone_updates']:
                model_name = 'models/update-%07d.pt' % step
                model_path = output / model_name
                model_path.parent.mkdir(exist_ok=True)
                if not model_path.exists():
                    model_temp = model_path.with_suffix('.pt.tmp')
                    assert not model_temp.exists()
                    torch.save({'identity_sha256': identity_sha, 'step': step, 'epoch_index': epoch, 'position_in_epoch': position, 'actual_epochs_complete': step / config['steps_per_epoch'],
                                'base_equivalent_epochs': step / protocol['base_steps_per_epoch'], 'taxonomy_mode': config['taxonomy_mode'], 'model': st['model'], 'source_full_checkpoint_sha256': digest}, model_temp)
                    with model_temp.open('rb') as f:
                        os.fsync(f.fileno())
                    os.replace(model_temp, model_path)
                artifacts[model_name] = sha(model_path)
            full_names = sorted([n for n in artifacts if n.startswith('checkpoints/')], key=lambda n: int(re.fullmatch(r'checkpoints/step-(\d+)\.pt', n).group(1)))
            retired = [n for n in full_names[:-2]]
            artifacts = {n: h for n, h in artifacts.items() if n not in retired}
            progress = {'identity_sha256': identity_sha, 'step': step, 'epoch': epoch, 'position': position, 'checkpoint': name, 'checkpoint_sha256': digest, 'checkpoint_bytes': path.stat().st_size,
                        'fresh_checkpoint_written': True, 'reason': reason, 'state_cpu_copy_seconds': copied - begun, 'serialize_fsync_seconds': serialized - copied, 'write_hash_seconds': time.time() - begun,
                        'log_bytes': st['log_bytes'], 'log_sha256': st['log_sha256'], 'artifacts': dict(artifacts)}
            dump(output / 'progress.json', progress)
            last_commit_step = step
            last_save = time.time()
            with (output / 'commits.jsonl').open('a') as f:
                f.write(json.dumps(progress) + '\n')
                f.flush()
                os.fsync(f.fileno())
            if retired:
                with (output / 'retention.jsonl').open('a') as f:
                    for old_name in retired:
                        old_path = output / old_name
                        assert old_path.parent == output / 'checkpoints'
                        f.write(json.dumps({'retired': old_name, 'new_checkpoint': name, 'reason': 'new-owned-run rolling2; historical hash retained in commits journal'}) + '\n')
                        old_path.unlink()
                    f.flush()
                    os.fsync(f.fileno())
            return progress
        if (output / 'progress.json').exists():
            progress = json.loads((output / 'progress.json').read_text())
            assert progress['identity_sha256'] == identity_sha
            path = output / progress['checkpoint']
            assert sha(path) == progress['checkpoint_sha256']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['identity_sha256'] == identity_sha
            recovery = output / 'recovery'
            for cp in (output / 'checkpoints').iterdir():
                match = re.fullmatch(r'step-(\d+)\.pt', cp.name)
                if not match or 'checkpoints/' + cp.name not in progress['artifacts']:
                    recovery.mkdir(exist_ok=True)
                    os.replace(cp, recovery / (str(time.time_ns()) + '-' + cp.name))
            raw = log.read_bytes() if log.exists() else b''
            prefix = raw[:saved['log_bytes']]
            assert len(prefix) == saved['log_bytes'] and hashlib.sha256(prefix).hexdigest() == saved['log_sha256']
            if len(raw) > len(prefix):
                recovery.mkdir(exist_ok=True)
                (recovery / ('uncommitted-log-%d.bin' % time.time_ns())).write_bytes(raw[len(prefix):])
                atomic(log, prefix)
            restore_from(saved)
            artifacts = dict(progress['artifacts'])
            last_commit_step = step
            for marker in ('PAUSED.json', 'ATTENTION.json'):
                if (output / marker).exists():
                    recovery.mkdir(exist_ok=True)
                    os.replace(output / marker, recovery / ('%s-%d.json' % (marker[:-5], time.time_ns())))
            restored = state()
            proof = compare(saved, restored)
            assert proof['exact']
            dump(output / ('restore-proof-%06d-%d.json' % (step, time.time_ns())), {'source_checkpoint': progress, 'all_state_exact': proof, 'snapshot_saved': False, 'reason': 'Source-bound in-allocation comparison before any resumed update'})
            del restored, saved
        elif 'import_parent' in config:
            assert not list(output.glob('checkpoints/*')) and not log.exists()
            parent = pathlib.Path(config['import_parent']['training_dir'])
            done = json.loads((parent / 'COMPLETE.json').read_text())
            assert sha(parent / 'COMPLETE.json') == config['import_parent']['complete_sha256']
            parent_identity = json.loads((parent / 'identity.json').read_text())
            assert sha(parent / 'identity.json') == done['files']['identity.json']
            assert canonical(parent_identity) == done['identity_sha256'] == config['import_parent']['identity_sha256']
            parent_config = parent_identity['config']
            for key in ('rank', 'init_size', 'batch_size', 'optimizer', 'seed', 'dtype', 'learning_rate', 'regularization', 'training_triples', 'entity_count', 'relation_count', 'steps_per_epoch', 'data_complete_sha256'):
                assert parent_config[key] == config[key], key
            assert parent_identity['sources']['complex_model.py'] == source_map['complex_model.py']
            progress = json.loads((parent / 'progress.json').read_text())
            assert sha(parent / 'progress.json') == done['files']['progress.json']
            path = parent / progress['checkpoint']
            assert sha(path) == progress['checkpoint_sha256'] == config['import_parent']['checkpoint_sha256'] == done['files'][progress['checkpoint']]
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['identity_sha256'] == done['identity_sha256']
            assert saved['step'] == done['steps'] == config['import_parent']['steps'] and saved['step'] in config['milestone_updates']
            prefix = (parent / 'updates.jsonl').read_bytes()[:saved['log_bytes']]
            assert hashlib.sha256(prefix).hexdigest() == saved['log_sha256']
            atomic(log, prefix)
            restore_from(saved)
            restored = state()
            fields = ('step', 'epoch', 'position', 'order', 'model', 'optimizer', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng', 'log_bytes', 'log_sha256')
            proof = compare({k: saved[k] for k in fields}, {k: restored[k] for k in fields})
            assert proof['exact']
            dump(output / 'IMPORT.json', {'parent_training_dir': str(parent.resolve()), 'parent_complete_sha256': config['import_parent']['complete_sha256'], 'parent_checkpoint_sha256': progress['checkpoint_sha256'],
                                          'steps_imported': step, 'restored_exact': proof, 'parent_identity_sha256': saved['identity_sha256'],
                                          'parent_model_only_milestones': {name: digest for name, digest in done['files'].items() if name.startswith('models/')}})
            del saved, restored
            commit('import_parent_terminal_full_state')
        else:
            assert not list(output.glob('checkpoints/*')) and not log.exists(), 'No committed state: preserve interrupted root'
            commit('fresh_initialization')
        log_stream = log.open('a', buffering=1024 * 1024)
        dump(output / 'execution.json', {'gpu': torch.cuda.get_device_name(0), 'device_properties': str(torch.cuda.get_device_properties(0)), 'pid': os.getpid(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                                         'step_id': os.environ.get('SLURM_STEP_ID'), 'slurm_step': os.environ.get('SLURM_STEP_ID'), 'time': time.time(), 'identity_sha256': identity_sha, 'allocation_stop_at': allocation_stop_at,
                                         'flags': {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(), 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                                                   'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}})
        while step < config['maximum_steps']:
            if stopping or time.time() - started > config['maximum_seconds'] or (allocation_stop_at is not None and time.time() >= allocation_stop_at):
                progress = commit('signal' if stopping else 'deadline')
                dump(output / 'PAUSED.json', {'step': step, 'signals': stopping, 'progress': progress, 'consistent_exit_receipt_time': time.time(), 'reason': 'signal' if stopping else 'deadline'})
                return 75
            if order is None or position == len(graph):
                if order is not None:
                    epoch += 1
                order = torch.randperm(len(graph))
                position = 0
            loop_started = time.time()
            indices = order[position:position + 1000]
            batch = torch.as_tensor(np.array(graph[indices.numpy()], copy=True), device='cuda', dtype=torch.long)
            begun = time.time()
            predictions, factors = model(batch)
            fit = torch.nn.functional.cross_entropy(predictions, batch[:, 2], reduction='mean')
            regularization = n3(factors, config['regularization'])
            loss = fit + regularization
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            finite = all(torch.isfinite(p.grad.coalesce().values() if p.grad.is_sparse else p.grad).all() for p in model.parameters() if p.grad is not None)
            assert finite
            optimizer.step()
            torch.cuda.synchronize()
            compute_seconds = time.time() - begun
            step += 1
            position += len(indices)
            logging_started = time.time()
            log_stream.write(json.dumps({'step': step, 'epoch': epoch, 'position': position, 'batch_count': len(indices), 'indices_sha256': hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                                         'fit': fit.item(), 'n3': regularization.item(), 'loss': loss.item(), 'finite_gradients': True, 'update_seconds': time.time() - begun, 'gpu_peak_bytes': torch.cuda.max_memory_allocated()}) + '\n')
            phase_totals['batch_prepare'] += begun - loop_started
            phase_totals['compute_sync'] += compute_seconds
            phase_totals['log_buffer_write'] += time.time() - logging_started
            phase_totals['updates'] += 1
            if time.time() - last_heartbeat >= 10:
                heartbeat_started = time.time()
                dump(output / 'heartbeat.json', {'step': step, 'epoch': epoch, 'position': position, 'time': time.time(), 'cumulative_phase_seconds': dict(phase_totals)})
                last_heartbeat = time.time()
                phase_totals['heartbeat'] += last_heartbeat - heartbeat_started
            if time.time() - last_log_flush >= 10:
                log_stream.flush()
                last_log_flush = time.time()
            if not stopping and step in config['milestone_updates']:
                commit('milestone_update_%d' % step)
            elif not stopping and (step % config['checkpoint_every_updates'] == 0 or time.time() - last_save >= config['checkpoint_every_seconds']):
                commit('periodic_%d_seconds' % config['checkpoint_every_seconds'])
        flush_log()
        progress = commit('final')
        files = dict(artifacts)
        for name in ('identity.json', 'updates.jsonl', 'progress.json', 'commits.jsonl', 'execution.json'):
            files[name] = sha(output / name)
        if (output / 'IMPORT.json').exists():
            files['IMPORT.json'] = sha(output / 'IMPORT.json')
        dump(output / 'COMPLETE.json', {'identity_sha256': identity_sha, 'steps': step, 'actual_epochs_complete': step / config['steps_per_epoch'],
                                        'base_equivalent_epochs': step / protocol['base_steps_per_epoch'], 'taxonomy_mode': config['taxonomy_mode'], 'elapsed_seconds': time.time() - started, 'files': files})
        return 0
    except BaseException as e:
        if owned:
            dump(output / 'ATTENTION.json', {'identity_sha256': identity_sha, 'error': repr(e), 'time': time.time()})
        raise
    finally:
        if 'log_stream' in locals() and log_stream is not None:
            log_stream.flush()
            log_stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allocation-stop-at', type=float)
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.output, args.allocation_stop_at))
