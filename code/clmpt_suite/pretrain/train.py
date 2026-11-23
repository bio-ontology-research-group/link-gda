"""ComplEx pretraining for the CLMPT suite with a configurable seed and exact full-state resume.

Adapted from code/clmpt_boundary/train.py (sha256 c60de2e5...). The scientific model, cross-entropy plus N3
loss, sparse Adagrad update, batch size, initialization and batch-order algorithm are unchanged: the run seed
seeds Python, NumPy, Torch CPU and CUDA, then the model is constructed with no randomness drawn in between,
and each epoch's triple order is torch.randperm on the CPU generator. Seed 0 therefore follows the historical
initialization and order on an identical graph. Every full state stores the model, Adagrad accumulators,
epoch, the current permutation and position in it, all RNG states and the committed log prefix.

Changed: seed 0-9 instead of a hardcoded 0; recipe (learning rate, N3 weight, total updates) and graph
identity (modality, fold, cohort) are admitted by a frozen suite protocol; production needs the protocol's
frozen recipe; cuDNN/TF32 policy and PYTHONHASHSEED=0 are pinned; no parent import (the suite reuses existing
initializers directly rather than extending them); qualification runs may pause at a chosen step.
"""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import random
import re
import signal
import shutil
import time

import numpy as np
import torch

from complex_model import ComplEx, n3
from state_compare import compare


SOURCE_FILES = ('train.py', 'complex_model.py', 'state_compare.py')
PURPOSES = ('qualification', 'production')


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


def pin_numerics():
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert os.environ.get('PYTHONHASHSEED') == '0'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')


def check_config(config, protocol):
    assert protocol['status'] == 'FROZEN' and protocol['protocol_id'].startswith('clmpt-suite-pretrain-')
    assert config['purpose'] in PURPOSES
    for key, value in protocol['fixed'].items():
        assert config[key] == value, (key, value)
    assert isinstance(config['seed'], int) and config['seed'] in protocol['seeds']
    assert config['strict_deterministic_algorithms'] is True and config['rolling_full_checkpoints'] == 2
    assert config['checkpoint_every_updates'] >= 1 and 60 <= config['checkpoint_every_seconds'] <= 1800
    assert 0 < config['maximum_seconds'] <= 86400
    graph = protocol['admitted_graphs'][config['data_complete_sha256']]
    assert config['graph'] == {key: graph[key] for key in ('taxonomy_mode', 'modality', 'fold', 'cohort')}
    assert config['purpose'] in graph['purpose_admitted']
    recipe = protocol['recipes'][config['recipe_id']]
    assert config['learning_rate'] == recipe['learning_rate'] and config['regularization'] == recipe['regularization']
    if config['purpose'] == 'production':
        assert protocol['production_recipe_id'] is not None and config['recipe_id'] == protocol['production_recipe_id']
        assert config['maximum_steps'] == recipe['total_updates'] and config['milestone_updates'] == [recipe['total_updates']]
    else:
        assert config['maximum_steps'] in protocol['qualification_maximum_steps']
        assert config['milestone_updates'] == [config['maximum_steps']]
    for key in ('training_triples', 'entity_count', 'relation_count'):
        assert config[key] == graph[key], key


def run(config_path, output, allocation_stop_at=None, pause_at_step=None):
    config_path = pathlib.Path(config_path)
    config = json.loads(config_path.read_text())
    output = pathlib.Path(output)
    owned = False
    protocol_path = pathlib.Path(config['protocol_path'])
    assert sha(protocol_path) == config['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    check_config(config, protocol)
    assert pause_at_step is None or (config['purpose'] == 'qualification' and 0 < pause_at_step < config['maximum_steps'])
    data = pathlib.Path(config['data_dir'])
    completion = json.loads((data / 'COMPLETE.json').read_text())
    assert sha(data / 'COMPLETE.json') == config['data_complete_sha256']
    for n, r in completion['files'].items():
        assert sha(data / n) == (r['sha256'] if isinstance(r, dict) else r)
    source_map = {n: sha(pathlib.Path(__file__).resolve().parent / n) for n in SOURCE_FILES}
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
        pin_numerics()
        if (output / 'execution.json').exists():
            recovery = output / 'recovery'
            recovery.mkdir(exist_ok=True)
            os.replace(output / 'execution.json', recovery / ('execution-prior-%d.json' % time.time_ns()))
        dump(output / 'execution.json', {'job_id': os.environ.get('SLURM_JOB_ID'), 'slurm_step': os.environ.get('SLURM_STEP_ID'), 'time': time.time(), 'phase': 'startup_before_model_load',
                                         'allocation_stop_at': allocation_stop_at, 'pause_at_step': pause_at_step})
        assert torch.cuda.is_available()
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
            name = 'checkpoints/step-%07d.pt' % step
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
                    torch.save({'identity_sha256': identity_sha, 'step': step, 'seed': seed, 'graph': config['graph'], 'recipe_id': config['recipe_id'], 'epoch_index': epoch,
                                'position_in_epoch': position, 'actual_epochs_complete': step / config['steps_per_epoch'], 'taxonomy_mode': config['graph']['taxonomy_mode'],
                                'model': st['model'], 'source_full_checkpoint_sha256': digest}, model_temp)
                    with model_temp.open('rb') as f:
                        os.fsync(f.fileno())
                    os.replace(model_temp, model_path)
                artifacts[model_name] = sha(model_path)
            full_names = sorted([n for n in artifacts if n.startswith('checkpoints/')], key=lambda n: int(re.fullmatch(r'checkpoints/step-(\d+)\.pt', n).group(1)))
            retired = [n for n in full_names[:-2]]
            artifacts = {n: h for n, h in artifacts.items() if n not in retired}
            progress = {'identity_sha256': identity_sha, 'step': step, 'epoch': epoch, 'position': position, 'checkpoint': name, 'checkpoint_sha256': digest, 'checkpoint_bytes': path.stat().st_size,
                        'fresh_checkpoint_written': True, 'reason': reason, 'state_cpu_copy_seconds': copied - begun, 'serialize_fsync_seconds': serialized - copied, 'write_hash_seconds': time.time() - begun,
                        'log': log.name, 'log_bytes': st['log_bytes'], 'log_sha256': st['log_sha256'], 'artifacts': dict(artifacts)}
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
                        f.write(json.dumps({'retired': old_name, 'new_checkpoint': name, 'reason': 'owned-run rolling2; historical hash retained in commits journal'}) + '\n')
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
            dump(output / ('restore-proof-%07d-%d.json' % (step, time.time_ns())), {'source_checkpoint': progress, 'all_state_exact': proof, 'reason': 'Source-bound comparison before any resumed update'})
            del restored, saved
        else:
            assert not list(output.glob('checkpoints/*')) and not log.exists(), 'No committed state: preserve interrupted root'
            commit('fresh_initialization')
        log_stream = log.open('a', buffering=1024 * 1024)
        dump(output / 'execution.json', {'gpu': torch.cuda.get_device_name(0), 'device_properties': str(torch.cuda.get_device_properties(0)), 'pid': os.getpid(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                                         'slurm_step': os.environ.get('SLURM_STEP_ID'), 'time': time.time(), 'identity_sha256': identity_sha, 'allocation_stop_at': allocation_stop_at,
                                         'pause_at_step': pause_at_step,
                                         'flags': {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
                                                   'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
                                                   'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                                                   'pythonhashseed': os.environ.get('PYTHONHASHSEED'), 'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                                                   'torch_threads': torch.get_num_threads()}})
        while step < config['maximum_steps']:
            if stopping or time.time() - started > config['maximum_seconds'] or (allocation_stop_at is not None and time.time() >= allocation_stop_at) or (pause_at_step is not None and step == pause_at_step):
                reason = 'signal' if stopping else 'qualification_pause' if pause_at_step is not None and step == pause_at_step else 'deadline'
                progress = commit(reason)
                dump(output / 'PAUSED.json', {'step': step, 'signals': stopping, 'progress': progress, 'consistent_exit_receipt_time': time.time(), 'reason': reason})
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
        dump(output / 'COMPLETE.json', {'identity_sha256': identity_sha, 'steps': step, 'seed': seed, 'graph': config['graph'], 'recipe_id': config['recipe_id'],
                                        'actual_epochs_complete': step / config['steps_per_epoch'], 'elapsed_seconds': time.time() - started, 'files': files,
                                        'initializer_model': 'models/update-%07d.pt' % step})
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
    parser.add_argument('--pause-at-step', type=int)
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.output, args.allocation_stop_at, args.pause_at_step))
