"""Fixed-budget cached-bank CLMPT training for the expanded study with exact full-state resume.

Differences from the frozen executed trainer (clmpt-two-stage-search-20261112/query/code/train.py, 0feb7b):
numerical whitelists come from a FROZEN protocol file whose hash is pinned in the config; the curriculum
mixture is required configuration; scheduled-best and step-0 fallback states are retained as model-only
exports while only the latest (and a pending-evaluation) full state is kept; negative-ID logging is an
explicit protocol switch, the draw digest is always logged; continuations resume the exact terminal full
state of a completed parent to the protocol's longer TOTAL budget. No early stopping exists.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import shutil
import time
import numpy as np
import torch
from model import QueryModel
from sampler import Sampler
from evaluate import evaluate
from metrics import selector
from state_compare import compare


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def canonical(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def dump(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


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
    """Every numerical and curriculum setting must be admitted by the frozen protocol."""
    assert protocol['status'] == 'FROZEN'
    assert config['dtype'] == 'float32' and config['strict_deterministic_algorithms'] is True
    assert config['rank'] == 400 and config['batch_size'] == 128 and config['seed'] == 0
    assert config['negative_sample_size'] == 128 and config['profile_batch_size'] == 32
    assert config['optimizer'] == 'AdamW' and config['weight_decay'] == protocol['query_weight_decay']
    assert config['transformer_layers'] == 2 and config['heads'] == 8 and config['dropout'] == .1
    assert config['learning_rate'] in protocol['query_learning_rates']
    assert config['temperature'] in protocol['temperatures']
    assert config['maximum_steps'] in protocol['maximum_steps']
    assert config['evaluation_every_updates'] == protocol['evaluation_every_updates']
    assert config['checkpoint_retention'] == protocol['checkpoint_retention'] == 'latest_full_plus_modelonly_initial_rawbest_calbest'
    assert config['log_negative_ids'] == protocol['log_negative_ids']
    assert config['data_complete_sha256'] in protocol['admitted_bank_complete_sha256']
    assert config['mixture'] == protocol['mixtures'][config['mixture_id']]
    assert config['taxonomy_mode'] in ('original', 'closure')
    assert protocol['admitted_bank_complete_sha256'][config['data_complete_sha256']] == config['taxonomy_mode']
    assert config['initializer_taxonomy_mode'] == config['taxonomy_mode']
    assert config['initializer_update_budget'] in protocol['initializer_update_budgets']


def run(config_path, output, allocation_stop_at=None):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    protocol_path = Path(config['protocol_path'])
    assert sha(protocol_path) == config['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    check_config(config, protocol)
    sources = {name: sha(Path(__file__).parent / name) for name in config['source_hashes']}
    assert sources == config['source_hashes']
    data = Path(config['data_dir'])
    assert sha(data / 'COMPLETE.json') == config['data_complete_sha256']
    for name, digest in json.loads((data / 'COMPLETE.json').read_text())['files'].items():
        assert sha(data / name) == (digest['sha256'] if isinstance(digest, dict) else digest)
    identity = {'config': {key: value for key, value in config.items() if key != 'data_dir'}, 'config_sha256': sha(config_path), 'sources': sources,
                'protocol_sha256': config['protocol_sha256'], 'torch': torch.__version__, 'cuda': torch.version.cuda}
    identity_sha = canonical(identity)
    if output.exists():
        assert (output / 'identity.json').exists() and canonical(json.loads((output / 'identity.json').read_text())) == identity_sha
    else:
        output.mkdir(parents=True)
        dump(output / 'identity.json', identity)
    lock = (output / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'COMPLETE.json').exists():
        complete = json.loads((output / 'COMPLETE.json').read_text())
        assert complete['identity_sha256'] == identity_sha
        for name, digest in complete['files'].items():
            assert sha(output / name) == digest
        return 0
    stopping = []

    def terminate(sig, frame):
        stopping.append({'signal': sig, 'received_at': time.time()})
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    started = time.time()
    dump(output / ('execution-%d.json' % time.time_ns()), {'job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_step': os.environ.get('SLURM_STEP_ID'), 'pid': os.getpid(), 'time': started,
        'allocation_stop_at': allocation_stop_at, 'phase': 'before_model_load'})
    root = str(output.resolve())
    try:
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cuda.matmul.allow_tf32 = False
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        bank = json.loads((data / 'bank.json').read_text())
        sampler = Sampler(bank, config['mixture'])
        dump(output / 'sampler-description.json', sampler.describe())
        pretrained = Path(config['pretrained_checkpoint'])
        assert sha(pretrained) == config['pretrained_sha256']
        initial = torch.load(pretrained, map_location='cpu', weights_only=False)
        model = QueryModel(initial['model'], bank['hp_ids'], temperature=config['temperature'])
        del initial
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        log = output / 'updates.jsonl'
        step, position, curve, artifacts, pruned = 0, 0, [], {}, {}
        pending = 0
        pending_checkpoint = None
        initial_checkpoint = None
        current_checkpoint = None
        retained_best = {'raw': None, 'calibrated': None}
        last_save = time.time()

        def snapshot():
            return {'identity_sha256': identity_sha, 'step': step, 'position': position,
                'model': to_cpu(model.state_dict()), 'optimizer': to_cpu(optimizer.state_dict()),
                'sampler': copy.deepcopy(sampler.state()), 'python_rng': random.getstate(),
                'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'curve': copy.deepcopy(curve),
                'pending_evaluation': pending, 'pending_checkpoint': copy.deepcopy(pending_checkpoint), 'log_bytes': log.stat().st_size if log.exists() else 0,
                'log_sha256': sha(log) if log.exists() else hashlib.sha256(b'').hexdigest(),
                'initial_checkpoint': initial_checkpoint, 'retained_best': copy.deepcopy(retained_best)}

        def restore(value):
            nonlocal step, position, curve, pending, pending_checkpoint, initial_checkpoint, retained_best
            model.load_state_dict(value['model'], strict=True)
            optimizer.load_state_dict(value['optimizer'])
            sampler.restore(value['sampler'])
            random.setstate(value['python_rng'])
            np.random.set_state(value['numpy_rng'])
            torch.set_rng_state(value['torch_rng'])
            torch.cuda.set_rng_state_all(value['cuda_rng'])
            step, position = value['step'], value['position']
            curve, pending = value['curve'], value['pending_evaluation']
            pending_checkpoint = value.get('pending_checkpoint')
            initial_checkpoint = value['initial_checkpoint']
            retained_best = value['retained_best']
            actual = snapshot()
            fields = ('step', 'position', 'model', 'optimizer', 'sampler', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng', 'curve', 'pending_evaluation')
            result = compare({key: value[key] for key in fields}, {key: actual[key] for key in fields})
            assert result['exact']
            return result

        def export_model(source_sha256):
            """Model-only state at the current step, bound to the full checkpoint it was taken from.

            Keys under layer.nbp. alias the same storage as nbp. and are stored once; a loader restores them
            from the recorded alias map before load_state_dict(strict=True).
            """
            name = 'models/step-%06d.pt' % step
            path = output / name
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                state = model.state_dict()
                aliases = {key: 'nbp.' + key[len('layer.nbp.'):] for key in state if key.startswith('layer.nbp.')}
                assert all(state[key].data_ptr() == state[target].data_ptr() for key, target in aliases.items())
                temporary = path.with_suffix('.pt.tmp')
                torch.save({'identity_sha256': identity_sha, 'step': step, 'model': {key: to_cpu(value) for key, value in state.items() if key not in aliases},
                            'aliases': aliases, 'source_full_checkpoint_sha256': source_sha256}, temporary)
                with temporary.open('rb') as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            return {'path': name, 'sha256': sha(path), 'step': step, 'root': root, 'kind': 'model_only', 'source_full_checkpoint_sha256': source_sha256}

        def commit(reason, refresh_best=False):
            nonlocal current_checkpoint, artifacts, last_save, initial_checkpoint, pending_checkpoint, pruned, retained_best
            if reason in ('signal', 'deadline', 'terminal') and current_checkpoint and current_checkpoint['step'] == step:
                previous = json.loads((output / 'progress.json').read_text())
                if previous['pending_evaluation'] == pending:
                    assert sha(output / previous['checkpoint']) == previous['checkpoint_sha256']
                    return previous
            if log.exists():
                with log.open('rb') as stream:
                    os.fsync(stream.fileno())
            assert shutil.disk_usage(output).free >= 8 * 1024**3
            if initial_checkpoint is None:
                assert step == 0
                initial_checkpoint = export_model(None)
            if refresh_best:
                for score_type in ('raw', 'calibrated'):
                    chosen = selector(curve, score_type)['step']
                    if chosen == 0:
                        retained_best[score_type] = dict(initial_checkpoint)
                    elif chosen == step:
                        retained_best[score_type] = export_model(curve[-1]['checkpoint']['sha256'])
                    else:
                        assert retained_best[score_type] and retained_best[score_type]['step'] == chosen
            name = 'checkpoints/step-%06d-%d.pt' % (step, time.time_ns())
            path = output / name
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_suffix('.pt.tmp')
            torch.save(snapshot(), temporary)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            digest = sha(path)
            current_checkpoint = {'path': name, 'sha256': digest, 'step': step, 'root': root, 'kind': 'full'}
            if pending is not None and pending_checkpoint is None:
                pending_checkpoint = dict(current_checkpoint)
            updated = {**artifacts, name: digest}
            for pointer in [initial_checkpoint] + list(retained_best.values()):
                if pointer and pointer['root'] == root:
                    updated[pointer['path']] = pointer['sha256']
            keep = {name}
            for pointer in [initial_checkpoint, pending_checkpoint] + list(retained_best.values()):
                if pointer and pointer['root'] == root:
                    keep.add(pointer['path'])
            assert sum(item.startswith('checkpoints/') for item in keep) <= 2 and len(keep) <= 5
            removal = {old: value for old, value in updated.items() if old not in keep}
            artifacts = {old: value for old, value in updated.items() if old in keep}
            progress = {'identity_sha256': identity_sha, 'step': step, 'position': position,
                'checkpoint': name, 'checkpoint_sha256': digest, 'checkpoint_bytes': path.stat().st_size,
                'reason': reason, 'pending_evaluation': pending, 'pending_checkpoint': pending_checkpoint, 'artifacts': artifacts,
                'initial_checkpoint': initial_checkpoint, 'retained_best': retained_best,
                'retained_artifacts': len(artifacts), 'pruned': {**pruned, **removal}, 'log_bytes': log.stat().st_size if log.exists() else 0,
                'log_sha256': sha(log) if log.exists() else hashlib.sha256(b'').hexdigest()}
            dump(output / 'progress.json', progress)
            for old, expected in removal.items():
                assert (old.startswith('checkpoints/step-') or old.startswith('models/step-')) and old.endswith('.pt')
                assert sha(output / old) == expected
                (output / old).unlink()
            pruned.update(removal)
            with (output / 'commits.jsonl').open('a') as stream:
                stream.write(json.dumps(progress) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            last_save = time.time()
            return progress
        if (output / 'progress.json').exists():
            progress = json.loads((output / 'progress.json').read_text())
            assert progress['identity_sha256'] == identity_sha
            assert sha(output / progress['checkpoint']) == progress['checkpoint_sha256']
            for name, digest in progress['artifacts'].items():
                assert sha(output / name) == digest
            value = torch.load(output / progress['checkpoint'], map_location='cpu', weights_only=False)
            raw = log.read_bytes() if log.exists() else b''
            prefix = raw[:value['log_bytes']]
            assert len(prefix) == value['log_bytes'] and hashlib.sha256(prefix).hexdigest() == value['log_sha256']
            if len(raw) > len(prefix):
                recovery = output / 'recovery'
                recovery.mkdir(exist_ok=True)
                (recovery / ('uncommitted-log-%d.bin' % time.time_ns())).write_bytes(raw[len(prefix):])
                log.write_bytes(prefix)
            proof = restore(value)
            artifacts, pruned = progress['artifacts'], progress['pruned']
            initial_checkpoint = progress['initial_checkpoint']
            pending_checkpoint = progress.get('pending_checkpoint')
            current_checkpoint = {'path': progress['checkpoint'], 'sha256': progress['checkpoint_sha256'], 'step': step, 'root': root, 'kind': 'full'}
            dump(output / ('restore-proof-%d.json' % time.time_ns()), {'source': current_checkpoint, 'actual_restored_state': proof})
            del value
        elif config.get('continuation_parent'):
            assert config['maximum_steps'] == protocol['continuation_total_steps']
            parent = Path(config['continuation_parent'])
            assert sha(parent / 'COMPLETE.json') == config['parent_complete_sha256']
            done = json.loads((parent / 'COMPLETE.json').read_text())
            assert done['steps'] == protocol['continuation_parent_steps']
            assert sha(parent / 'progress.json') == done['files']['progress.json']
            assert sha(parent / 'identity.json') == done['files']['identity.json']
            parent_config = json.loads((parent / 'identity.json').read_text())['config']
            for name in ('learning_rate', 'temperature', 'rank', 'weight_decay', 'batch_size', 'seed', 'data_complete_sha256', 'pretrained_sha256', 'source_hashes',
                         'evaluation_every_updates', 'mixture_id', 'mixture', 'taxonomy_mode', 'protocol_sha256', 'log_negative_ids', 'checkpoint_retention'):
                assert parent_config[name] == config[name], name
            progress = json.loads((parent / 'progress.json').read_text())
            cp = parent / progress['checkpoint']
            assert sha(cp) == config['parent_checkpoint_sha256'] == progress['checkpoint_sha256']
            value = torch.load(cp, map_location='cpu', weights_only=False)
            parent_steps = protocol['continuation_parent_steps']
            assert value['step'] == parent_steps and value['position'] == parent_steps * 128 and value['pending_evaluation'] is None
            assert [row['step'] for row in value['curve']] == list(range(0, parent_steps + 1, config['evaluation_every_updates']))
            assert value['identity_sha256'] == done['identity_sha256']
            for pointer in [value['initial_checkpoint']] + list(value['retained_best'].values()):
                assert pointer and sha(Path(pointer['root']) / pointer['path']) == pointer['sha256']
            prefix = (parent / 'updates.jsonl').read_bytes()[:value['log_bytes']]
            assert hashlib.sha256(prefix).hexdigest() == value['log_sha256']
            log.write_bytes(prefix)
            proof = restore(value)
            dump(output / 'IMPORT.json', {'parent_checkpoint': str(cp.resolve()), 'sha256': sha(cp), 'restored_exact': proof,
                'parent_complete_sha256': sha(parent / 'COMPLETE.json'), 'steps_imported': parent_steps, 'best_histories_preserved': True})
            del value
            commit('import_terminal_full_state_with_selection_history')
        else:
            assert not log.exists()
            commit('initial')
        dump(output / ('runtime-%d.json' % time.time_ns()), {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'flags': {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(), 'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32, 'cublas': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}})
        fits = output / 'fits'
        fits.mkdir(exist_ok=True)

        def deadline():
            return bool(stopping) or (allocation_stop_at is not None and time.time() >= allocation_stop_at) or time.time() - started >= config['maximum_seconds']
        while True:
            if deadline():
                progress = commit('signal' if stopping else 'deadline')
                dump(output / 'PAUSED.json', {'reason': 'signal' if stopping else 'deadline', 'step': step,
                    'signals': stopping, 'progress': progress, 'consistent_exit_receipt_time': time.time()})
                return 75
            if pending is not None:
                assert pending == step
                if current_checkpoint['step'] != step:
                    commit('pending_scheduled_evaluation')
                assert pending_checkpoint and pending_checkpoint['step'] == step
                result = evaluate(model, sampler, bank, fits, step, identity_sha, pending_checkpoint['sha256'], should_stop=deadline)
                if result is None:
                    progress = commit('signal' if stopping else 'deadline')
                    dump(output / 'PAUSED.json', {'reason': 'signal' if stopping else 'deadline', 'step': step,
                        'signals': stopping, 'progress': progress, 'pending_evaluation_preserved': True, 'consistent_exit_receipt_time': time.time()})
                    return 75
                record = {'step': step, 'validation': result['validation'], 'checkpoint': dict(pending_checkpoint),
                    'fit_metadata': str((fits / ('fit-step-%06d.json' % step)).resolve()),
                    'fit_metadata_sha256': sha(fits / ('fit-step-%06d.json' % step)),
                    'scores': str((fits / ('fit-step-%06d.npz' % step)).resolve()), 'scores_sha256': result['scores_sha256']}
                curve.append(record)
                pending = None
                pending_checkpoint = None
                dump(output / 'curve.json', curve)
                dump(output / 'selection.json', {name: selector(curve, name) for name in ('raw', 'calibrated')})
                commit('scheduled_evaluation_complete', refresh_best=True)
            if step >= config['maximum_steps']:
                break
            queries, positive, negative, receipt = sampler.draw(step)
            begun = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(queries, torch.tensor(positive, device='cuda'), torch.tensor(negative, device='cuda'))
            assert torch.isfinite(loss)
            loss.backward()
            flags = [torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]
            assert bool(torch.stack(flags).all().item())
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            position += 128
            logged = receipt if config['log_negative_ids'] else {key: value for key, value in receipt.items() if key != 'negative_ids'}
            with log.open('a') as stream:
                stream.write(json.dumps({'step': step, 'loss': loss.item(), 'finite_gradients': True,
                    'query_count': 128, 'query_types': [query['type'] for query in queries], 'update_seconds': time.time() - begun, **logged}) + '\n')
            dump(output / 'heartbeat.json', {'step': step, 'query_occurrences': position, 'time': time.time()})
            if step % config['evaluation_every_updates'] == 0:
                pending = step
                pending_checkpoint = None
                commit('pending_scheduled_evaluation')
            elif time.time() - last_save >= 300:
                commit('periodic')
        progress = commit('terminal')
        files = dict(artifacts)
        for name in ('identity.json', 'updates.jsonl', 'progress.json', 'commits.jsonl', 'curve.json', 'selection.json', 'sampler-description.json'):
            files[name] = sha(output / name)
        for path in output.iterdir():
            if path.is_file() and path.suffix == '.json' and path.name not in ('COMPLETE.json', 'heartbeat.json', 'PAUSED.json', 'ATTENTION.json'):
                files[path.name] = sha(path)
        for path in fits.iterdir():
            if path.is_file():
                files[str(path.relative_to(output))] = sha(path)
        dump(output / 'COMPLETE.json', {'identity_sha256': identity_sha, 'steps': step, 'query_occurrences': position,
            'files': files, 'elapsed_seconds_this_invocation': time.time() - started, 'terminal_checkpoint': progress,
            'selection': {name: selector(curve, name) for name in ('raw', 'calibrated')}, 'fixed_budget_no_early_stop': True})
        return 0
    except BaseException as error:
        dump(output / 'ATTENTION.json', {'identity_sha256': identity_sha, 'error': repr(error), 'time': time.time()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allocation-stop-at', type=float)
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.output, args.allocation_stop_at))
