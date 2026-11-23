"""Fixed-budget CLMPT query-stage trainer for the suite, with exact full-state resume.

Adapted from code/clmpt_boundary_queries/query/code/train.py (sha256 4bf080fe...). Kept: durable output
contract (identity, progress, full state, model-only initial/raw-best/calibrated-best, saved score matrices,
COMPLETE, PAUSED with return code 75, ATTENTION, non-blocking writer lock), exact restore proof, fixed budget
without early stopping, scheduled VALIDATION evaluation and separate raw/calibrated selection, exact terminal
import for continuations.

Changed: the configured seed (0-9) seeds Python, NumPy, Torch CPU and CUDA immediately before model
construction; the sampler streams derive from sampler_seed; cuDNN deterministic/benchmark/TF32 and CUDA
matmul TF32 are pinned and PYTHONHASHSEED=0 is required; encoder (shared, independent, mean), curriculum
(full13, core3, aux_off) and ontology_loss are configuration admitted by the frozen protocol; validation
counts come from the protocol's admitted-bank record; purpose separates qualification, HPO screens,
continuations and production; phase must be 'train' and a bank carrying TEST cases is refused; files a
verified PRUNED.json names may be absent from a completed run; qualification runs may pause at a chosen step.
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

from evaluate import evaluate
from metrics import selector
from model import ENCODERS, SuiteQueryModel
from sampler import Sampler, domain_identity, gene_domain
from state_compare import compare


SOURCE_FILES = ('train.py', 'model.py', 'original_model.py', 'packed_encode.py', 'sampler.py', 'evaluate.py', 'metrics.py', 'state_compare.py')
REFERENCE_FILES = ('official-clmpt.py', 'official-complex.py', 'official-transformer.py', 'official-training.py')
PURPOSES = ('qualification', 'hpo_screen', 'hpo_continuation', 'production')
NEVER_PRUNED = ('fits/', 'curve.json', 'selection.json', 'identity.json', 'updates.jsonl', 'progress.json', 'commits.jsonl')
RETENTION = 'latest_full_plus_modelonly_initial_rawbest_calbest'
INITIALIZER_KEYS = ('taxonomy_mode', 'cohort', 'modality', 'fold', 'pretrain_seed', 'pretrain_learning_rate', 'pretrain_regularization', 'initializer_update_budget')
COHORTS = ('main', 'excluded')


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


def pin_numerics():
    """Deterministic numerical policy shared by training, evaluation and qualification."""
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert os.environ.get('PYTHONHASHSEED') == '0'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')


def numerics_record():
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(), 'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'float32_matmul_precision': torch.get_float32_matmul_precision(), 'cublas': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'pythonhashseed': os.environ.get('PYTHONHASHSEED'), 'threads': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'torch_threads': torch.get_num_threads()}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config, bank, device='cuda'):
    """Seed every generator with the run seed, then construct; nothing may draw randomness in between."""
    pretrained = Path(config['pretrained_checkpoint'])
    assert sha(pretrained) == config['pretrained_sha256']
    initial = torch.load(pretrained, map_location='cpu', weights_only=False)
    seed_everything(config['seed'])
    model = SuiteQueryModel(initial['model'], bank['hp_ids'], config['encoder'], device=device, temperature=config['temperature'])
    del initial
    return model


def check_config(config, protocol):
    """Every numerical, curriculum and identity setting must be admitted by the frozen protocol."""
    assert protocol['status'] == 'FROZEN' and protocol['protocol_id'].startswith('clmpt-suite-query-')
    assert config['phase'] == 'train' and config['purpose'] in PURPOSES and config.get('test_evaluation', False) is False
    rules = protocol['purposes'][config['purpose']]
    cadence = rules['evaluation_every_updates']
    assert config['dtype'] == 'float32' and config['strict_deterministic_algorithms'] is True
    assert config['rank'] == 400 and config['batch_size'] == 128 and config['profile_batch_size'] == 32 and config['negative_sample_size'] == 128
    assert config['optimizer'] == 'AdamW' and config['weight_decay'] == protocol['query_weight_decay']
    assert config['transformer_layers'] == 2 and config['heads'] == 8 and config['dropout'] == .1
    assert config['seed'] in protocol['seeds'] and config['sampler_seed'] in protocol['seeds']
    assert isinstance(config['seed'], int) and isinstance(config['sampler_seed'], int)
    assert config['encoder'] in ENCODERS and config['encoder'] in protocol['encoders']
    curriculum = protocol['curricula'][config['curriculum']]
    assert config['ontology_loss'] is curriculum['ontology_loss'] and config['mixture_id'] == curriculum['mixture_id']
    assert config['mixture'] == (protocol['mixtures'][config['mixture_id']] if config['mixture_id'] else None)
    assert config['learning_rate'] in protocol['query_learning_rates'] and config['temperature'] in protocol['temperatures']
    assert config['maximum_steps'] in rules['maximum_steps'] and config['evaluation_every_updates'] in (cadence if isinstance(cadence, list) else [cadence])
    assert config['maximum_steps'] % config['evaluation_every_updates'] == 0
    assert 0 < config['maximum_seconds'] <= protocol['maximum_seconds']
    assert config['checkpoint_retention'] == protocol['checkpoint_retention'] == RETENTION
    assert config['log_negative_ids'] == protocol['log_negative_ids']
    bank = protocol['admitted_banks'][config['data_complete_sha256']]
    initializer = config['initializer']
    assert set(initializer) == set(INITIALIZER_KEYS), {'missing': sorted(set(INITIALIZER_KEYS) - set(initializer)), 'unknown': sorted(set(initializer) - set(INITIALIZER_KEYS))}
    assert initializer['taxonomy_mode'] == bank['taxonomy_mode'] and initializer['fold'] == bank['fold'] and initializer['cohort'] == bank['cohort']
    assert bank['modality'] in bank['admitted_modalities'] and initializer['modality'] in bank['admitted_modalities'], (initializer['modality'], bank['admitted_modalities'])
    assert config['purpose'].split('_')[0] in bank['purpose_admitted'], (config['purpose'], bank['purpose_admitted'])
    if config['purpose'] in ('hpo_screen', 'hpo_continuation'):
        assert initializer['modality'] == bank['modality'], 'HPO admits only the bank primary modality'
    recipe = {key: initializer[key] for key in ('pretrain_learning_rate', 'pretrain_regularization', 'initializer_update_budget')}
    for key, allowed in rules.get('fixed', {}).items():
        assert config[key] in allowed, (key, config[key], allowed)
    if config['purpose'] in ('hpo_screen', 'hpo_continuation'):
        assert recipe in protocol['hpo_initializer_recipes'] and initializer['pretrain_seed'] == 0
        assert 'hpo' in bank['purpose_admitted']
    if config['purpose'] == 'production':
        frozen = protocol.get('production_recipe')
        assert frozen is not None, 'production requires a protocol with a frozen common recipe'
        assert recipe == frozen['initializer_recipe'] and config['learning_rate'] == frozen['learning_rate'] and config['temperature'] == frozen['temperature']
        assert 'production' in bank['purpose_admitted']
    if config['purpose'] == 'hpo_continuation':
        assert config['maximum_steps'] == rules['maximum_steps'][0] and config.get('continuation_parent')
    else:
        assert not config.get('continuation_parent')


def check_bank(bank, spec, manifest):
    """The bank's clinical TRAIN gene universe must equal the protocol record and, if declared, the bank COMPLETE."""
    assert 'test' not in bank and 'test_cases' not in bank, 'training banks must not carry TEST cases'
    assert spec['cohort'] in COHORTS and isinstance(spec['admitted_modalities'], list) and spec['admitted_modalities']
    assert bank['num_entities'] == spec['entity_count'], 'bank vocabulary size differs from the protocol record (entity table, not the ranking pool)'
    domain = domain_identity(gene_domain(bank))
    assert domain == spec['clinical_training_gene_domain'], {'bank': domain, 'protocol': spec['clinical_training_gene_domain']}
    if 'training_gene_domain' in manifest:
        assert manifest['training_gene_domain'] == domain, {'bank_complete': manifest['training_gene_domain'], 'bank': domain}
    assert len(bank['candidate_ids']) == spec['candidate_genes'] == len(set(bank['candidate_ids']))
    assert len(bank['validation']) == spec['validation_cases']
    assert sum(len(row['gene_ids']) for row in bank['validation']) == spec['validation_associations']
    candidates = set(bank['candidate_ids'])
    assert all(set(row['gene_ids']) <= candidates for row in bank['validation']), 'validation targets must lie in the ranking pool'
    universe = set(gene_domain(bank)['ids'])
    assert all(set(row['answers']) <= universe for row in bank['training_profiles']), 'every clinical TRAIN positive must lie in the training gene universe'
    assert max(universe) < bank['num_entities'] and min(universe) >= 0
    return domain


def verify_complete(output, identity_sha):
    """Verify a completed run, admitting only absences recorded by a consistent PRUNED.json."""
    complete = json.loads((output / 'COMPLETE.json').read_text())
    assert complete['identity_sha256'] == identity_sha
    pruned = {}
    if (output / 'PRUNED.json').exists():
        record = json.loads((output / 'PRUNED.json').read_text())
        assert record['identity_sha256'] == identity_sha and record['receipt_sha256']
        pruned = record['files']
        terminal = complete['terminal_checkpoint']
        protected = {pointer['path'] for pointer in [terminal['initial_checkpoint']] + list(terminal['retained_best'].values()) if pointer}
        for name, digest in pruned.items():
            assert complete['files'][name] == digest and not name.startswith(NEVER_PRUNED) and name not in protected
    for name, digest in complete['files'].items():
        if name in pruned and not (output / name).exists():
            continue
        assert sha(output / name) == digest, name


def run(config_path, output, allocation_stop_at=None, pause_at_step=None):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    protocol_path = Path(config['protocol_path'])
    assert sha(protocol_path) == config['protocol_sha256']
    protocol = json.loads(protocol_path.read_text())
    check_config(config, protocol)
    assert pause_at_step is None or (config['purpose'] == 'qualification' and 0 < pause_at_step < config['maximum_steps'])
    here = Path(__file__).resolve().parent
    sources = {name: sha(here / name) for name in SOURCE_FILES}
    assert sources == config['source_hashes']
    references = {name: sha(here.parent / 'references' / name) for name in REFERENCE_FILES}
    assert references == config['reference_hashes'] == protocol['reference_hashes']
    data = Path(config['data_dir'])
    assert sha(data / 'COMPLETE.json') == config['data_complete_sha256']
    manifest = json.loads((data / 'COMPLETE.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(data / name) == (digest['sha256'] if isinstance(digest, dict) else digest)
    spec = protocol['admitted_banks'][config['data_complete_sha256']]
    identity = {'config': {key: value for key, value in config.items() if key != 'data_dir'}, 'config_sha256': sha(config_path), 'sources': sources,
                'references': references, 'protocol_sha256': config['protocol_sha256'], 'torch': torch.__version__, 'cuda': torch.version.cuda}
    identity_sha = canonical(identity)
    if output.exists():
        assert (output / 'identity.json').exists() and canonical(json.loads((output / 'identity.json').read_text())) == identity_sha
    else:
        output.mkdir(parents=True)
        dump(output / 'identity.json', identity)
    lock = (output / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'COMPLETE.json').exists():
        verify_complete(output, identity_sha)
        return 0
    stopping = []

    def terminate(sig, frame):
        stopping.append({'signal': sig, 'received_at': time.time()})
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    started = time.time()
    dump(output / ('execution-%d.json' % time.time_ns()), {'job_id': os.environ.get('SLURM_JOB_ID'), 'slurm_step': os.environ.get('SLURM_STEP_ID'),
         'pid': os.getpid(), 'time': started, 'allocation_stop_at': allocation_stop_at, 'pause_at_step': pause_at_step, 'phase': 'before_model_load'})
    root = str(output.resolve())
    try:
        pin_numerics()
        bank = json.loads((data / 'bank.json').read_text())
        check_bank(bank, spec, manifest)
        sampler = Sampler(bank, config['mixture'], config['sampler_seed'])
        dump(output / 'sampler-description.json', sampler.describe())
        model = build_model(config, bank)
        assert model.num_entities == bank['num_entities'], 'initializer entity table does not match the bank vocabulary'
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        rows_per_update = sampler.ontology_slots + 32
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
                torch.save({'identity_sha256': identity_sha, 'step': step, 'encoder': config['encoder'], 'model': {key: to_cpu(value) for key, value in state.items() if key not in aliases},
                            'aliases': aliases, 'source_full_checkpoint_sha256': source_sha256}, temporary)
                with temporary.open('rb') as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            return {'path': name, 'sha256': sha(path), 'step': step, 'root': root, 'kind': 'model_only', 'source_full_checkpoint_sha256': source_sha256}

        def commit(reason, refresh_best=False):
            nonlocal current_checkpoint, artifacts, last_save, initial_checkpoint, pending_checkpoint, pruned, retained_best
            if reason in ('signal', 'deadline', 'terminal', 'qualification_pause') and current_checkpoint and current_checkpoint['step'] == step:
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
                        'retained_artifacts': len(artifacts), 'pruned': {**pruned, **removal}, 'log': log.name, 'log_bytes': log.stat().st_size if log.exists() else 0,
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
            for marker in ('PAUSED.json', 'ATTENTION.json'):
                if (output / marker).exists():
                    recovery = output / 'recovery'
                    recovery.mkdir(exist_ok=True)
                    os.replace(output / marker, recovery / ('%s-%d.json' % (marker[:-5], time.time_ns())))
            dump(output / ('restore-proof-%d.json' % time.time_ns()), {'source': current_checkpoint, 'actual_restored_state': proof})
            del value
        elif config.get('continuation_parent'):
            rules = protocol['purposes']['hpo_continuation']
            parent_steps = rules['parent_steps']
            parent = Path(config['continuation_parent'])
            assert sha(parent / 'COMPLETE.json') == config['parent_complete_sha256']
            done = json.loads((parent / 'COMPLETE.json').read_text())
            assert done['steps'] == parent_steps
            assert sha(parent / 'progress.json') == done['files']['progress.json']
            assert sha(parent / 'identity.json') == done['files']['identity.json']
            parent_identity = json.loads((parent / 'identity.json').read_text())
            assert parent_identity['sources'] == sources and parent_identity['references'] == references
            parent_config = parent_identity['config']
            assert parent_config['purpose'] == 'hpo_screen'
            for name in ('learning_rate', 'temperature', 'rank', 'weight_decay', 'batch_size', 'seed', 'sampler_seed', 'encoder', 'curriculum', 'ontology_loss',
                         'data_complete_sha256', 'pretrained_sha256', 'initializer', 'source_hashes', 'reference_hashes', 'evaluation_every_updates', 'mixture_id',
                         'mixture', 'protocol_sha256', 'log_negative_ids', 'checkpoint_retention', 'dropout', 'heads', 'transformer_layers'):
                assert parent_config[name] == config[name], name
            progress = json.loads((parent / 'progress.json').read_text())
            checkpoint = parent / progress['checkpoint']
            assert sha(checkpoint) == config['parent_checkpoint_sha256'] == progress['checkpoint_sha256']
            value = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert value['step'] == parent_steps and value['position'] == parent_steps * rows_per_update and value['pending_evaluation'] is None
            assert [row['step'] for row in value['curve']] == list(range(0, parent_steps + 1, config['evaluation_every_updates']))
            assert value['identity_sha256'] == done['identity_sha256']
            for pointer in [value['initial_checkpoint']] + list(value['retained_best'].values()):
                assert pointer and sha(Path(pointer['root']) / pointer['path']) == pointer['sha256']
            prefix = (parent / 'updates.jsonl').read_bytes()[:value['log_bytes']]
            assert hashlib.sha256(prefix).hexdigest() == value['log_sha256']
            log.write_bytes(prefix)
            proof = restore(value)
            dump(output / 'IMPORT.json', {'parent_checkpoint': str(checkpoint.resolve()), 'sha256': sha(checkpoint), 'restored_exact': proof,
                                          'parent_complete_sha256': sha(parent / 'COMPLETE.json'), 'steps_imported': parent_steps, 'best_histories_preserved': True})
            del value
            commit('import_terminal_full_state_with_selection_history')
        else:
            assert not log.exists()
            commit('initial')
        dump(output / ('runtime-%d.json' % time.time_ns()), {'gpu': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
             'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'flags': numerics_record()})
        fits = output / 'fits'
        fits.mkdir(exist_ok=True)

        def deadline():
            return bool(stopping) or (allocation_stop_at is not None and time.time() >= allocation_stop_at) or time.time() - started >= config['maximum_seconds']

        def pause(reason, **extra):
            progress = commit(reason)
            dump(output / 'PAUSED.json', {'reason': reason, 'step': step, 'signals': stopping, 'progress': progress, 'consistent_exit_receipt_time': time.time(), **extra})
            return 75
        while True:
            if deadline():
                return pause('signal' if stopping else 'deadline')
            if pause_at_step is not None and step == pause_at_step and pending is None:
                return pause('qualification_pause')
            if pending is not None:
                assert pending == step
                if current_checkpoint['step'] != step:
                    commit('pending_scheduled_evaluation')
                assert pending_checkpoint and pending_checkpoint['step'] == step
                result = evaluate(model, sampler, bank, spec, fits, step, identity_sha, pending_checkpoint['sha256'], should_stop=deadline)
                if result is None:
                    return pause('signal' if stopping else 'deadline', pending_evaluation_preserved=True)
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
                continue
            if step >= config['maximum_steps']:
                break
            queries, positive, negative, receipt = sampler.draw(step)
            assert len(queries) == rows_per_update
            begun = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model.suite_loss(queries, torch.tensor(positive, device='cuda'), torch.tensor(negative, device='cuda'), sampler.ontology_slots, config['ontology_loss'])
            assert torch.isfinite(loss)
            loss.backward()
            flags = [torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]
            assert bool(torch.stack(flags).all().item())
            optimizer.step()
            torch.cuda.synchronize()
            step += 1
            position += rows_per_update
            logged = receipt if config['log_negative_ids'] else {key: value for key, value in receipt.items() if key != 'negative_ids'}
            with log.open('a') as stream:
                stream.write(json.dumps({'step': step, 'loss': loss.item(), 'finite_gradients': True, 'query_count': rows_per_update,
                                         'query_types': [query['type'] for query in queries], 'update_seconds': time.time() - begun, **logged}) + '\n')
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
            if path.is_file() and path.suffix == '.json' and path.name not in ('COMPLETE.json', 'heartbeat.json', 'PAUSED.json', 'ATTENTION.json', 'PRUNED.json'):
                files[path.name] = sha(path)
        for path in fits.iterdir():
            if path.is_file():
                files[str(path.relative_to(output))] = sha(path)
        dump(output / 'COMPLETE.json', {'identity_sha256': identity_sha, 'steps': step, 'query_occurrences': position, 'files': files,
                                        'elapsed_seconds_this_invocation': time.time() - started, 'terminal_checkpoint': progress,
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
    parser.add_argument('--pause-at-step', type=int)
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.output, args.allocation_stop_at, args.pause_at_step))
