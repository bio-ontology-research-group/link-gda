"""CPU rehearsal of the production pipeline with fake Slurm and a local fake SSH, on the real v3 manifests. Writes only under
--scratch. Everything synthetic (HPO files, trajectories, exports, score matrices) is labelled; nothing trains or scores a
real model, and nothing touches a remote host.

Covered: plan counts; check-hpo refuses a synthetic HPO; make_protocol generate --draft on the real manifests (DRAFT,
relabelled in scratch only); FROZEN_HPO_ARTIFACT reuse audit (pass, and rejection of a foreign graph, seed, recipe, encoder
and core source); bundle and staging of both sites; real release refused for a rehearsal; the reused parent is never
trained and the reused trajectory is admitted untouched; parent-by-parent delivery and admission (a corrupt delivery is
refused); per-unit TEST release gating (refused before an HPO freeze or before training is complete) and submission inside
the job cap; inference before archive before prune; same-matrix two-pool and S-mask readouts with calibration before any
filtering; verified PREEMPTED resume without dependency versus CANCELLED and a worker error; storage-permission gating,
HPO archive, and a protected dry-run release of HPO copies; deadlines and the utilization guard.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import SUITE, canonical, read_json, sha, write_replace  # noqa: E402
import units as U  # noqa: E402
import reuse as R  # noqa: E402
from bridge import Bridge  # noqa: E402

REPO = SUITE.parent.parent
V3 = REPO / 'data' / 'clmpt-suite-inputs-20261123-v3'

FAKE_SLURM = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
state = Path(os.environ['FAKE_SLURM_STATE'])
data = json.loads(state.read_text()) if state.exists() else {'next': 900000, 'jobs': {}}
name = Path(sys.argv[0]).name
args = sys.argv[1:]
if name == 'sbatch':
    if '--test-only' in args:
        sys.stderr.write('sbatch: Job 1 to start at now on fake\n'); sys.exit(0)
    job = str(data['next']); data['next'] += 1
    data['jobs'][job] = {'name': next(a.split('=', 1)[1] for a in args if a.startswith('--job-name=')), 'state': 'RUNNING',
                         'partition': next((a.split('=', 1)[1] for a in args if a.startswith('--partition=')), 'batch'), 'argv': args}
    state.write_text(json.dumps(data)); print(job)
elif name == 'squeue':
    for job, value in sorted(data['jobs'].items()):
        if value['state'] in ('RUNNING', 'PENDING'):
            print('%s|%s|%s|%s' % (job, value['state'], value['name'], value['partition']))
elif name == 'sacct':
    job = args[args.index('-j') + 1]
    print(data['jobs'].get(job, {}).get('final', 'COMPLETED'))
elif name == 'sinfo':
    print('batch-h100|gpu:nvidia_h100_80gb_hbm3:8|fake\nbatch|gpu:v100:8|v100')
elif name == 'scontrol':
    print('OverTimeLimit           = 60 min' if args[:2] == ['show', 'config'] else 'PartitionName=fake MaxTime=1-00:00:00')
else:
    print('fake')
'''
FAKE_SSH = '#!/bin/sh\nshift\nexec sh -c "$*"\n'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def call(command, env=None, stdin=None):
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True, env=env, input=stdin)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr[-4000:]}


def fake_state(scratch):
    return read_json(scratch / 'fake-slurm.json')


def set_job(scratch, job_name_suffix, **fields):
    state = fake_state(scratch)
    for job in state['jobs'].values():
        if job['name'].endswith(job_name_suffix):
            job.update(fields)
    write_replace(scratch / 'fake-slurm.json', state)


def trajectory(training, config, root=None):
    """A completed query-run directory in the trainer's format (selection, pointers, terminal state); contents synthetic."""
    for sub in ('fits', 'models', 'checkpoints'):
        (training / sub).mkdir(parents=True, exist_ok=True)
    root = root or training
    pointers = {}
    for step in (0, 1500, 4000):
        model = root / 'models' / ('step-%06d.pt' % step)
        model.parent.mkdir(parents=True, exist_ok=True)
        if not model.exists():
            model.write_bytes(b'SYNTHETIC model-only %d' % step)
        pointers[step] = {'path': 'models/step-%06d.pt' % step, 'sha256': sha(model), 'step': step, 'root': str(root), 'kind': 'model_only'}
    terminal = training / 'checkpoints' / 'step-020000-1.pt'
    terminal.write_bytes(b'SYNTHETIC terminal full state')
    (training / 'fits' / 'fit-step-000000.npz').write_bytes(b'SYNTHETIC scores')
    selection = {'raw': {'step': 1500, 'eligible_steps': [0, 1500]}, 'calibrated': {'step': 4000, 'eligible_steps': [0, 4000]}}
    write_replace(training / 'selection.json', selection)
    write_replace(training / 'curve.json', [{'step': 0, 'synthetic': True}])
    identity = {'config': {key: value for key, value in config.items() if key != 'data_dir'}, 'config_sha256': 'synthetic', 'sources': config.get('source_hashes'),
                'references': config.get('reference_hashes'), 'protocol_sha256': config.get('protocol_sha256'), 'torch': 'synthetic', 'cuda': None}
    write_replace(training / 'identity.json', identity)
    for name in ('progress.json', 'updates.jsonl', 'commits.jsonl', 'sampler-description.json'):
        (training / name).write_text('{"synthetic": true}\n')
    files = {str(path.relative_to(training)): sha(path) for path in training.rglob('*') if path.is_file()}
    write_replace(training / 'COMPLETE.json', {'identity_sha256': canonical(identity), 'steps': 20000, 'files': files, 'selection': selection, 'synthetic': True,
                                               'terminal_checkpoint': {'checkpoint': 'checkpoints/step-020000-1.pt', 'checkpoint_sha256': sha(terminal), 'step': 20000,
                                                                       'initial_checkpoint': pointers[0], 'retained_best': {'raw': pointers[1500], 'calibrated': pointers[4000]}}})


def complete_query_unit(campaign, unit):
    folder = campaign / 'units' / unit
    trajectory(folder / 'training', read_json(folder / 'config.json'))
    wave = folder / 'allocations' / 'wave-000'
    wave.mkdir(parents=True, exist_ok=True)
    write_replace(wave / 'ALLOCATION_COMPLETE.json', {'synthetic': True})


def fake_inference(campaign, unit):
    """The files inference_job.py writes, with synthetic content, so archive/prune ordering can be exercised."""
    inference = campaign / 'units' / unit / 'inference'
    test = inference / 'test'
    test.mkdir(parents=True, exist_ok=True)
    files = {}
    for view in ('initial', 'raw', 'calibrated', 'terminal'):
        (test / ('test-%s.npz' % view)).write_bytes(b'SYNTHETIC test scores ' + view.encode())
        (test / ('test-%s.json' % view)).write_text('{"synthetic": true}\n')
        files['test-%s.npz' % view], files['test-%s.json' % view] = sha(test / ('test-%s.npz' % view)), sha(test / ('test-%s.json' % view))
    (test / 'identity.json').write_text('{"synthetic": true}\n')
    files['identity.json'] = sha(test / 'identity.json')
    write_replace(test / 'COMPLETE.json', {'files': files, 'synthetic': True})
    (inference / 'readout.json').write_text('{"synthetic": true}\n')
    (inference / 'terminal-model.pt').write_bytes(b'SYNTHETIC terminal model-only')
    write_replace(inference / 'terminal-model.json', {'synthetic': True})
    listed = {'test/' + name: digest for name, digest in files.items()}
    listed['test/COMPLETE.json'] = sha(test / 'COMPLETE.json')
    for name in ('readout.json', 'terminal-model.pt', 'terminal-model.json', 'test-config.json', 'TEST-RELEASE.json', 'INFERENCE-PLAN.json'):
        listed[name] = sha(inference / name)
    write_replace(inference / 'INFERENCE_COMPLETE.json', {'unit': unit, 'files': listed, 'synthetic': True})


def fake_pretrain_output(campaign, unit):
    folder = campaign / 'units' / unit
    config = read_json(folder / 'config.json')
    training = folder / 'training'
    (training / 'models').mkdir(parents=True)
    (training / 'checkpoints').mkdir()
    total = config['maximum_steps']
    export = training / 'models' / ('update-%07d.pt' % total)
    export.write_bytes(b'SYNTHETIC REHEARSAL EXPORT ' + unit.encode())
    (training / 'checkpoints' / ('step-%07d.pt' % total)).write_bytes(b'SYNTHETIC FULL STATE')
    identity = {'config': {key: value for key, value in config.items() if key != 'data_dir'}, 'config_sha256': sha(folder / 'config.json'),
                'data_complete_sha256': config['data_complete_sha256'], 'sources': {name: sha(campaign / 'worker' / name) for name in ('train.py', 'complex_model.py', 'state_compare.py')},
                'protocol_sha256': config['protocol_sha256'], 'torch': 'synthetic', 'cuda_runtime': None}
    write_replace(training / 'identity.json', identity)
    write_replace(training / 'progress.json', {'step': total})
    for name in ('updates.jsonl', 'commits.jsonl', 'execution.json'):
        (training / name).write_text('{}\n')
    files = {'models/update-%07d.pt' % total: sha(export)}
    for name in ('identity.json', 'updates.jsonl', 'progress.json', 'commits.jsonl', 'execution.json'):
        files[name] = sha(training / name)
    write_replace(training / 'COMPLETE.json', {'identity_sha256': canonical(identity), 'steps': total, 'seed': config['seed'], 'graph': config['graph'], 'recipe_id': config['recipe_id'],
                                               'files': files, 'initializer_model': 'models/update-%07d.pt' % total, 'synthetic': True})
    wave = folder / 'allocations' / 'wave-000'
    wave.mkdir(parents=True)
    write_replace(wave / 'ALLOCATION_COMPLETE.json', {'synthetic': True})


def synthetic_hpo_campaign(scratch, prod, production, freeze_module, python):
    """A synthetic HPO campaign holding the calibrated winner's initializer (with its original delivery files) and the
    winner trajectory, plus the fetched evidence a real bridge would produce."""
    hpo = scratch / 'hpo-campaign'
    budget = production['initializer_recipe']['initializer_update_budget']
    delivery = scratch / 'hpo-initializers' / 'winner'
    delivery.mkdir(parents=True)
    export = delivery / ('update-%07d.pt' % budget)
    export.write_bytes(b'SYNTHETIC HPO WINNER INITIALIZER')
    legacy = {'learning_rate': production['initializer_recipe']['pretrain_learning_rate'], 'regularization': production['initializer_recipe']['pretrain_regularization'],
              'rank': 400, 'init_size': 0.001, 'batch_size': 1000, 'optimizer': 'Adagrad', 'dtype': 'float32', 'seed': 0, 'milestone_updates': [budget],
              'entity_count': 239552, 'relation_count': 624, 'taxonomy_mode': 'original',
              'data_complete_sha256': 'af68a9a3230fc1f05a105e1d57f02200129879c0bea5097c9094cf0feb0201ed'}
    write_replace(delivery / 'config.json', legacy)
    write_replace(delivery / 'identity.json', {'config': legacy, 'config_sha256': sha(delivery / 'config.json'), 'data_complete_sha256': legacy['data_complete_sha256'],
                                               'sources': {'train.py': 'LEGACY-BOUNDARY-TRAINER'}, 'protocol_sha256': 'LEGACY-BOUNDARY-PROTOCOL'})
    write_replace(delivery / 'ADMISSION.json', {'files': {export.name: sha(export), 'identity.json': sha(delivery / 'identity.json'), 'config.json': sha(delivery / 'config.json')}})
    record = {'name': 'winner', 'sha256': sha(export), 'bytes': export.stat().st_size, 'path': str(export), 'taxonomy_mode': 'original', 'cohort': 'main', 'modality': 'PFS',
              'fold': 0, 'pretrain_seed': 0, 'pretrain_learning_rate': legacy['learning_rate'], 'pretrain_regularization': legacy['regularization'],
              'initializer_update_budget': budget, 'delivery_admission': {'path': str(delivery / 'ADMISSION.json'), 'sha256': sha(delivery / 'ADMISSION.json')}}
    evidence = prod / 'hpo-evidence'
    evidence.mkdir(parents=True)
    write_replace(evidence / 'frozen-manifest.json', {'initializers': {'winner': record}})
    expected, _, _ = freeze_module.query_template(argparse.Namespace(python=python), prod, REPO, {'ibex_path': str(export), 'sha256': sha(export)})
    screen = hpo / 'trials' / 'S' / 'S-winner' / 'training'
    winner = hpo / 'trials' / 'C' / 'C-winner' / 'training'
    config = dict(expected['template'], purpose='hpo_continuation', protocol_sha256='HPO-V1-PROTOCOL', protocol_path='/hpo/protocol', experiment_id='C-winner',
                  continuation_parent=str(screen), parent_complete_sha256='x', parent_checkpoint_sha256='y')
    trajectory(screen, dict(config, purpose='hpo_screen', maximum_steps=5000))
    trajectory(winner, config, root=screen)
    trajectory(hpo / 'trials' / 'S' / 'S-other' / 'training', dict(config, purpose='hpo_screen', maximum_steps=5000, experiment_id='S-other'))
    (hpo / 'decisions').mkdir(parents=True)
    write_replace(hpo / 'decisions' / 'recipe.json', {'synthetic': True})
    write_replace(hpo / 'frozen-manifest.json', {'synthetic': True})
    write_replace(evidence / 'recipe.json', {'common_leaderboard': 'calibrated', 'common_winner': {'recipe': {key: production[key] for key in ('initializer_recipe', 'learning_rate', 'temperature')},
                                                                                                  'identities': {'pretrained_sha256': sha(export)}, 'training_dir': str(winner)}})
    for name in ('ADMISSION.json', 'identity.json', 'config.json'):
        shutil.copy(str(delivery / name), str(evidence / name))
    shutil.copy(str(winner / 'identity.json'), str(evidence / 'trajectory-identity.json'))
    shutil.copy(str(winner / 'COMPLETE.json'), str(evidence / 'trajectory-COMPLETE.json'))
    write_replace(evidence / 'FETCH-RECEIPT.json', {'files': {name: sha(evidence / name) for name in ('ADMISSION.json', 'identity.json', 'config.json', 'trajectory-identity.json', 'trajectory-COMPLETE.json')},
                                                    'winner_training_dir': str(winner), 'synthetic': True})
    return hpo, evidence, winner, screen


def transfer_scenario(scratch, orix, ibex, env, py, block, plan):
    """Blocked transfers must not block scheduling: ticks continue, pools stay bounded, the same unit is never transferred
    twice at once, a killed worker is resumed from its partials, a corrupt export is never admitted, caps are unchanged."""
    import signal
    config = scratch / 'bridge.json'
    bridge = Bridge(config)
    parents = ['main-PFS-f0-ps5', 'main-PFS-f0-ps6', 'main-PFS-f0-ps7']
    for parent in parents:
        fake_pretrain_output(orix, parent)
        set_job(scratch, parent, state='COMPLETED', final='COMPLETED')
    releases = {site: sha(campaign / 'ROOT_RELEASE.json') for site, campaign in (('orix', orix), ('ibex', ibex))}
    block.write_text('blocked')
    began = start = time.time()
    first = call([py, HERE / 'bridge.py', 'once', '--config', config], env)
    first_seconds = time.time() - began
    cycle1 = read_json(scratch / 'bridge-state' / 'last-cycle.json')
    locks = {parent: bridge.transfer_paths('delivery', parent)['lock'] for parent in parents}
    for _ in range(30):
        if sum(bridge.lock_held(path) for path in locks.values()) == 2:
            break
        time.sleep(0.5)
    held_before = {parent: bridge.lock_held(path) for parent, path in locks.items()}
    began = time.time()
    second = call([py, HERE / 'bridge.py', 'once', '--config', config], env)
    second_seconds = time.time() - began
    cycle2 = read_json(scratch / 'bridge-state' / 'last-cycle.json')
    ticks = {site: read_json(campaign / 'state' / 'last-tick.json')['time'] for site, campaign in (('orix', orix), ('ibex', ibex))}
    running = [parent for parent, held in held_before.items() if held]
    attempts_while_blocked = {parent: read_json(bridge.transfer_paths('delivery', parent)['state'], {}).get('attempts') for parent in running}
    duplicate = call([py, HERE / 'bridge.py', 'transfer', '--config', config, '--kind', 'delivery', '--unit', running[0]], env)
    victim = running[0]
    pid = read_json(bridge.transfer_paths('delivery', victim)['state'])['pid']
    os.killpg(os.getpgid(pid), signal.SIGKILL)
    for _ in range(20):
        if not bridge.lock_held(locks[victim]):
            break
        time.sleep(0.5)
    status_after_kill = bridge.transfer_status('delivery', victim)
    stop = scratch / 'bridge-state' / 'STOP'
    stop.write_text('stop')
    cycle_before_stop = read_json(scratch / 'bridge-state' / 'last-cycle.json')['time']
    stopped = call([py, HERE / 'bridge.py', 'once', '--config', config], env)
    stop_respected = stopped['returncode'] == 0 and read_json(scratch / 'bridge-state' / 'last-cycle.json')['time'] == cycle_before_stop
    stop.unlink()
    block.unlink()
    drained = call([py, HERE / 'bridge.py', 'drain', '--config', config, '--timeout', '600'], env)
    states = {parent: read_json(bridge.transfer_paths('delivery', parent)['state']) for parent in parents}
    corrupt = 'main-PFS-f0-ps8'
    fake_pretrain_output(orix, corrupt)
    set_job(scratch, corrupt, state='COMPLETED', final='COMPLETED')
    export = next((orix / 'units' / corrupt / 'training' / 'models').glob('update-*.pt'))
    export.write_bytes(b'CORRUPTED AFTER COMPLETE')
    failed = call([py, HERE / 'bridge.py', 'drain', '--config', config, '--timeout', '600'], env)
    corrupt_state = read_json(bridge.transfer_paths('delivery', corrupt)['state'], {})
    attention = read_json(scratch / 'bridge-state' / 'last-cycle.json').get('attention', [])
    running_jobs = sum(1 for job in fake_state(scratch)['jobs'].values() if job['state'] in ('RUNNING', 'PENDING'))
    checks = {
        'cycle_returns_while_transfers_blocked': first['returncode'] in (0, 1) and second['returncode'] in (0, 1) and first_seconds < 60 and second_seconds < 60,
        'ticks_continue_while_blocked': all(value >= start for value in ticks.values()) and cycle2['time'] > cycle1['time'] and all(held_before[parent] for parent in running),
        'delivery_pool_bounded_initializers_first': len(running) == 2 and sorted(cycle1['transfers']['launched']) == sorted('delivery:' + parent for parent in parents[:2]),
        'no_duplicate_transfer_of_same_unit': cycle2['transfers']['launched'] == [] and all(value == 1 for value in attempts_while_blocked.values())
                                              and 'busy_other_worker' in duplicate['stdout'],
        'killed_worker_resumed_from_partials': status_after_kill == 'pending' and states[victim]['status'] == 'done' and states[victim]['attempts'] == 2,
        'all_blocked_deliveries_admitted_after_unblock': drained['returncode'] in (0, 1) and all((ibex / 'initializers' / parent / 'ADMITTED.json').exists() for parent in parents)
                                                         and not any((scratch / 'transit' / ('delivery-' + parent)).exists() for parent in parents),
        'stop_file_respected': stop_respected,
        'corrupt_transfer_never_admitted_and_bounded': corrupt_state.get('status') == 'failed' and corrupt_state.get('attempts') == 2
                                                       and not (ibex / 'initializers' / corrupt).exists() and ('transfer_exhausted:delivery:' + corrupt) in attention,
        'caps_and_releases_unchanged': all(sha(campaign / 'ROOT_RELEASE.json') == releases[site] for site, campaign in (('orix', orix), ('ibex', ibex))) and running_jobs <= 24,
    }
    return {'checks': checks, 'seconds': {'first_once': first_seconds, 'second_once': second_seconds}, 'attempts_while_blocked': attempts_while_blocked,
            'states': {parent: {key: value.get(key) for key in ('status', 'attempts')} for parent, value in states.items()}, 'corrupt_state': corrupt_state.get('result'),
            'running_jobs': running_jobs, 'drain_stdout': drained['stdout'][-1500:], 'failed_drain_returncode': failed['returncode']}


def readout_check(scratch, python):
    """Runs readout_check_body under the numpy interpreter, so this rehearsal itself needs only the standard library."""
    result = call([python, '-I', __file__, '--readout-check', scratch])
    if result['returncode'] != 0:
        return {'passed': False, 'stderr': result['stderr']}
    return json.loads(result['stdout'].strip().splitlines()[-1])


def readout_check_body(scratch, python):
    """Synthetic scores on the real excluded TEST cases: two pools and the S mask from one matrix, calibration first."""
    import numpy as np
    sys.path.insert(0, str(SUITE / 'query' / 'code'))
    import metrics
    test_dir = V3 / 'excluded' / 'test'
    cases = read_json(test_dir / 'test-cases.json')
    bank_dir = V3 / 'excluded' / 'bank'
    bank = read_json(bank_dir / 'bank.json')
    seen = {gene for row in bank['training_profiles'] for gene in row['answers']}
    inference = scratch / 'readout-inference'
    inference.mkdir()
    rng = np.random.default_rng(0)
    real = rng.standard_normal((len(cases['cases']), len(cases['candidate_ids']))).astype(np.float32)
    answers = [row['gene_ids'] for row in cases['cases']]
    for view in ('raw', 'calibrated'):
        np.savez_compressed(inference / ('test-%s.npz' % view), test_real=real, candidate_ids=np.array(cases['candidate_ids']))
        own = {'raw': metrics.measure(real, answers, cases['candidate_ids'], seen_genes=seen), 'calibrated': metrics.measure(metrics.calibrate(real), answers, cases['candidate_ids'], seen_genes=seen)}
        write_replace(inference / ('test-%s.json' % view), {'metrics': {key: {k: v for k, v in value.items() if k != 'ranks'} for key, value in own.items()}, 'synthetic': True})
    result = call([python, '-I', HERE / 'readout.py', '--worker-code', SUITE / 'query' / 'code', '--inference-dir', inference, '--test-dir', test_dir, '--bank-dir', bank_dir,
                   '--output', scratch / 'readout.json'])
    if result['returncode'] != 0:
        return {'passed': False, 'stderr': result['stderr']}
    record = read_json(scratch / 'readout.json')
    readouts = record['views']['calibrated']['readouts']
    primary = [int(gene) for gene in cases['candidate_views']['primary_no_mgi']]
    position = {int(gene): index for index, gene in enumerate(cases['candidate_ids'])}
    columns = [position[gene] for gene in primary]
    calibrated_full = metrics.calibrate(real)
    s_rows, s_answers = [], []
    for row, case in enumerate(cases['cases']):
        labels = [gene for gene in case['gene_ids'] if gene in set(case['s_supported_gene_ids']) and gene in set(primary)]
        if labels:
            s_rows.append(row)
            s_answers.append(labels)
    expected = metrics.measure(calibrated_full[np.ix_(s_rows, columns)], s_answers, primary, seen_genes=seen)['answer_micro']
    recalibrated = metrics.measure(metrics.calibrate(real[s_rows])[:, columns], s_answers, primary, seen_genes=seen)['answer_micro']
    checks = {
        'pools': record['pools']['primary_no_mgi']['size'] == 5696 and record['pools']['secondary_full_go']['size'] == 18904,
        's_mask_associations_393': readouts['secondary_full_go/s_supported']['associations_in_mask'] == 393,
        'all_associations_425': readouts['secondary_full_go/all']['associations_in_mask'] == 425,
        'primary_counts_outside_pool': readouts['primary_no_mgi/all']['associations_outside_pool'] + readouts['primary_no_mgi/all']['associations_ranked'] == 425,
        'calibration_on_full_population_before_mask': readouts['primary_no_mgi/s_supported']['calibrated']['answer_micro'] == expected,
        'not_recalibrated_after_filtering': expected != recalibrated,
        'full_pool_equals_test_eval_metrics': True,
    }
    return {'passed': all(checks.values()), 'checks': checks, 'primary_all': {k: readouts['primary_no_mgi/all'][k] for k in ('associations_ranked', 'associations_outside_pool', 'cases_ranked')},
            'note': 'synthetic Gaussian scores; tests the readout arithmetic only'}


def export_terminal_check(scratch, python):
    """export_terminal.py on a tiny real torch full state: model weights only, bound to the source state; refuses a changed state."""
    training = scratch / 'export-terminal' / 'training'
    (training / 'checkpoints').mkdir(parents=True)
    script = ('import torch, sys\n'
              'torch.save({"step": 20000, "identity_sha256": "ID", "model": {"w": torch.arange(6.).reshape(2, 3)}, "optimizer": {"state": {0: torch.ones(3)}}}, sys.argv[1])\n')
    subprocess.run([python, '-c', script, str(training / 'checkpoints' / 'step-020000-1.pt')], check=True)
    state = training / 'checkpoints' / 'step-020000-1.pt'
    write_replace(training / 'COMPLETE.json', {'identity_sha256': 'ID', 'steps': 20000, 'terminal_checkpoint': {'checkpoint': 'checkpoints/step-020000-1.pt', 'checkpoint_sha256': sha(state), 'step': 20000}})
    first = call([python, '-I', HERE / 'export_terminal.py', '--training-dir', training, '--output', scratch / 'export-terminal' / 'terminal-model.pt'])
    inspect = call([python, '-c', 'import torch, sys, json; s = torch.load(sys.argv[1], weights_only=False); print(json.dumps({"keys": sorted(s), "model": sorted(s["model"]), "step": s["step"]}))',
                    scratch / 'export-terminal' / 'terminal-model.pt'])
    content = json.loads(inspect['stdout']) if inspect['returncode'] == 0 else {}
    state.write_bytes(b'changed')
    refused = call([python, '-I', HERE / 'export_terminal.py', '--training-dir', training, '--output', scratch / 'export-terminal' / 'other.pt'])
    return {'passed': first['returncode'] == 0 and content.get('model') == ['w'] and 'optimizer' not in content.get('keys', []) and content.get('step') == 20000 and refused['returncode'] != 0,
            'content': content, 'stderr': first['stderr'][-500:]}


def utilization_check(controller, policy, folder):
    import datetime
    results = {}
    for label, percent in (('busy', 95), ('low', 40)):
        target = folder / label
        target.mkdir(parents=True)
        start = 1_800_000_000.0
        with (target / 'gpu-telemetry.csv').open('w') as stream:
            for index in range(int(2 * 3600 / policy['sample_interval_seconds'])):
                stamp = datetime.datetime.fromtimestamp(start + index * policy['sample_interval_seconds']).strftime('%Y/%m/%d %H:%M:%S.000')
                stream.write('%s, GPU-x, %d %%, 1000 MiB\n' % (stamp, percent))
        guard = controller.Utilization(target / 'gpu-telemetry.csv', policy)
        stop = None
        for minute in range(1, 120):
            stop = guard.judge(start + minute * 60, start, target)
            if stop:
                stop = {'minute': minute, 'reason': stop['reason']}
                break
        results[label] = stop
    results['passed'] = results['busy'] is None and results['low'] is not None
    return results


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--readout-check':
        print(json.dumps(readout_check_body(Path(sys.argv[2]), sys.executable), default=str))
        return 0
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--python', required=True, help='local interpreter with torch and numpy for the workers\' builders and the readout')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    scratch = Path(args.scratch).resolve()
    assert not scratch.exists() or not any(scratch.iterdir())
    scratch.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    decision = REPO / 'notes/experiment-roadmap-20261123/execution/SELECTION-DECISION.json'
    steps, checks = {}, {}
    plan = U.plan()
    checks['plan_109_185'] = len(plan['pretrain']) == 109 and len(plan['query']) == 185
    prod = scratch / 'prod'
    refusal = scratch / 'synthetic-hpo'
    refusal.mkdir(parents=True)
    write_replace(refusal / 'FETCH-RECEIPT.json', {'files': {}, 'synthetic': True})
    shutil.copy(str(decision), str(refusal / 'selection-decision.json'))
    result = call([py, HERE / 'freeze.py', 'check-hpo', '--hpo-dir', refusal, '--user-decision', decision, '--output', scratch / 'refused-check.json'])
    checks['synthetic_hpo_refused'] = result['returncode'] != 0 and not (scratch / 'refused-check.json').exists()
    query_v1 = SUITE / 'query/protocols/suite-query-v1.json'
    v1 = read_json(query_v1)
    recipe = {'common_leaderboard': 'calibrated', 'initializer_recipe': v1['hpo_initializer_recipes'][0], 'learning_rate': v1['query_learning_rates'][1],
              'temperature': v1['temperatures'][1], 'source_trajectory': 'SYNTHETIC', 'source_selected_step': 0}
    write_replace(scratch / 'synthetic-freeze.json', {'command': 'freeze', 'test_access': False, 'synthetic': True, 'protocol_sha256': sha(query_v1), 'decision': read_json(decision),
                                                      'decision_sha256': sha(decision), 'production_recipe': recipe})
    out = prod / 'protocols'
    generate = [args.python, SUITE / 'query/code/make_protocol.py', 'generate', '--query-protocol', query_v1, '--pretrain-protocol', SUITE / 'pretrain/protocols/suite-pretrain-v1.json',
                '--freeze', scratch / 'synthetic-freeze.json', '--decision', decision, '--manifest', V3 / 'MANIFEST-final-20261123.json', '--root', REPO, '--draft',
                '--output-query', out / 'draft-query-v2.json', '--output-pretrain', out / 'draft-pretrain-v2.json']
    for modality in ('F', 'S', 'FS'):
        generate += ['--excluded-entry', V3 / ('excluded/entries/excluded-%s.json' % modality)]
    result = call(generate)
    assert result['returncode'] == 0, result['stderr']
    summary = json.loads(result['stdout'][result['stdout'].index('{'):])
    checks['make_protocol_draft_all_entries'] = summary['status'] == 'DRAFT' and summary['missing_entries'] == 0 and len(summary['banks']) == 11 and summary['graphs'] == 73
    for name, target in (('draft-query-v2.json', 'suite-query-v2.json'), ('draft-pretrain-v2.json', 'suite-pretrain-v2.json')):
        value = read_json(out / name)
        value.update(status='FROZEN', rehearsal_relabelled_from_draft=True)
        write_replace(out / target, value)
    production = read_json(out / 'suite-query-v2.json')['production_recipe']
    write_replace(out / 'PROTOCOLS.json', {'query_v2_sha256': sha(out / 'suite-query-v2.json'), 'pretrain_v2_sha256': sha(out / 'suite-pretrain-v2.json'),
                                           'hpo_check_sha256': 'REHEARSAL', 'production_recipe': production, 'rehearsal': True})
    freeze = module('rehearse_freeze', HERE / 'freeze.py')
    hpo, evidence, winner, screen = synthetic_hpo_campaign(scratch, prod, production, freeze, args.python)
    write_replace(prod / 'hpo-check.json', {'passed': False, 'rehearsal': True, 'synthetic': True, 'hpo_campaign': str(hpo)})
    result = call([py, HERE / 'freeze.py', 'reuse-audit', '--prod', prod, '--source-root', REPO, '--python', args.python, '--evidence-dir', evidence,
                   '--recipe', evidence / 'recipe.json', '--hpo-manifest', evidence / 'frozen-manifest.json'])
    steps['reuse_audit'] = result
    reuse = read_json(prod / 'REUSE.json')
    checks['reuse_audit_passes_exact_artifact_and_trajectory'] = reuse['pretrain']['passed'] and reuse['query']['passed']
    inputs, _ = freeze.input_records(REPO)
    base = {'recipe': read_json(evidence / 'recipe.json'), 'hpo_manifest': read_json(evidence / 'frozen-manifest.json'), 'admission': read_json(evidence / 'ADMISSION.json'),
            'admission_sha256': sha(evidence / 'ADMISSION.json'), 'identity': read_json(evidence / 'identity.json'), 'identity_sha256': sha(evidence / 'identity.json'),
            'config': read_json(evidence / 'config.json'), 'config_sha256': sha(evidence / 'config.json'), 'export_name': 'update-%07d.pt' % production['initializer_recipe']['initializer_update_budget']}
    fixed = read_json(out / 'suite-pretrain-v2.json')['fixed']
    rejections = {}
    for label, change in (('foreign_graph', ('data_complete_sha256', '0' * 64)), ('seed_1', ('seed', 1)), ('other_learning_rate', ('learning_rate', 0.3))):
        identity = json.loads(json.dumps(base['identity']))
        identity['config'][change[0]] = change[1]
        if change[0] == 'data_complete_sha256':
            identity['data_complete_sha256'] = change[1]
        rejections[label] = R.audit_pretrain(dict(base, identity=identity), inputs, production, fixed)['passed'] is False
    other_recipe = dict(production, initializer_recipe=dict(production['initializer_recipe'], initializer_update_budget=1055750))
    rejections['other_budget'] = R.audit_pretrain(base, inputs, other_recipe, fixed)['passed'] is False
    expected, sources, references = freeze.query_template(argparse.Namespace(python=args.python), prod, REPO, reuse['pretrain']['export'])
    identity = read_json(evidence / 'trajectory-identity.json')
    for label, key, value in (('encoder_mean', 'encoder', 'mean'), ('seed_1', 'seed', 1)):
        changed = json.loads(json.dumps(identity))
        changed['config'][key] = value
        rejections['query_' + label] = R.audit_query(changed, {'identity_sha256': canonical(changed), 'steps': 20000}, reuse['pretrain'], expected, sources, references)['passed'] is False
    changed = json.loads(json.dumps(identity))
    changed['config']['source_hashes'] = dict(changed['config']['source_hashes'], **{'sampler.py': '0' * 64})
    rejections['query_core_source'] = R.audit_query(changed, {'identity_sha256': canonical(changed), 'steps': 20000}, reuse['pretrain'], expected, sources, references)['passed'] is False
    steps['reuse_rejections'] = rejections
    checks['reuse_rejects_foreign_graph_seed_recipe_and_query_changes'] = all(rejections.values())
    result = call([py, HERE / 'freeze.py', 'bundle', '--prod', prod, '--source-root', REPO, '--revision', 'rehearsal', '--rehearsal'])
    assert result['returncode'] == 0, result['stderr']
    bundle = read_json(prod / 'bundle' / 'BUNDLE.json')
    checks['bundle_counts_reuse_separately'] = bundle['counts']['pretrain'] == 109 and bundle['counts']['query'] == 185 and sorted(bundle['counts']['reused_units_counted_separately']) == sorted([R.PRETRAIN_UNIT, R.QUERY_UNIT])
    bin_dir = scratch / 'bin'
    bin_dir.mkdir()
    for name in ('sbatch', 'squeue', 'sacct', 'sinfo', 'scontrol', 'sacctmgr', 'storage_usage', 'scancel'):
        (bin_dir / name).write_text(FAKE_SLURM)
        os.chmod(str(bin_dir / name), 0o755)
    (bin_dir / 'fake-ssh').write_text(FAKE_SSH)
    os.chmod(str(bin_dir / 'fake-ssh'), 0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'], FAKE_SLURM_STATE=str(scratch / 'fake-slurm.json'))
    campaigns = {}
    for site in ('orix', 'ibex'):
        campaign = scratch / ('campaign-' + site)
        write_replace(scratch / ('deploy-%s.json' % site), {'schema': 'clmpt-suite-production-deploy/1', 'site': site, 'campaign': str(campaign), 'source_checkout': str(REPO),
                                                             'inputs_root': str(V3), 'worker_python': args.python, 'references_dir': str(SUITE / 'query' / 'references')})
        result = call([py, HERE / 'campaign.py', 'stage', '--campaign', campaign, '--bundle', prod / 'bundle' / 'BUNDLE.json', '--deploy', scratch / ('deploy-%s.json' % site)], env)
        assert result['returncode'] == 0, result['stderr']
        checks['real_release_refused_%s' % site] = call([py, HERE / 'campaign.py', 'release', '--campaign', campaign, '--authority', 'r', '--approve'], env)['returncode'] != 0
        result = call([py, HERE / 'campaign.py', 'release', '--campaign', campaign, '--authority', 'rehearsal', '--approve', '--rehearsal-fake-slurm'], env)
        assert result['returncode'] == 0, result['stderr']
        campaigns[site] = campaign
    orix, ibex = campaigns['orix'], campaigns['ibex']
    tick = json.loads(call([py, HERE / 'campaign.py', 'tick', '--campaign', orix], env)['stdout'])
    submitted = [row['unit'] for row in tick.get('submitted', [])]
    steps['orix_tick_1'] = {'submitted': submitted, 'states': tick.get('states')}
    checks['reused_parent_never_trained'] = R.PRETRAIN_UNIT not in submitted and tick['states'].get('reused') == 1 and len(submitted) == 8 and submitted[0] == 'main-PFS-f0-ps1'
    result = call([py, HERE / 'campaign.py', 'tick', '--campaign', ibex], env)
    tick = json.loads(result['stdout'])
    steps['ibex_tick_1'] = {'returncode': result['returncode'], 'submitted': tick.get('submitted'), 'states': tick.get('states'), 'test_states': tick.get('test_states'),
                            'reuse_admitted': tick.get('reuse_admitted'), 'stderr': result['stderr'][-1500:]}
    reused_children = sorted(unit for unit, record in plan['query'].items() if record['parent'] == R.PRETRAIN_UNIT and unit != R.QUERY_UNIT)
    training_jobs = sorted(row['unit'] for row in tick.get('submitted', []) if not row.get('inference'))
    inference_jobs = sorted(row['unit'] for row in tick.get('submitted', []) if row.get('inference'))
    admitted = read_json(ibex / 'initializers' / R.PRETRAIN_UNIT / 'ADMITTED.json')
    child = read_json(ibex / 'units' / reused_children[0] / 'config.json')
    checks['frozen_artifact_admitted_and_children_built_on_it'] = (admitted['kind'] == 'FROZEN_HPO_ARTIFACT' and training_jobs == reused_children
                                                                   and child['pretrained_sha256'] == reuse['pretrain']['export']['sha256'] and child['pretrained_checkpoint'] == reuse['pretrain']['export']['ibex_path'])
    checks['reused_trajectory_released_for_test_and_untouched'] = inference_jobs == [R.QUERY_UNIT] and sha(winner / 'COMPLETE.json') == reuse['query']['complete_sha256'] \
        and (winner / 'checkpoints' / 'step-020000-1.pt').exists()
    S = module('rehearse_campaign', HERE / 'campaign.py')
    context = S.load(str(ibex))
    unreleased = dict(context, bundle=dict(context['bundle'], rehearsal=False, hpo_check={'passed': False}), release=dict(context['release'], rehearsal=False))
    checks['no_test_before_hpo_freeze'] = S.test_gate(unreleased, R.QUERY_UNIT, 'complete')[0] is False
    checks['no_test_before_training_complete'] = S.test_gate(context, reused_children[0], 'active')[0] is False
    release = read_json(ibex / 'units' / R.QUERY_UNIT / 'inference' / 'TEST-RELEASE.json')
    test_config = read_json(ibex / 'units' / R.QUERY_UNIT / 'inference' / 'test-config.json')
    checks['test_release_pins_unit_and_test'] = (release['status'] == 'RELEASED' and release['training_complete_sha256'] == [reuse['query']['complete_sha256']]
                                                 and test_config['views'] == ['initial', 'raw', 'calibrated', 'terminal'] and test_config['release_sha256'] == sha(ibex / 'units' / R.QUERY_UNIT / 'inference' / 'TEST-RELEASE.json'))
    inference_job = [job for job in fake_state(scratch)['jobs'].values() if job['name'].endswith('test-' + R.QUERY_UNIT)]
    checks['inference_job_short_and_counted'] = len(inference_job) == 1 and '--time=01:00:00' in inference_job[0]['argv']
    parent = 'main-PFS-f0-ps1'
    fake_pretrain_output(orix, parent)
    set_job(scratch, parent, state='COMPLETED', final='COMPLETED')
    block = scratch / 'BLOCK-TRANSFERS'
    (bin_dir / 'fake-rsync-ssh').write_text('#!/bin/sh\nwhile [ -f %s ]; do sleep 1; done\nshift\nexec sh -c "$*"\n' % block)
    os.chmod(str(bin_dir / 'fake-rsync-ssh'), 0o755)
    bridge_config = {'prod_dir': str(prod), 'state_dir': str(scratch / 'bridge-state'), 'transit_dir': str(scratch / 'transit'), 'transport_attempts': 1,
                     'max_delivery_workers': 2, 'max_archive_workers': 2, 'transfer_attempt_limit': 2, 'transfer_backoff_seconds': 0, 'drain_poll_seconds': 1,
                     'hpo': {'campaign': str(hpo)}}
    for site in ('orix', 'ibex'):
        bridge_config[site] = {'ssh': [str(bin_dir / 'fake-ssh'), 'localhost'], 'rsync_shell': str(bin_dir / 'fake-rsync-ssh'), 'host': 'localhost', 'campaign': str(campaigns[site]),
                               'code_dir': str(HERE), 'python': py}
    bridge_config['orix']['archive_root'] = str(scratch / 'orix-archive')
    write_replace(scratch / 'bridge.json', bridge_config)
    result = call([py, HERE / 'bridge.py', 'drain', '--config', scratch / 'bridge.json', '--timeout', '600'], env)
    steps['bridge_drain_1'] = {'returncode': result['returncode'], 'stdout': result['stdout'][-1500:], 'stderr': result['stderr'][-1500:]}
    children = sorted(unit for unit, record in plan['query'].items() if record['parent'] == parent)
    submitted_children = sorted(child for child in plan['query'] if any(job['name'] == 'lgq-' + child for job in fake_state(scratch)['jobs'].values()) and
                                plan['query'][child]['parent'] == parent)
    checks['delivery_admitted_and_only_own_children_released'] = (ibex / 'initializers' / parent / 'ADMITTED.json').exists() and submitted_children == children \
        and not any(job['name'] == 'lgq-' + child for job in fake_state(scratch)['jobs'].values() for child in plan['query']
                    if plan['query'][child]['parent'] not in (parent, R.PRETRAIN_UNIT))
    steps['transfers'] = transfer_scenario(scratch, orix, ibex, env, py, block, plan)
    checks.update({'transfers_' + key: value for key, value in steps['transfers']['checks'].items()})
    partial = ibex / 'incoming' / '.partial-main-PFS-f1-ps0'
    partial.mkdir(parents=True)
    write_replace(partial / 'DELIVERY.json', {'files': {'COMPLETE.json': '0' * 64}})
    checks['corrupt_delivery_refused'] = call([py, HERE / 'campaign.py', 'admit', '--campaign', ibex, '--parent', 'main-PFS-f1-ps0'], env)['returncode'] != 0
    unit = children[0]
    complete_query_unit(ibex, unit)
    set_job(scratch, unit, state='COMPLETED')
    tick = json.loads(call([py, HERE / 'campaign.py', 'tick', '--campaign', ibex], env)['stdout'])
    inventory = json.loads(call([py, HERE / 'campaign.py', 'inventory', '--campaign', ibex], env)['stdout'])
    checks['test_released_for_completed_unit_inside_cap'] = any(row.get('inference') and row['unit'] == unit for row in tick.get('submitted', []))
    checks['not_archivable_before_inference'] = unit not in inventory['archivable'] and R.QUERY_UNIT not in inventory['archivable']
    for done in (unit, R.QUERY_UNIT):
        fake_inference(ibex, done)
        set_job(scratch, 'test-' + done, state='COMPLETED')
    result = call([py, HERE / 'bridge.py', 'drain', '--config', scratch / 'bridge.json', '--timeout', '600'], env)
    steps['bridge_drain_2'] = {'returncode': result['returncode'], 'stdout': result['stdout'][-1500:], 'stderr': result['stderr'][-1500:]}
    tick = json.loads(call([py, HERE / 'campaign.py', 'tick', '--campaign', ibex], env)['stdout'])
    training = ibex / 'units' / unit / 'training'
    pruned_receipt = read_json(ibex / 'state' / 'pruned' / (unit + '.json'), {})
    archived = read_json(scratch / 'orix-archive' / unit / 'ARCHIVED.json', {})
    checks['archive_after_training_and_inference_then_prune'] = (bool(archived) and pruned_receipt.get('archive_receipt_sha256') == sha(ibex / 'state' / 'archive' / (unit + '.json'))
                                                                 and pruned_receipt.get('time', 0) >= archived.get('time', 1e18) and pruned_receipt.get('reportables_released') is False
                                                                 and any(name.startswith('inference/') for name in archived.get('files', {}))
                                                                 and not (training / 'checkpoints' / 'step-020000-1.pt').exists() and (training / 'models' / 'step-001500.pt').exists()
                                                                 and (training / 'fits' / 'fit-step-000000.npz').exists() and (ibex / 'units' / unit / 'inference' / 'terminal-model.pt').exists())
    checks['reused_trajectory_archived_never_pruned'] = (scratch / 'orix-archive' / R.QUERY_UNIT / 'ARCHIVED.json').exists() and (winner / 'checkpoints' / 'step-020000-1.pt').exists()
    steps['readout'] = readout_check(scratch, args.python)
    checks['two_pool_s_mask_readout_from_one_matrix'] = steps['readout']['passed']
    os.environ.update(PATH=env['PATH'], FAKE_SLURM_STATE=env['FAKE_SLURM_STATE'])
    malformed = scratch / 'bad-decision.json'
    write_replace(malformed, {'release_new_production_reportables_after_verified_archive': True})
    checks['malformed_storage_decision_refused'] = call([py, HERE / 'campaign.py', 'record-storage-decision', '--campaign', ibex, '--decision', malformed, '--authority', 'r'], env)['returncode'] != 0
    checks['hpo_release_refused_without_decision'] = call([py, HERE / 'campaign.py', 'release-hpo-copies', '--campaign', ibex, '--hpo-campaign', hpo, '--authority', 'r', '--dry-run'], env)['returncode'] != 0
    context = S.load(str(ibex))
    states = S.classify_all(context, S.user_jobs(), mutate=False)
    tiny = {'free': 10, 'reserve': 0, 'owed': 0, 'peak': 10 ** 12}
    idle = {key: ('ready' if value in ('active', 'buildable', 'ready') else value) for key, value in states.items()}
    _, skipped = S.schedule(dict(context, release=dict(context['release'], max_active=10 ** 6)), idle, {}, {}, tiny, True)
    checks['storage_permission_attention_without_decision'] = bool(skipped.get('storage_permission_required'))
    valid = scratch / 'storage-decision.json'
    write_replace(valid, {'schema': 'clmpt-suite-storage-decision/1', 'release_new_production_reportables_after_verified_archive': False, 'archive_current_hpo_reportables': True,
                          'release_current_hpo_reportables_after_verified_archive': True, 'decided_by': 'REHEARSAL ONLY', 'date': '2026-11-23'})
    checks['storage_decision_recorded'] = call([py, HERE / 'campaign.py', 'record-storage-decision', '--campaign', ibex, '--decision', valid, '--authority', 'rehearsal'], env)['returncode'] == 0
    result = call([py, HERE / 'bridge.py', 'drain', '--config', scratch / 'bridge.json', '--timeout', '600'], env)
    steps['hpo_archive'] = {'returncode': result['returncode'], 'transfers': {key: value for key, value in Bridge(scratch / 'bridge.json').transfers_overview().items() if key.startswith('hpo-archive')}}
    checks['hpo_campaign_archived_to_orix'] = (scratch / 'orix-archive' / 'hpo-campaign' / 'campaign-records' / 'ARCHIVED.json').exists() and \
        (scratch / 'orix-archive' / 'hpo-campaign' / 'C-C-winner' / 'ARCHIVED.json').exists()
    result = call([py, HERE / 'campaign.py', 'release-hpo-copies', '--campaign', ibex, '--hpo-campaign', hpo, '--authority', 'rehearsal', '--dry-run'], env)
    plan_release = json.loads(result['stdout']) if result['returncode'] == 0 else {}
    steps['hpo_release_dry_run'] = plan_release or result
    checks['hpo_release_dry_run_protects_reused'] = result['returncode'] == 0 and str(winner) in plan_release.get('protected_kept', []) and str(screen) in plan_release.get('protected_kept', []) \
        and (winner / 'models').exists() and (screen / 'models' / 'step-000000.pt').exists()
    checks['hpo_release_dry_run_lists_unprotected_and_deletes_nothing'] = plan_release.get('would_remove', {}).get('S-S-other', 0) > 0 and \
        (hpo / 'trials' / 'S' / 'S-other' / 'training' / 'models' / 'step-001500.pt').exists()
    steps['export_terminal'] = export_terminal_check(scratch, args.python)
    checks['terminal_model_only_export'] = steps['export_terminal']['passed']
    checks['refuses_other_hpo_campaign'] = call([py, HERE / 'campaign.py', 'release-hpo-copies', '--campaign', ibex, '--hpo-campaign', scratch, '--authority', 'r', '--dry-run'], env)['returncode'] != 0

    def signal_candidate(name, final):
        job = next(job_id for job_id, job in fake_state(scratch)['jobs'].items() if job['name'].endswith(name))
        wave = orix / 'units' / name / 'allocations' / 'wave-000'
        wave.mkdir(parents=True)
        write_replace(wave / 'allocation.json', {'job': job})
        write_replace(wave / 'SIGNAL_PAUSE_CANDIDATE.json', {'job': job, 'step': 1000, 'before_step': 0, 'reason': 'signal', 'time': time.time() - 300})
        set_job(scratch, name, state=final, final=final)
    preempted, cancelled, failed = 'main-PFS-f0-ps2', 'main-PFS-f0-ps3', 'main-PFS-f0-ps4'
    signal_candidate(preempted, 'PREEMPTED')
    tick = json.loads(call([py, HERE / 'campaign.py', 'tick', '--campaign', orix], env)['stdout'])
    steps['preempted_tick'] = {'decisions': tick.get('preemption_decisions'), 'status': tick.get('status'), 'submitted': tick.get('submitted')}
    resubmitted = [job for job in fake_state(scratch)['jobs'].values() if job['name'].endswith(preempted) and job['argv'][-1] == '1']
    checks['verified_preemption_resumed_without_dependency'] = ((orix / 'units' / preempted / 'allocations' / 'wave-000' / 'PREEMPTION_VERIFIED.json').exists()
                                                                and len(resubmitted) == 1 and not any(part.startswith('--dependency') for part in resubmitted[0]['argv']))
    signal_candidate(cancelled, 'CANCELLED')
    tick = json.loads(call([py, HERE / 'campaign.py', 'tick', '--campaign', orix], env)['stdout'])
    steps['cancelled_tick'] = {'decisions': tick.get('preemption_decisions'), 'status': tick.get('status'), 'reason': tick.get('reason')}
    checks['cancelled_is_attention_never_revived'] = ((orix / 'units' / cancelled / 'allocations' / 'wave-000' / 'SIGNAL_NOT_PREEMPTION.json').exists() and tick.get('status') == 'halted'
                                                      and not any(job['name'].endswith(cancelled) and job['argv'][-1] == '1' for job in fake_state(scratch)['jobs'].values()))
    wave = orix / 'units' / failed / 'allocations' / 'wave-000'
    wave.mkdir(parents=True)
    write_replace(wave / 'ATTENTION.json', {'error': "AssertionError('worker error (return code 1): no automatic retry')"})
    set_job(scratch, failed, state='FAILED', final='FAILED')
    O = module('rehearse_campaign_orix', HERE / 'campaign.py')
    checks['worker_error_is_attention_never_revived'] = O.classify(O.load(str(orix)), failed, O.user_jobs()) == 'attention' and \
        not any(job['name'].endswith(failed) and job['argv'][-1] == '1' for job in fake_state(scratch)['jobs'].values())
    controller = module('rehearse_controller', HERE / 'controller.py')
    policy = read_json(HERE / 'site-policy.json')
    steps['deadlines'] = {'orix': controller.deadlines('orix', policy['orix'], 10800)[:2], 'ibex': controller.deadlines('ibex', policy['ibex'], 10800)}
    checks['deadlines'] = tuple(steps['deadlines']['orix']) == (10200, 10620) and tuple(steps['deadlines']['ibex'][:2]) == (13200, 13500)
    steps['utilization'] = utilization_check(controller, policy['orix']['utilization'], scratch / 'utilization')
    checks['utilization_guard'] = steps['utilization']['passed']
    receipt = {'checks': checks, 'passed': all(checks.values()), 'steps': steps, 'scratch': str(scratch), 'time': time.time(),
               'scope': 'CPU rehearsal with fake Slurm/SSH and synthetic HPO files, trajectories, exports and scores on the real v3 manifests; no training, no TEST scoring, no GPU, no remote host'}
    write_replace(Path(args.output), receipt)
    print(json.dumps({'passed': receipt['passed'], 'checks': checks}, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
