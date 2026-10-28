#!/usr/bin/env python3
"""Prepare, gate, train and score the fixed 240-run PhenoQuery comparison.

The prepared root supplies data/ and projection/ from prepare_phenoquery.py.
Use plan to write all immutable configs. Individual configs can be dispatched
with run_phenoquery.py; freeze requires every validation selection to be complete.
"""
import argparse
import fcntl
import shutil
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from phenoquery.common import read, write, sha
from phenoquery.campaign import freeze_global, verify_source, verify_gates
from phenoquery.resume import verify_boundary_state
from run_query_campaign import initialize, eq_configs, eq_freeze, command

ARMS = ['bridge_pool', 'plain_pool', 'plain_and', 'typed_and']
INITS = ['ultra_4g', 'ultraquery']


def child_args(args, name, initialization, seed=0):
    return SimpleNamespace(root=args.root / name, base=args.base, ultra_root=args.ultra_root,
                           kind='eq', protocol=args.protocol, arms=ARMS, seed=seed,
                           initialization=initialization, steps=500, cohort='both', device=args.device)


def immutable(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Frozen configuration differs: {path}')
    write(path, value)


def prepare_child(args, child):
    initialize(child)
    for part in ['data', 'projection']:
        destination = child.root / part
        source = (args.prepared_root / part).resolve(strict=True)
        if destination.exists():
            if destination.resolve() != source:
                raise ValueError(f'Prepared inputs differ: {destination}')
        else:
            destination.symlink_to(source, target_is_directory=True)


def production_configs(child):
    return [dict(config, global_selection_required=True)
            for config in eq_configs(child.root, child.seed, child.initialization, 500, ARMS)]


def plan(args):
    campaigns = []
    for initialization in INITS:
        for seed in range(3):
            name = f'{initialization}-seed{seed}'
            child = child_args(args, name, initialization, seed)
            prepare_child(args, child)
            for config in production_configs(child):
                immutable(child.root / 'configs' / f"{config['arm']}-fold{config['fold']}.json", config)
            campaigns.append(name)
    immutable(args.root / 'CONTROLLED.json', {
        'campaigns': campaigns, 'expected_runs': 240, 'runs_per_campaign': 40,
        'arms': ARMS, 'seeds': [0, 1, 2], 'initializations': INITS,
        'protocol_sha256': sha(args.protocol),
        'source_sha256': sha(args.root / campaigns[0] / 'source.sha256')})


def invoke(child, config, stop_after=None, stop_before_validation=None):
    path = child.root / 'configs' / f"{config.get('arm', 'score')}-fold{config['fold']}.json"
    immutable(path, config)
    cmd = command(child, path)
    if stop_after is not None:
        cmd += ['--stop-after', str(stop_after)]
    if stop_before_validation is not None:
        cmd += ['--stop-before-validation', str(stop_before_validation)]
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run(cmd, check=True)


def compare_resume(first, second):
    import torch
    import numpy as np
    a = torch.load(first / 'resume.pt', map_location='cpu', weights_only=False)
    b = torch.load(second / 'resume.pt', map_location='cpu', weights_only=False)
    def equal(x, y):
        if isinstance(x, torch.Tensor):
            torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x: equal(x[k], y[k])
        elif isinstance(x, (tuple, list)):
            assert len(x) == len(y)
            for xx, yy in zip(x, y): equal(xx, yy)
        else:
            assert x == y, (x, y)
    for key in ['model', 'pool', 'optimizer', 'rng', 'torch_rng', 'cuda_rng', 'step', 'best', 'baseline', 'curve']:
        equal(a[key], b[key])
    for filename in ['initial-validation.npz', 'validation.npz']:
        with np.load(first / filename) as x, np.load(second / filename) as y:
            np.testing.assert_array_equal(x['genes'], y['genes'])
            np.testing.assert_array_equal(x['diseases'], y['diseases'])
            np.testing.assert_allclose(x['scores'], y['scores'], rtol=1e-5, atol=1e-6)
    for filename in ['initial.pt','best.pt']:
        equal(torch.load(first / filename,map_location='cpu',weights_only=False),
              torch.load(second / filename,map_location='cpu',weights_only=False))
    return {'steps': a['step'], 'selected_step': a['best']['step'],
            'max_parameter_error': max(float((a['model'][k] - b['model'][k]).abs().max()) for k in a['model']),
            'rtol': 1e-5, 'atol': 1e-6,
            'uninterrupted_resume_sha256': sha(first / 'resume.pt'),
            'resumed_resume_sha256': sha(second / 'resume.pt')}


def pilot(args, initialization):
    child = child_args(args, f'pilot-{initialization}', initialization)
    prepare_child(args, child)
    for arm in ['bridge_pool', 'plain_and', 'typed_and']:
        invoke(child, {'phase': 'train', 'arm': arm, 'fold': 0, 'pilot': True,
                       'steps': 2, 'seed': 0, 'initialization': initialization,
                       'protocol_sha256': sha(child.root / 'PROTOCOL.md')})
    paths = []
    for name, stopped in [('uninterrupted', False), ('resumed', True), ('validation-boundary', False)]:
        check = child_args(args, f'resume-{initialization}-{name}', initialization)
        prepare_child(args, check)
        config = {'phase': 'train', 'arm': 'typed_and', 'fold': 0, 'pilot': True,
                  'steps': 4, 'seed': 0, 'initialization': initialization,
                  'protocol_sha256': sha(check.root / 'PROTOCOL.md')}
        if stopped:
            invoke(check, config, stop_after=2)
        if name == 'validation-boundary':
            folder=check.root/'pilots'/'typed_and-fold0'
            before=folder/'before-validation.pt'
            if not before.exists():
                if (folder/'COMPLETE.json').exists():
                    raise ValueError('Completed boundary pilot lacks its interrupted-state evidence')
                invoke(check, config, stop_before_validation=4)
                temporary=before.with_suffix('.copy.tmp')
                shutil.copyfile(folder/'resume.pt',temporary)
                temporary.replace(before)
        invoke(check, config)
        paths.append(check.root / 'pilots' / 'typed_and-fold0')
    result = compare_resume(*paths[:2])
    import torch
    before=paths[2]/'before-validation.pt'
    after=paths[2]/'resume.pt'
    result['validation_boundary'] = verify_boundary_state(
        torch.load(before,map_location='cpu',weights_only=False),
        torch.load(after,map_location='cpu',weights_only=False))
    result['validation_boundary'].update(before_sha256=sha(before),after_sha256=sha(after))
    result.update(initialization=initialization, source_sha256=sha(child.root / 'source.sha256'))
    artifacts = {}
    for arm in ['bridge_pool', 'plain_and', 'typed_and']:
        for name in ['COMPLETE.json', 'parity.json', 'update-check.json']:
            path = child.root / 'pilots' / f'{arm}-fold0' / name
            artifacts[str(path.relative_to(args.root))] = sha(path)
    for folder in paths:
        for name in ['COMPLETE.json', 'resume.pt']:
            path = folder / name
            artifacts[str(path.relative_to(args.root))] = sha(path)
    artifacts[str(before.relative_to(args.root))]=sha(before)
    result['artifacts'] = artifacts
    write(args.root / f'GATE-{initialization}.json', result)


def require_gates(args):
    verify_gates(args.root)


def run(args):
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'planning.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan(args)
    if args.phase == 'plan': return
    if args.phase == 'pilot':
        for initialization in [args.initialization] if args.initialization else INITS:
            pilot(args, initialization)
        return
    require_gates(args)
    if args.phase in ['train', 'all']:
        for initialization in INITS:
            for seed in range(3):
                child = child_args(args, f'{initialization}-seed{seed}', initialization, seed)
                for config in production_configs(child): invoke(child, config)
    for initialization in INITS:
        for seed in range(3):
            child = child_args(args, f'{initialization}-seed{seed}', initialization, seed)
            verify_source(child.root, CODE)
            eq_freeze(child.root, production_configs(child))
    frozen = freeze_global(args.root)
    if args.phase in ['train', 'freeze']: return
    for initialization in INITS:
        for seed in range(3):
            child = child_args(args, f'{initialization}-seed{seed}', initialization, seed)
            for fold in range(10):
                invoke(child, {'phase': 'score', 'fold': fold, 'arms': ARMS,
                               'seed': seed, 'initialization': initialization, 'score_initial': True,
                               'protocol_sha256': sha(child.root / 'PROTOCOL.md'),
                               'selection_sha256': sha(child.root / 'SELECTION.json'),
                               'global_selection_sha256': frozen})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepared-root', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--ultra-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=CODE.parent / 'reproducibility/protocols/controlled-query-20261028.md')
    p.add_argument('--phase', choices=['plan', 'pilot', 'train', 'freeze', 'score', 'all'], required=True)
    p.add_argument('--initialization', choices=INITS, help='Run only this initialization for the pilot phase')
    p.add_argument('--device', default='cuda')
    run(p.parse_args())


if __name__ == '__main__': main()
