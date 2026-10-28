#!/usr/bin/env python3
"""Run validation-gated query experiments serially on one GPU.

Each invocation resumes verified completed runs and checkpointed trajectories.
Test scoring begins only after all required training selections are frozen.
For parallel execution, use the same per-run JSON files with the individual
run_phenoquery.py / run_ultra_matched.py entry points.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from phenoquery.common import ARMS, read, write, sha
from phenoquery import matched_selection
from phenoquery.campaign import source_text
from phenoquery.resume import require_complete_validation


def eq_configs(root, seed, initialization, steps, arms):
    return [{'phase': 'train', 'arm': arm, 'fold': fold, 'pilot': False,
             'steps': steps, 'seed': seed, 'initialization': initialization,
             'protocol_sha256': sha(root / 'PROTOCOL.md')}
            for fold in range(10) for arm in arms]


def eq_freeze(root, configs):
    runs = {}
    for config in configs:
        name = f"{config['arm']}-fold{config['fold']}"
        folder = root / 'training' / name
        completion = read(folder / 'COMPLETE.json')
        require_complete_validation(read(folder / 'progress.json'),config['steps'],completion['best'])
        identity = read(folder / 'identity.json')
        if identity['config'] != config or completion['checkpoint_sha256'] != sha(folder / 'best.pt'):
            raise ValueError(f'Configuration/checkpoint mismatch: {name}')
        if completion['source_manifest_sha256'] != sha(root / 'source.sha256'):
            raise ValueError(f'Source mismatch: {name}')
        runs[name] = {'completion_sha256': sha(folder / 'COMPLETE.json'),
                      'checkpoint_sha256': completion['checkpoint_sha256'], 'best': completion['best']}
    selection = {'runs': runs, 'protocol_sha256': sha(root / 'PROTOCOL.md'),
                 'selection_data': 'validation only; all arms retained'}
    path = root / 'SELECTION.json'
    if path.exists() and read(path) != selection:
        raise ValueError('Frozen selection changed')
    write(path, selection)
    return sha(path)


def initialize(args):
    args.root.mkdir(parents=True, exist_ok=True)
    protocol = args.protocol or CODE.parent / 'reproducibility/protocols' / (
        'eq-query-20261026.md' if args.kind == 'eq' else 'matched-query-20261027.md')
    destination = args.root / 'PROTOCOL.md'
    if destination.exists() and destination.read_bytes() != protocol.read_bytes():
        raise ValueError('Campaign protocol differs; pass its --protocol explicitly')
    destination.write_bytes(protocol.read_bytes())
    source = source_text(CODE)
    manifest = args.root / 'source.sha256'
    if manifest.exists() and manifest.read_text() != source:
        raise ValueError('Campaign source changed; use a new root and retain prior provenance')
    manifest.write_text(source)
    identity = {'seed': args.seed, 'initialization': args.initialization,
                'kind': args.kind, 'steps': args.steps, 'cohort': args.cohort, 'arms': args.arms,
                'protocol_sha256': sha(destination), 'source_sha256': sha(manifest)}
    old = args.root / 'campaign.json'
    if old.exists() and read(old) != identity:
        raise ValueError('Campaign configuration changed')
    write(old, identity)
    for name in ['configs', 'general', 'query']:
        (args.root / name).mkdir(exist_ok=True)


def command(args, config_path):
    driver = 'run_phenoquery.py' if args.kind == 'eq' else 'run_ultra_matched.py'
    return [sys.executable, str(CODE / 'baselines' / driver), '--root', str(args.root),
            '--base', str(args.base), '--ultra-root', str(args.ultra_root),
            '--config', str(config_path), '--device', args.device]


def execute(args, config):
    name = (f"{config.get('arm', 'score')}-fold{config['fold']}" if args.kind == 'eq'
            else config['cohort'] + '--' + config['name'])
    path = args.root / 'configs' / (name + '.json')
    if path.exists() and read(path) != config:
        raise ValueError(f'Run configuration changed: {path}')
    write(path, config)
    subprocess.run(command(args, path), check=True)


def run(args):
    initialize(args)
    with (args.root / 'campaign.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.kind == 'eq':
            configs = eq_configs(args.root, args.seed, args.initialization, args.steps, args.arms)
            if args.phase != 'score':
                for config in configs:
                    execute(args, config)
            selection = eq_freeze(args.root, configs)
            if args.phase != 'train':
                for fold in range(10):
                    execute(args, {'phase': 'score', 'fold': fold, 'seed': args.seed,
                                   'initialization': args.initialization, 'selection_sha256': selection, 'arms': args.arms,
                                   'protocol_sha256': sha(args.root / 'PROTOCOL.md')})
        else:
            matched_selection.SEED = args.seed
            for cohort in ['general', 'query'] if args.cohort == 'both' else [args.cohort]:
                while True:
                    tasks, phase = matched_selection.tasks(args.root, cohort)
                    if phase == 'complete':
                        break
                    pending = [c for c in tasks if not matched_selection.done(args.root, c).exists()]
                    if args.phase == 'train' and phase == 'test-scoring':
                        break
                    if args.phase == 'score' and phase != 'test-scoring':
                        raise ValueError('Training and validation selection must finish before test scoring')
                    for config in pending:
                        execute(args, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['eq', 'matched'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--ultra-root', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--initialization', choices=['ultra_4g', 'ultraquery'], default='ultra_4g')
    parser.add_argument('--cohort', choices=['general', 'query', 'both'], default='both')
    parser.add_argument('--steps', type=int, default=500, help='EQ training updates per fold/arm')
    parser.add_argument('--phase', choices=['train', 'score', 'all'], default='all')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.arms != ARMS and args.protocol is None:
        parser.error('A changed arm set requires an explicit --protocol')
    if args.steps != 500 and args.protocol is None:
        parser.error('A changed training budget requires its own --protocol')
    if (args.seed != 0 or args.initialization != 'ultra_4g') and args.protocol is None:
        parser.error('A new seed or initialization requires an explicit --protocol')
    run(args)

if __name__ == '__main__':
    main()
