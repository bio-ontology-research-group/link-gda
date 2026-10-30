#!/usr/bin/env python3
"""Score one fixed campaign after the original coordinator freezes all 240 runs."""
import argparse
import fcntl
import importlib.util
from pathlib import Path
import subprocess
import sys


def load_coordinator(source):
    code = source/'code'
    sys.path[:0] = [str(code), str(code/'baselines')]
    spec = importlib.util.spec_from_file_location('frozen_controlled_coordinator',
                                                code/'baselines/run_controlled_query.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_plan(args, coordinator):
    names = [f'{i}-seed{s}' for i in coordinator.INITS for s in range(3)]
    for name in names:
        child = args.root/name
        coordinator.verify_source(child, args.source_root/'code')
        for part in ('data', 'projection'):
            if (child/part).resolve(strict=True) != (args.prepared_root/part).resolve(strict=True):
                raise ValueError('Prepared inputs differ: ' + str(child/part))
    expected = dict(campaigns=names, expected_runs=240, runs_per_campaign=40,
        arms=coordinator.ARMS, seeds=[0,1,2], initializations=coordinator.INITS,
        protocol_sha256=coordinator.sha(args.protocol),
        source_sha256=coordinator.sha(args.root/names[0]/'source.sha256'))
    if coordinator.read(args.root/'CONTROLLED.json') != expected:
        raise ValueError('Scoring requires the complete fixed controlled design')
    coordinator.verify_gates(args.root)


def freeze(args, coordinator, worker, shared):
    # The child inherits both locks if the supervisor exits before its child.
    with (args.root/'scoring-freeze.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        validate_plan(args, coordinator)
        path = args.root/'GLOBAL-SELECTION.json'
        if not path.exists():
            if any(args.root.glob('*-seed*/test/fold*/*/COMPLETE.json')):
                raise ValueError('Test outputs exist without their original global freeze')
            subprocess.run([sys.executable, str(args.source_root/'code/baselines/run_controlled_query.py'),
                '--root',str(args.root),'--prepared-root',str(args.prepared_root),
                '--base',str(args.base),'--ultra-root',str(args.ultra_root),'--phase','freeze'],
                check=True, pass_fds=(worker.fileno(),lock.fileno(),shared.fileno()))
        digest = coordinator.sha(path)
        from phenoquery.campaign import verify_global
        for name in coordinator.read(args.root/'CONTROLLED.json')['campaigns']:
            verify_global(args.root/name, digest)
        return digest


def configs(args, coordinator, digest):
    name = f'{args.initialization}-seed{args.seed}'
    child = coordinator.child_args(args, name, args.initialization, args.seed)
    for fold in range(10):
        yield child, dict(phase='score', fold=fold, arms=coordinator.ARMS,
            seed=args.seed, initialization=args.initialization, score_initial=True,
            protocol_sha256=coordinator.sha(child.root/'PROTOCOL.md'),
            selection_sha256=coordinator.sha(child.root/'SELECTION.json'),
            global_selection_sha256=digest)


def invoke_score(coordinator, child, config, worker, shared):
    path = child.root/'configs'/f"score-fold{config['fold']}.json"
    coordinator.immutable(path, config)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run(coordinator.command(child,path),check=True,
                       pass_fds=(worker.fileno(),lock.fileno(),shared.fileno()))


def run(args):
    coordinator = load_coordinator(args.source_root)
    name = f'{args.initialization}-seed{args.seed}'
    # Also conflict with the historical serial scorer, which owns scoring.lock.
    with (args.root.parent/'scoring.lock').open('a') as shared:
        fcntl.flock(shared, fcntl.LOCK_SH | fcntl.LOCK_NB)
        workers = args.root/'scoring-workers'; workers.mkdir(exist_ok=True)
        with (workers/(name+'.lock')).open('a') as worker:
            fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
            digest = freeze(args,coordinator,worker,shared)
            completed = []
            from phenoquery.campaign import verify_global
            for child, config in configs(args,coordinator,digest):
                verify_global(child.root,digest)
                invoke_score(coordinator,child,config,worker,shared)
                verify_global(child.root,digest)
                folder = child.root/'test'/f"fold{config['fold']}"
                record = coordinator.read(folder/'COMPLETE.json')
                expected = set(coordinator.ARMS) | {a+'_frozen' for a in coordinator.ARMS}
                if (record['fold'] != config['fold'] or record['arms'] != coordinator.ARMS
                        or set(record['records']) != expected
                        or record['selection_sha256'] != config['selection_sha256']):
                    raise ValueError('Original scorer did not complete all eight fold bundles')
                completed.append(config['fold'])
                print('VERIFIED_CAMPAIGN_FOLD',name,config['fold'],flush=True)
            coordinator.write(workers/(name+'.json'),dict(campaign=name,folds=completed,
                score_bundles=80,global_selection_sha256=digest,
                launcher_sha256=coordinator.sha(__file__),
                scope='Original frozen inference commands; numerical replay remains a separate gate'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','root','prepared-root','base','ultra-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--initialization',choices=['ultra_4g','ultraquery'],required=True)
    parser.add_argument('--seed',type=int,choices=range(3),required=True)
    args = parser.parse_args()
    for name in ('source_root','root','prepared_root','base','ultra_root'):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    args.protocol = args.source_root/'reproducibility/protocols/controlled-query-20261028.md'
    args.device = 'cuda'
    run(args)


if __name__ == '__main__': main()
