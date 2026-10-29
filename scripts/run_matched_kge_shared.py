#!/usr/bin/env python3
"""Schedule independent matched-KGE trajectories on one allocated GPU.

Run outside the frozen scientific source. Training remains the default. Scoring
requires a complete, pre-existing global freeze and invokes the original trainer
for inference and artifact checks. All workers must use the shared scheduling
leases after legacy workers stop. Completion markers guide scheduling; the
trainer and final manifest own full scientific identity and artifact validation.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def tasks():
    for model in ('U1', 'T1', 'INDIGENA', 'CONVKB', 'PPOOL'):
        for fold in range(10):
            for graph in ('original', 'nobridge'):
                yield Path(f'fold{fold}') / graph / model


def frozen_training(root):
    """Check the full production freeze before any scoring child can start."""
    path = root / 'runs/FROZEN.json'
    raw = path.read_bytes()
    frozen = json.loads(raw)
    expected = {str(task) for task in tasks()}
    if (frozen.get('schema') != 1 or frozen.get('folds') != list(range(10))
            or frozen.get('models') != ['U1', 'T1', 'INDIGENA', 'CONVKB', 'PPOOL']
            or frozen.get('graphs') != ['original', 'nobridge']
            or set(frozen.get('training', {})) != expected):
        raise ValueError('Scoring requires the complete production global freeze')
    for task, digest in frozen['training'].items():
        if hashlib.sha256((root / 'runs' / task / 'TRAINED.json').read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen training record changed: ' + task)
    return hashlib.sha256(raw).hexdigest()


def claim(root, phase='train'):
    marker = 'TRAINED.json' if phase == 'train' else 'SCORED.json'
    leases = root / ('scheduling-leases' if phase == 'train' else 'scoring-leases')
    leases.mkdir(exist_ok=True)
    for task in tasks():
        run = root / 'runs' / task
        if (run / marker).exists():
            continue
        if phase == 'train' and task.name in ('CONVKB', 'PPOOL') and not (run.parent / 'U1/TRAINED.json').exists():
            continue
        lease = (leases / ('-'.join(task.parts) + '.lock')).open('a')
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            continue
        # A previous owner can finish between the first check and acquisition.
        if (run / marker).exists():
            lease.close()
            continue
        failure = leases / ('-'.join(task.parts) + '.failure.json')
        if failure.exists():
            lease.close()
            raise RuntimeError(f"Unresolved {'training' if phase == 'train' else 'scoring'} failure: {failure}")
        return task, lease
    return None


def command(args, task):
    fold, graph, model = task.parts
    return [sys.executable, '-u', str(args.source_root / 'code/training/run_matched_kge.py'),
            '--phase', getattr(args, 'phase', 'train'), '--root', str(args.campaign_root / 'runs'),
            '--data-root', str(args.data_root), '--fold', fold[4:],
            '--graph', graph, '--model', model,
            '--gene-phenotypes', str(args.campaign_root / 'inputs/gene_phenotypes.csv'),
            '--epochs', '1000', '--profile-epochs', '20', '--validation-frequency', '20',
            '--checkpoint-frequency', '20']


def write(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def run(args):
    root = args.campaign_root
    phase = getattr(args, 'phase', 'train')
    marker = 'TRAINED.json' if phase == 'train' else 'SCORED.json'
    freeze_digest = frozen_training(root) if phase == 'score' else None
    lease_dir = 'scheduling-leases' if phase == 'train' else 'scoring-leases'
    label = os.environ['SLURM_JOB_ID'] + '-restart' + os.environ.get('SLURM_RESTART_COUNT', '0')
    out = root / ('shared-workers' if phase == 'train' else 'shared-score-workers') / label
    out.mkdir(parents=True, exist_ok=False)
    active = []
    finished = []
    started = time.time()

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    write(out / 'execution.json', dict(started_unix=started, slots=args.slots,
          source_root=str(args.source_root), launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          phase=phase, freeze_sha256=freeze_digest,
          scientific_settings='Original frozen trainer and explicit original budgets'))
    try:
        with (out / 'gpu.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['unix_time', 'active_trajectories', 'gpu_uuid', 'utilization_percent', 'used_mib', 'total_mib'])
            while True:
                if freeze_digest is not None and frozen_training(root) != freeze_digest:
                    raise ValueError('Global freeze changed during scoring')
                for child in active[:]:
                    process, task, lease, log = child
                    status = process.poll()
                    if status is None:
                        continue
                    if status or not (root / 'runs' / task / marker).exists():
                        failure = dict(task=str(task), returncode=status, worker=label,
                                       error=f'Child failed or exited without {marker} completion')
                        write(root / lease_dir / ('-'.join(task.parts) + '.failure.json'), failure)
                        raise RuntimeError(str(failure))
                    finished.append(str(task))
                    write(out / 'finished.json', finished)
                    lease.close()
                    log.close()
                    active.remove(child)
                while len(active) < args.slots:
                    acquired = claim(root, phase)
                    if acquired is None:
                        break
                    task, lease = acquired
                    log = (out / ('-'.join(task.parts) + '.log')).open('w')
                    argv = command(args, task)
                    try:
                        # Keep the lease alive even if the supervisor is killed first.
                        process = subprocess.Popen(argv, cwd=args.source_root, stdout=log,
                                                   stderr=subprocess.STDOUT, pass_fds=(lease.fileno(),))
                    except BaseException:
                        log.close()
                        lease.close()
                        raise
                    active.append((process, task, lease, log))
                    write(out / ('-'.join(task.parts) + '.execution.json'), dict(command=argv, pid=process.pid))
                if not active:
                    write(out / 'completion.json', dict(status='no_unclaimed_work', finished=finished,
                          elapsed_seconds=time.time() - started,
                          scope='Worker ended; does not certify campaign completion'))
                    return
                observation = subprocess.check_output([
                    'nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
                    '--query-gpu=uuid,utilization.gpu,memory.used,memory.total',
                    '--format=csv,noheader,nounits'], text=True, timeout=20)
                rows = list(csv.reader(observation.splitlines()))
                if len(rows) != 1:
                    raise RuntimeError('Expected exactly one allocated GPU')
                writer.writerow([time.time(), len(active), *[v.strip() for v in rows[0]]])
                stream.flush()
                time.sleep(10)
    except BaseException as exc:
        write(out / 'failure.json', dict(error=repr(exc), elapsed_seconds=time.time() - started))
        raise
    finally:
        # Do not release leases until our children have actually exited.
        for process, _, _, _ in active:
            if process.poll() is None:
                process.terminate()
        for process, _, lease, log in active:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            lease.close()
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('train', 'score'), default='train')
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--slots', type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.campaign_root = args.campaign_root.resolve()
    args.data_root = args.data_root.resolve()
    run(args)
