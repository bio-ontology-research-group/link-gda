"""One short Slurm GPU job: released TEST readout of one completed query unit, before any pruning.

Steps, in order, all inside units/<unit>/inference/: (1) export_terminal.py writes the terminal model-only export from
the terminal full state; (2) srun the campaign copy of query/code/test_eval.py with the tick-written test config and the
per-unit TEST-RELEASE.json (views initial, raw, calibrated, terminal; it refuses unless the release names this run's
COMPLETE and this TEST COMPLETE); (3) readout.py adds the pool and label-mask readouts from the same saved matrices.
INFERENCE_COMPLETE.json then pins every output by hash. Any failure writes ATTENTION.json; nothing is retried.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import read_json, run, sha, write_exclusive, write_replace  # noqa: E402
import campaign as S  # noqa: E402


def main(campaign_argument, unit, attempt):
    context = S.load(campaign_argument)
    campaign, release, policy = context['campaign'], context['release'], context['policy']
    assert not release.get('rehearsal'), 'rehearsal release: no GPU work'
    folder = S.unit_dir(campaign, unit) / 'inference'
    plan = read_json(folder / 'INFERENCE-PLAN.json')
    for name, digest in plan['files'].items():
        assert sha(folder / name) == digest, 'inference input changed: %s' % name
    job = os.environ['SLURM_JOB_ID']
    attempt_dir = folder / ('attempt-%03d' % attempt)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    write_replace(attempt_dir / 'allocation.json', {'job': job, 'time': time.time(), 'host': os.uname().nodename, 'plan_sha256': sha(folder / 'INFERENCE-PLAN.json')})
    python = sys.executable
    try:
        export = run([python, '-I', str(HERE / 'export_terminal.py'), '--training-dir', plan['training_dir'], '--output', str(folder / 'terminal-model.pt')], 3000)
        write_replace(attempt_dir / 'export.json', export)
        assert export['returncode'] == 0, 'terminal export failed'
        test = folder / 'test'
        command = ['srun', '--exact', '--ntasks=1', '--cpus-per-task=%d' % policy['inference']['cpus_per_task'], '--gres=gpu:1', '--kill-on-bad-exit=1',
                   python, '-u', str(campaign / 'query' / 'code' / 'test_eval.py'), '--config', str(folder / 'test-config.json'), '--output', str(test)]
        scored = run(command, 3300)
        write_replace(attempt_dir / 'test-eval.json', scored)
        assert scored['returncode'] == 0 and (test / 'COMPLETE.json').exists(), 'test_eval failed'
        config = read_json(folder / 'test-config.json')
        readout = run([python, '-I', str(HERE / 'readout.py'), '--worker-code', str(campaign / 'query' / 'code'), '--inference-dir', str(test), '--test-dir', config['test_dir'],
                       '--bank-dir', config['bank_dir'], '--output', str(folder / 'readout.json')], 3000)
        write_replace(attempt_dir / 'readout.json', readout)
        assert readout['returncode'] == 0, 'readout failed'
        files = {'test/' + name: digest for name, digest in read_json(test / 'COMPLETE.json')['files'].items()}
        files['test/COMPLETE.json'] = sha(test / 'COMPLETE.json')
        for name in ('readout.json', 'terminal-model.pt', 'terminal-model.json', 'test-config.json', 'TEST-RELEASE.json', 'INFERENCE-PLAN.json'):
            files[name] = sha(folder / name)
        write_exclusive(folder / 'INFERENCE_COMPLETE.json', {'unit': unit, 'job': job, 'files': files, 'training_complete_sha256': plan['training_complete_sha256'],
                                                             'views': config['views'], 'time': time.time()})
        return 0
    except BaseException as error:
        write_replace(attempt_dir / 'ATTENTION.json', {'error': repr(error), 'job': job, 'no_automatic_retry': True, 'time': time.time()})
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.campaign, args.unit, args.attempt))
