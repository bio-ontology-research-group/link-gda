"""Read-only campaign status: queue, journal tail, and per-trajectory progress, guard markers and GPU utilization.

Utilization is the mean of every wall-clock sample in the most recent window of the latest allocation,
including startup and checkpoint pauses. Prints one JSON object per line; writes nothing.
"""
import argparse, collections, json, pathlib, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from controller import Telemetry, read


def run(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=60)
        return {'command': command, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': command, 'error': repr(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=pathlib.Path, required=True)
    parser.add_argument('--jobstats', action='store_true', help='also print the site per-GPU job report link for running jobs')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    now = time.time()
    template = read(campaign / 'config/release-template.json')
    prefix = template['slurm_request']['job_name_prefix']
    policy = template['utilization_policy']
    queue = run(['squeue', '--me', '-h', '-o', '%i|%j|%T|%M|%L|%R|%b'])
    rows = [line for line in queue.get('stdout', '').splitlines() if '|' + prefix + '-' in line]
    print(json.dumps({'time': now, 'queue': rows, 'queue_returncode': queue.get('returncode')}))
    journal = campaign / 'ops-state/jobs.jsonl'
    events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    print(json.dumps({'journal_tail': events[-12:]}))
    jobs = sorted({e['job'] for e in events if 'job' in e})
    if jobs:
        accounting = run(['sacct', '-j', ','.join(jobs), '-X', '--noheader', '--parsable2', '--format=JobID,JobName,State,ExitCode,Start,End,Elapsed,AllocTRES'])
        print(json.dumps({'accounting': accounting.get('stdout', '').splitlines()}))
    for trial in sorted(p for p in (campaign / 'trials').iterdir() if (p / 'runtime.json').exists()):
        out = collections.OrderedDict(trial=trial.name)
        training = trial / 'training'
        for name in ('heartbeat.json', 'progress.json', 'PAUSED.json', 'ATTENTION.json', 'COMPLETE.json', 'IMPORT.json'):
            if (training / name).exists():
                value = read(training / name)
                out[name] = {k: value.get(k) for k in ('step', 'time', 'reason', 'error', 'steps', 'checkpoint_sha256', 'write_hash_seconds', 'steps_imported') if k in value}
        waves = sorted((trial / 'allocations').glob('wave-*'))
        if waves:
            wave = waves[-1]
            out['wave'] = wave.name
            out['markers'] = sorted(p.name for p in wave.glob('*.json') if p.name[0].isupper())
            telemetry = Telemetry(wave / 'gpu-telemetry.csv')
            telemetry.update()
            if telemetry.rows:
                last = telemetry.rows[-1][0]
                out['gpu_last_sample_age_seconds'] = now - last
                for width in (1200, policy['window_seconds']):
                    window = telemetry.window(last - width, last, policy['sample_interval_seconds'])
                    out['gpu_mean_last_%ds' % width] = window['mean_percent']
                    out['gpu_coverage_last_%ds' % width] = window['coverage']
                out['gpu_threshold_percent'] = policy['threshold_percent']
            if (wave / 'utilization-summary.json').exists():
                summary = read(wave / 'utilization-summary.json')
                out['allocation_mean_percent'] = summary['allocation']['mean_percent']
                out['rolling_hour_minimum_percent'] = summary['rolling_hour_minimum_percent']
        print(json.dumps(out, default=str))
    if args.jobstats:
        for line in rows:
            if '|RUNNING|' in line:
                print(json.dumps({'jobstats': run(['jobstats', line.split('|')[0], '-g'])}))


if __name__ == '__main__':
    main()
