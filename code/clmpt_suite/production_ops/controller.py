"""One Slurm allocation of one production unit on Orix (ComplEx pretraining) or IBEX (query training).

Ported from the HPO controller (code/clmpt_suite/controller.py) and the Orix boundary pretraining controller
(code/clmpt_boundary/controller.py). Deadlines: IBEX reads this job's TimeLimit and the cluster OverTimeLimit at run
time (save 3:40, exit 3:45 only with a validated 60-minute grace; inside the request otherwise); Orix assumes no grace
and saves and exits inside the 3-hour request. The worker runs under srun and commits an exact full state at the save
deadline or on SIGTERM. A successor (afterok on this job) is submitted only after a clean deadline pause with real
progress, inside the unit's allocation bound, while no stop marker or campaign halt exists and no other running job
carries the unit's name. A stop marker turns a signal pause into STOPPED. A clean worker pause (return 75, PAUSED,
progress, checkpoint and log verified, progress made) on an unmarked signal is only a SIGNAL_PAUSE_CANDIDATE: no
successor is submitted from the signalled job; campaign.py tick reads the terminal sacct state after the allocation is
gone and resumes (fresh submission, no afterok on the preempted job) only for PREEMPTED within the per-unit bound.
CANCELLED, FAILED, TIMEOUT or any worker error stays attention. Errors never retry.
On Orix the per-GPU utilization is sampled every 5 s: a warning after a low hour, a clean stop if it stays low 30
more minutes, an early stop on gross underutilization, and a stall stop when the worker heartbeat stops.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import read_json, read_overtime, run, seconds, sha  # noqa: E402
import campaign as S  # noqa: E402

STARTUP_DELAYS = (5, 10, 20, 40, 60, 60, 60, 60)


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def job_times(job):
    """Actual StartTime and TimeLimit by bounded scontrol retries (ISO UTC enforced by common.slurm_env)."""
    began, attempts = time.time(), []
    for delay in STARTUP_DELAYS:
        result = run(['scontrol', 'show', 'job', job, '-o'], 45)
        state = re.search(r'\bJobState=(\S+)', result['stdout'])
        token = re.search(r'\bStartTime=(\S+)', result['stdout'])
        limit = re.search(r'\bTimeLimit=(\S+)', result['stdout'])
        if result['returncode'] == 0 and token and limit and state and state.group(1) == 'RUNNING':
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', token.group(1)):
                raise RuntimeError('StartTime is not ISO 8601: %r' % token.group(1))
            start = datetime.datetime.fromisoformat(token.group(1)).replace(tzinfo=datetime.timezone.utc).timestamp()
            if start <= time.time():
                return start, seconds(limit.group(1)), attempts
        attempts.append({'time': time.time(), 'returncode': result['returncode'], 'stdout': result['stdout'][-300:]})
        if time.time() - began + delay > 480:
            break
        time.sleep(delay)
    raise RuntimeError('scontrol StartTime unavailable: %r' % attempts)


def deadlines(site, policy, limit, delays=None):
    if limit != policy['requested_seconds']:
        raise RuntimeError('TimeLimit %r differs from the policy request %d' % (limit, policy['requested_seconds']))
    if site == 'orix':
        return limit - policy['save_margin_seconds'], limit - policy['exit_margin_seconds'], {'grace_assumed_seconds': 0}
    overtime, record = read_overtime(run, delays=delays)
    validated = overtime >= policy['required_overtime_minutes'] * 60
    grace = max(0, min(overtime, 10 ** 6) - policy['overtime_safety_seconds'])
    exit_relative = min(policy['exit_by_allocation_seconds'], limit + grace) if validated and grace > 0 else limit - policy['save_margin_seconds']
    return exit_relative - policy['save_margin_seconds'], exit_relative, dict(record, overtime_seconds=overtime, validated=validated,
                                                                              basis='validated_grace' if validated else 'requested_time_fallback')


class Utilization:
    """Orix per-GPU utilization from nvidia-smi samples of this allocation."""

    def __init__(self, path, policy):
        self.path, self.policy, self.offset, self.rows = path, policy, 0, []
        self.warning_at = None

    def update(self):
        if not self.path.exists():
            return
        with self.path.open() as stream:
            stream.seek(self.offset)
            chunk = stream.read()
        complete, _, _ = chunk.rpartition('\n')
        if not complete:
            return
        self.offset += len(complete.encode()) + 1
        for line in complete.split('\n'):
            parts = [part.strip() for part in line.split(',')]
            try:
                self.rows.append((datetime.datetime.strptime(parts[0], '%Y/%m/%d %H:%M:%S.%f').timestamp(), float(parts[2].split()[0])))
            except (ValueError, IndexError):
                continue

    def window(self, start, end):
        values = [value for stamp, value in self.rows if start <= stamp <= end]
        return {'start': start, 'end': end, 'samples': len(values), 'mean_percent': sum(values) / len(values) if values else None,
                'coverage': min(1., len(values) * self.policy['sample_interval_seconds'] / max(end - start, 1e-9))}

    def judge(self, now, allocation_start, folder):
        """None, or the reason to stop cleanly."""
        self.update()
        policy, early = self.policy, self.policy['early_guard']
        if now - allocation_start >= early['after_seconds']:
            recent = self.window(now - early['window_seconds'], now)
            if recent['mean_percent'] is not None and recent['coverage'] >= policy['minimum_window_coverage'] and recent['mean_percent'] < early['threshold_percent']:
                return {'reason': 'early_gross_underutilization', 'window': recent}
        if now - allocation_start < policy['window_seconds']:
            return None
        hour = self.window(now - policy['window_seconds'], now)
        if hour['coverage'] < policy['minimum_window_coverage']:
            put(folder / ('TELEMETRY_GAP-%d.json' % now), {'window': hour})
            return None
        low = hour['mean_percent'] < policy['threshold_percent']
        if low and self.warning_at is None:
            self.warning_at = now
            put(folder / 'UTILIZATION_WARNING.json', {'time': now, 'window': hour, 'threshold_percent': policy['threshold_percent']})
        elif self.warning_at is not None and now - self.warning_at >= policy['stop_after_warning_seconds']:
            since = self.window(self.warning_at, now)
            if since['mean_percent'] is None or since['mean_percent'] < policy['threshold_percent']:
                return {'reason': 'sustained_low_utilization_after_warning', 'hour': hour, 'since_warning': since}
            put(folder / ('UTILIZATION_RECOVERED-%d.json' % now), {'since_warning': since})
            self.warning_at = None
        return None


def previous_clean(folder, wave):
    present = S.waves(folder)
    if wave == 0:
        return not present
    last = folder / 'allocations' / ('wave-%03d' % (wave - 1))
    if not present or present[-1][0] != wave - 1:
        return False
    if (last / 'RESUME_AUTHORIZED.json').exists():
        return True
    return any((last / name).exists() for name in ('CLEAN_DEADLINE.json', 'HELD.json', 'STOPPED.json', 'PREEMPTION_VERIFIED.json')) \
        and not (last / 'ATTENTION.json').exists() and not (last / 'SIGNAL_NOT_PREEMPTION.json').exists()


def main(campaign_argument, unit, wave):
    context = S.load(campaign_argument)
    campaign, site, policy, release = context['campaign'], context['site'], context['policy'], context['release']
    assert not release.get('rehearsal'), 'rehearsal release: no GPU work'
    folder = S.unit_dir(campaign, unit)
    assert unit in context['manifest']['units'] and (folder / 'config.json').exists()
    assert wave < policy['maximum_allocations'] and previous_clean(folder, wave), 'previous allocation not clean or wave out of bounds'
    assert not (folder / 'training' / 'COMPLETE.json').exists(), 'unit already complete'
    job = os.environ['SLURM_JOB_ID']
    name = release['job_name_prefix'] + unit
    others = run(['squeue', '-h', '-n', name, '-o', '%i|%T'], 30)
    duplicates = [row.split('|')[0] for row in others['stdout'].split() if row.split('|')[0] != job and row.endswith(('RUNNING', 'COMPLETING', 'CONFIGURING'))]
    assert others['returncode'] == 0 and not duplicates, 'another running job carries this unit name: %r' % duplicates
    allocation = folder / 'allocations' / ('wave-%03d' % wave)
    if S.stop_marker(campaign, folder):
        allocation.mkdir(parents=True, exist_ok=False)
        put(allocation / 'STOPPED.json', {'job': job, 'before_start': True, 'time': time.time()})
        return 0
    start, limit, startup = job_times(job)
    save_relative, exit_relative, grace = deadlines(site, policy, limit)
    save_at, exit_at = start + save_relative, start + exit_relative
    assert time.time() < save_at, 'startup consumed the training budget'
    allocation.mkdir(parents=True, exist_ok=False)
    stopping, queued = [], []
    child, target, hard = None, None, False

    def on_signal(sig, frame):
        stopping.append({'time': time.time(), 'signal': sig})
        for identifier in queued:
            subprocess.run(['scancel', identifier], timeout=10)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    put(allocation / 'allocation.json', {'job': job, 'wave': wave, 'site': site, 'allocation_start': start, 'time_limit_seconds': limit, 'grace': grace, 'save_deadline': save_at,
                                         'exit_deadline': exit_at, 'startup_retries': startup, 'manifest_sha256': release['manifest_sha256'], 'host': os.uname().nodename})
    run_dir = folder / 'training'
    before = read_json(run_dir / 'progress.json')['step'] if (run_dir / 'progress.json').exists() else 0
    telemetry, telemetry_file = None, None
    guard = None
    if site == 'orix':
        rules = dict(policy['utilization'])
        for alternative in policy['alternative_partitions'].values():
            if '--partition=' + alternative['partition'] in release['slurm_arguments']:
                rules['threshold_percent'] = alternative['utilization_threshold_percent']
        guard = Utilization(allocation / 'gpu-telemetry.csv', rules)
    try:
        inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        visible = [row.split(',') for row in inventory.stdout.splitlines() if row.strip()]
        gpu = visible[0][1].strip() if inventory.returncode == 0 and len(visible) == 1 and len(visible[0]) >= 4 else None
        put(allocation / 'gpu.json', {'inventory': inventory.stdout, 'name': visible[0][2].strip() if gpu else None, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
        if site == 'orix':
            assert gpu and policy['slurm']['gpu_name_substring'] in visible[0][2], 'allocated GPU is not the released type'
        if gpu:
            telemetry_file = (allocation / 'gpu-telemetry.csv').open('w')
            telemetry = subprocess.Popen(['nvidia-smi', '--id=' + gpu, '--query-gpu=timestamp,uuid,utilization.gpu,memory.used', '--format=csv,noheader',
                                          '--loop=%d' % (policy['utilization']['sample_interval_seconds'] if site == 'orix' else 5)], stdout=telemetry_file, stderr=subprocess.STDOUT)
        worker = campaign / ('worker/train.py' if site == 'orix' else 'query/code/train.py')
        command = ['srun', '--exact', '--ntasks=1', '--cpus-per-task=%d' % policy['slurm']['cpus_per_task'], '--kill-on-bad-exit=0',
                   ('--gpus=1' if site == 'orix' else '--gres=gpu:1'), sys.executable, '-u', str(worker), '--config', str(folder / 'config.json'),
                   '--output', str(run_dir), '--allocation-stop-at', repr(save_at)]
        put(allocation / 'command.json', {'command': command})
        launched, last_beat, last_query, last_guard, forwarded = time.time(), 0, 0, 0, False
        with (allocation / 'worker.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                now = time.time()
                if target is None and now - last_query >= 5:
                    last_query = now
                    rows = run(['squeue', '--steps', '-h', '-j', job, '-o', '%i'], 10)
                    found = [row.strip() for row in rows['stdout'].splitlines() if re.fullmatch(re.escape(job) + r'\.\d+', row.strip())] if rows['returncode'] == 0 else []
                    target = found[0] if len(found) == 1 else None
                if guard is not None and not stopping and now - last_guard >= 60:
                    last_guard = now
                    reason = guard.judge(now, start, allocation)
                    heartbeat = run_dir / 'heartbeat.json'
                    reference = max(launched, read_json(heartbeat).get('time', 0)) if heartbeat.exists() else launched
                    if reason is None and now - reference > policy['utilization']['stall_seconds']:
                        reason = {'reason': 'worker_heartbeat_stalled', 'seconds': now - reference}
                    if reason is not None:
                        put(allocation / 'GUARD_STOP.json', dict(reason, time=now, no_successor=True))
                        stopping.append({'time': now, 'signal': 'guard:' + reason['reason']})
                if stopping and not forwarded and target:
                    put(allocation / 'TERM-forward.json', {'target': target, 'signals': stopping, 'result': run(['scancel', '--signal=TERM', target], 10)})
                    forwarded = True
                if time.time() >= exit_at - policy['hard_stop_margin_seconds']:
                    hard = True
                    if target:
                        run(['scancel', '--signal=KILL', target], 10)
                    raise TimeoutError('exit deadline reached: last atomic checkpoint retained; no successor')
                if stopping and time.time() - stopping[0]['time'] > policy['stop_budget_seconds']:
                    raise TimeoutError('stop exceeded its budget: no successor')
                if time.time() - last_beat >= 10:
                    put(allocation / 'heartbeat.json', {'time': time.time(), 'job': job, 'signals': stopping, 'before_step': before,
                                                        'utilization_warning_at': guard.warning_at if guard else None})
                    last_beat = time.time()
                time.sleep(.25)
        if guard is not None:
            guard.update()
            put(allocation / 'utilization-summary.json', {'allocation': guard.window(start, time.time()), 'threshold_percent': policy['utilization']['threshold_percent']})
        if child.returncode == 0:
            complete = read_json(run_dir / 'COMPLETE.json')
            put(allocation / 'ALLOCATION_COMPLETE.json', {'job': job, 'steps': complete['steps'], 'time': time.time(), 'signals_after_completion': stopping})
            return 0
        assert child.returncode == 75, 'worker error (return code %s): no automatic retry' % child.returncode
        pause, progress = read_json(run_dir / 'PAUSED.json'), read_json(run_dir / 'progress.json')
        log_name = progress.get('log', 'updates.jsonl')
        assert pause['step'] == progress['step'] and sha(run_dir / progress['checkpoint']) == progress['checkpoint_sha256']
        assert (run_dir / log_name).stat().st_size == progress['log_bytes'] and sha(run_dir / log_name) == progress['log_sha256']
        clean = {'job': job, 'step': progress['step'], 'before_step': before, 'checkpoint_sha256': progress['checkpoint_sha256'], 'reason': pause['reason'],
                 'signals': stopping, 'time': time.time(), 'before_exit_deadline': time.time() < exit_at}
        guard_stop = (allocation / 'GUARD_STOP.json').exists()
        if stopping or pause['reason'] != 'deadline':
            if S.stop_marker(campaign, folder):
                put(allocation / 'STOPPED.json', dict(clean, intentional=True))
                return 0
            if guard_stop:
                raise RuntimeError('guard stop (%s): state saved, no successor' % read_json(allocation / 'GUARD_STOP.json')['reason'])
            assert pause['reason'] == 'signal' and progress['step'] > before, 'signal pause without progress or with an unexpected reason: attention'
            put(allocation / 'SIGNAL_PAUSE_CANDIDATE.json', dict(clean, time=time.time(), job=job, worker_returncode=child.returncode,
                                                                 note='clean worker pause on an unmarked signal; campaign.py tick resumes it only if sacct reports PREEMPTED'))
            return 0
        assert progress['step'] - before >= 1, 'no progress in this allocation: no successor'
        put(allocation / 'CLEAN_DEADLINE.json', clean)
        assert wave + 1 < policy['maximum_allocations'], 'allocation bound exhausted; paused state preserved'
        if S.stop_marker(campaign, folder) or (campaign / 'state' / 'HALT.json').exists():
            put(allocation / 'HELD.json', dict(clean, held_because='stop marker or campaign halt'))
            return 0
        with (allocation / 'SUBMITTING_SUCCESSOR.json').open('x') as stream:
            json.dump({'time': time.time(), 'wave': wave + 1}, stream)
        result = run(S.sbatch_command(campaign, release, unit, wave + 1, dependency=job), 60)
        put(allocation / 'successor-raw.json', result)
        identifier = str(result['stdout']).strip().split(';')[0] if result['returncode'] == 0 else ''
        if not identifier.isdigit():
            put(allocation / 'AMBIGUOUS_SUCCESSOR.json', {'result': result, 'no_retry': True})
            raise RuntimeError('successor submission failed or ambiguous')
        queued.append(identifier)
        put(allocation / 'successor.json', {'job': identifier, 'dependency': 'afterok:' + job, 'wave': wave + 1})
        if stopping:
            subprocess.run(['scancel', identifier], timeout=10)
            raise RuntimeError('stop raced with the successor submission; successor cancelled')
        return 0
    except BaseException as error:
        put(allocation / 'ATTENTION.json', {'error': repr(error), 'signals': stopping, 'job': job, 'no_automatic_retry': True})
        for identifier in queued:
            subprocess.run(['scancel', identifier], timeout=10)
        if child is not None and child.poll() is None and target:
            if not hard:
                run(['scancel', '--signal=TERM', target], 10)
            try:
                child.wait(timeout=max(0, min(60, exit_at - time.time() - 25)))
            except subprocess.TimeoutExpired:
                run(['scancel', '--signal=KILL', target], 10)
        return 1
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=10)
            except subprocess.TimeoutExpired:
                telemetry.kill()
            telemetry_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--wave', type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.campaign, args.unit, args.wave))
