"""One Slurm allocation of one suite trajectory: run the worker under srun, save before the validated deadline, chain only clean pauses.

Ported from code/clmpt_boundary_queries/controller.py (commit 4518251). Unchanged: the allocation clock starts at
the actual Slurm StartTime; the worker gets --allocation-stop-at and commits an exact full state; a successor is
submitted only after a clean deadline pause whose committed state verifies; errors never retry.
Changed: the deadlines are derived from this job's own TimeLimit and the cluster OverTimeLimit read at run time (a
missing or short grace moves the deadlines inside the request instead of assuming 60 minutes); another running job
with the same trajectory name refuses the allocation; a successor needs real progress; an intentional stop
(OPERATOR_STOP.json or the trial's STOP.json) is recorded as STOPPED, not as an error; a clean pause while the
campaign is halted or stopped is recorded as HELD and left for the advancer instead of chaining; qualification
trials can force an early pause on their first allocation.
"""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location('suite_common', str(HERE / 'suite.py'))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

STARTUP_BUDGET = 480
STARTUP_DELAYS = (5, 10, 20, 40, 60, 60, 60, 60)
WAVE = re.compile(r'wave-(\d{3})')


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


def seconds(token):
    """Slurm duration [D-]HH:MM:SS or MM:SS or MM to seconds; None for UNLIMITED or unparsable."""
    match = re.fullmatch(r'(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)', token or '')
    if match:
        days, hours, minutes, secs = (int(part) if part else 0 for part in match.groups())
        return ((days * 24 + hours) * 60 + minutes) * 60 + secs
    if token and token.isdigit():
        return int(token) * 60
    return None


def overtime_seconds():
    """Cluster OverTimeLimit in seconds from scontrol show config; 0 when absent, unreadable or unparsable."""
    result = C.run(['scontrol', 'show', 'config'], 45)
    match = re.search(r'^OverTimeLimit\s*=\s*(\S+)(?:\s*min)?', result['stdout'], re.MULTILINE) if result['returncode'] == 0 else None
    record = {'returncode': result['returncode'], 'line': match.group(0) if match else None}
    if not match:
        return 0, record
    token = match.group(1)
    if token.upper() == 'UNLIMITED':
        return 10 ** 9, record
    return (int(token) * 60 if token.isdigit() else 0), record


def slurm_start(job):
    """Actual StartTime and TimeLimit of this job by bounded read-only scontrol retries; never the submit or an estimated time."""
    began, attempts = time.time(), []
    for delay in STARTUP_DELAYS:
        result = C.run(['scontrol', 'show', 'job', job, '-o'], 45)
        identity = re.search(r'\bJobId=(\d+)\b', result['stdout'])
        state = re.search(r'\bJobState=(\S+)', result['stdout'])
        token = re.search(r'\bStartTime=(\S+)', result['stdout'])
        limit = re.search(r'\bTimeLimit=(\S+)', result['stdout'])
        if result['returncode'] != 0 or not identity or not token or not limit:
            attempts.append({'time': time.time(), 'returncode': result['returncode'], 'stderr': (result['stderr'] or '')[-300:]})
        elif identity.group(1) != job:
            raise RuntimeError('scontrol returned job %s for %s' % (identity.group(1), job))
        elif token.group(1) in ('Unknown', 'None', 'N/A') or (state and state.group(1) != 'RUNNING'):
            attempts.append({'time': time.time(), 'start_time': token.group(1), 'state': state.group(1) if state else None})
        elif not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', token.group(1)):
            raise RuntimeError('scontrol StartTime is not ISO 8601 UTC: %r' % token.group(1))
        else:
            start = datetime.datetime.fromisoformat(token.group(1)).replace(tzinfo=datetime.timezone.utc).timestamp()
            if start <= time.time():
                return start, token.group(1), seconds(limit.group(1)), attempts
            attempts.append({'time': time.time(), 'start_time_in_future': token.group(1)})
        if time.time() - began + delay + 45 > STARTUP_BUDGET:
            break
        time.sleep(delay)
    raise RuntimeError('scontrol StartTime unavailable after bounded retries: %r' % attempts)


def deadlines(policy, limit, overtime):
    """Relative save and exit deadlines: inside request+grace when the grace is validated, inside the request otherwise."""
    if limit is None or limit != policy['requested_seconds']:
        raise RuntimeError('job TimeLimit %r differs from the frozen request %d' % (limit, policy['requested_seconds']))
    grace = max(0, min(overtime, 10 ** 6) - policy['overtime_safety_seconds'])
    exit_relative = min(policy['exit_by_allocation_seconds'], limit + grace) if grace > 0 else limit - policy['save_margin_seconds']
    return exit_relative - policy['save_margin_seconds'], exit_relative, overtime >= policy['required_overtime_minutes'] * 60


def duplicate_writers(job, name):
    result = C.run(['squeue', '-h', '-n', name, '-o', '%i|%T'], 30)
    if result['returncode'] != 0:
        return None
    rows = [line.strip().split('|') for line in result['stdout'].splitlines() if line.strip()]
    return [row[0] for row in rows if row[0] != job and len(row) > 1 and row[1] in ('RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED')]


def stop_requested(campaign, trial):
    return os.path.lexists(str(campaign / 'OPERATOR_STOP.json')) or os.path.lexists(str(trial / 'STOP.json'))


def previous_wave_clean(trial, wave):
    """Wave 0 needs no allocations; wave N needs wave N-1 to have ended in a clean pause and no later wave to exist."""
    folder = trial / 'allocations'
    present = sorted(int(WAVE.fullmatch(path.name).group(1)) for path in folder.iterdir() if WAVE.fullmatch(path.name)) if folder.is_dir() else []
    if wave == 0:
        return not present
    last = folder / ('wave-%03d' % (wave - 1))
    if (last / 'RESUME_AUTHORIZED.json').exists():
        return bool(present) and present[-1] == wave - 1
    return bool(present) and present[-1] == wave - 1 and any((last / name).exists() for name in ('CLEAN_DEADLINE.json', 'HELD.json', 'STOPPED.json')) and not (last / 'ATTENTION.json').exists()


def main(campaign, trial, wave):
    campaign = C.campaign_root(campaign)
    trial = Path(trial).resolve()
    assert campaign in trial.parents
    context = C.verify_manifest(campaign)
    release = C.read_json(campaign / 'ROOT_RELEASE.json')
    assert release['approved'] is True and release['manifest_sha256'] == context['manifest_sha256']
    assert Path(__file__).resolve() == context['code_dir'] / 'controller.py'
    policy = context['protocol']['runtime']
    settings = C.read_json(trial / 'runtime.json')
    assert settings['mode'] == 'query' and settings['worker'] == C.WORKER
    assert 1 <= settings['maximum_allocations'] <= policy['maximum_allocations_cap'] and wave < settings['maximum_allocations']
    job = os.environ['SLURM_JOB_ID']
    assert previous_wave_clean(trial, wave), 'previous allocation did not end in a clean pause or a later wave exists'
    assert not (trial / 'training' / 'COMPLETE.json').exists(), 'trajectory already complete'
    others = duplicate_writers(job, settings['job_name'])
    assert others == [], 'another running job carries this trajectory name: %r' % others
    if stop_requested(campaign, trial):
        folder = trial / 'allocations' / ('wave-%03d' % wave)
        folder.mkdir(parents=True, exist_ok=False)
        put(folder / 'STOPPED.json', {'job': job, 'before_start': True, 'intentional': True, 'time': time.time(), 'resume': 'advance.py resume after the stop marker is removed'})
        return 0
    allocation_start, token, limit, startup_attempts = slurm_start(job)
    overtime, overtime_record = overtime_seconds()
    save_relative, exit_relative, overtime_validated = deadlines(policy, limit, overtime)
    save_at, exit_at = allocation_start + save_relative, allocation_start + exit_relative
    forced = settings.get('forced_pause_after_seconds') if wave == 0 else None
    if forced is not None:
        save_at = min(save_at, time.time() + forced)
    assert allocation_start <= time.time() < save_at
    folder = trial / 'allocations' / ('wave-%03d' % wave)
    folder.mkdir(parents=True, exist_ok=False)
    stopping, queued = [], []
    child, target, hard_exit = None, None, False

    def stop(sig, frame):
        stopping.append({'time': time.time(), 'signal': sig})
        for identifier in queued:
            subprocess.run(['scancel', identifier], timeout=10)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    put(folder / 'allocation.json', {'job': job, 'wave': wave, 'mode': settings['mode'], 'stage': settings['stage'], 'allocation_start': allocation_start, 'slurm_start_time': token,
                                     'time_limit_seconds': limit, 'overtime_seconds': overtime, 'overtime_record': overtime_record, 'overtime_validated': overtime_validated,
                                     'save_deadline': save_at, 'exit_deadline': exit_at, 'forced_pause_after_seconds': forced,
                                     'pause_at_step': settings.get('pause_at_step') if wave == 0 else None, 'maximum_allocations': settings['maximum_allocations'],
                                     'manifest_sha256': context['manifest_sha256'], 'code_revision': context['manifest']['code_revision'], 'host': os.uname().nodename})
    if startup_attempts:
        put(folder / 'startup-scontrol-retries.json', {'job': job, 'failed_attempts': startup_attempts, 'start_time': token})
    run = trial / 'training'
    before = C.read_json(run / 'progress.json')['step'] if (run / 'progress.json').exists() else 0
    telemetry, telemetry_file = None, None
    try:
        inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        visible = [row.split(',') for row in inventory.stdout.splitlines() if row.strip()]
        gpu = visible[0][1].strip() if inventory.returncode == 0 and len(visible) == 1 and len(visible[0]) >= 4 else None
        put(folder / 'gpu.json', {'inventory': inventory.stdout, 'name': visible[0][2].strip() if gpu else None, 'slurm_job_gpus': os.environ.get('SLURM_JOB_GPUS'),
                                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'telemetry_id': gpu})
        if gpu:
            telemetry_file = (folder / 'gpu-telemetry.csv').open('w')
            telemetry = subprocess.Popen(['nvidia-smi', '--id=' + gpu, '--query-gpu=timestamp,uuid,utilization.gpu,memory.used', '--format=csv,noheader', '--loop=5'],
                                         stdout=telemetry_file, stderr=subprocess.STDOUT)
        assert time.time() < save_at and not stopping, 'Allocation startup consumed training budget'
        worker = campaign / C.WORKER
        with (folder / 'worker.log').open('w') as log:
            command = ['srun', '--exact', '--gres=gpu:1', '--ntasks=1', '--cpus-per-task=4', '--kill-on-bad-exit=0', sys.executable, '-u', str(worker),
                       '--config', str(trial / 'config.json'), '--output', str(run), '--allocation-stop-at', repr(save_at)]
            if wave == 0 and settings.get('pause_at_step') is not None:
                command += ['--pause-at-step', str(settings['pause_at_step'])]
            put(folder / 'command.json', {'command': command})
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            forwarded, last_heartbeat, last_step_query = False, 0, 0
            while child.poll() is None:
                if target is None:
                    for execution in sorted(run.glob('execution*.json'), key=lambda path: path.stat().st_mtime_ns, reverse=True):
                        try:
                            metadata = json.loads(execution.read_text())
                        except (OSError, json.JSONDecodeError):
                            continue
                        step = str(metadata.get('slurm_step', ''))
                        if str(metadata.get('job_id')) == job and step.isdigit():
                            target = job + '.' + step
                            break
                    if target is None and time.time() - last_step_query >= 5:
                        last_step_query = time.time()
                        rows = C.run(['squeue', '--steps', '-h', '-j', job, '-o', '%i'], 10)
                        targets = [x.strip() for x in rows['stdout'].splitlines() if re.fullmatch(re.escape(job) + r'\.\d+', x.strip())] if rows['returncode'] == 0 else []
                        if len(targets) == 1:
                            target = targets[0]
                if stopping and not forwarded and target:
                    result = C.run(['scancel', '--signal=TERM', target], 10)
                    put(folder / 'TERM-forward.json', {'target': target, 'time': time.time(), 'signals': stopping, 'result': result})
                    forwarded = True
                if time.time() >= exit_at - policy['hard_stop_margin_seconds']:
                    hard_exit = True
                    if target:
                        C.run(['scancel', '--signal=KILL', target], 10)
                    raise TimeoutError('Allocation exit deadline: last atomic checkpoint retained; no successor')
                if stopping and time.time() - stopping[0]['time'] > 65:
                    raise TimeoutError('Stop exceeded 65 s: no successor')
                if time.time() - last_heartbeat >= 10:
                    put(folder / 'heartbeat.json', {'time': time.time(), 'job': job, 'signals': stopping, 'before_step': before})
                    last_heartbeat = time.time()
                time.sleep(.25)
        if child.returncode == 0 and not stopping:
            complete = C.read_json(run / 'COMPLETE.json')
            put(folder / 'ALLOCATION_COMPLETE.json', {'job': job, 'mode': settings['mode'], 'training_complete': True, 'steps': complete['steps'],
                                                      'completed_at': time.time(), 'before_exit_deadline': time.time() < exit_at})
            return 0
        assert child.returncode in (0, 75), 'Real training error: no automatic retry'
        if child.returncode == 0:
            put(folder / 'ALLOCATION_COMPLETE.json', {'job': job, 'mode': settings['mode'], 'training_complete': True, 'completed_at': time.time(), 'signals_after_completion': stopping})
            return 0
        pause = C.read_json(run / 'PAUSED.json')
        progress = C.read_json(run / 'progress.json')
        log_name = progress.get('log', 'updates.jsonl')
        assert pause['step'] == progress['step'] and pause['progress'] == progress
        assert C.sha(run / progress['checkpoint']) == progress['checkpoint_sha256']
        assert (run / log_name).stat().st_size == progress['log_bytes'] and C.sha(run / log_name) == progress['log_sha256']
        clean = {'job': job, 'step': progress['step'], 'before_step': before, 'checkpoint_sha256': progress['checkpoint_sha256'], 'pause_sha256': C.sha(run / 'PAUSED.json'),
                 'progress_sha256': C.sha(run / 'progress.json'), 'reason': pause['reason'], 'signals': stopping, 'completed_at': time.time(),
                 'before_exit_deadline': time.time() < exit_at, 'training_complete': False}
        if stopping or pause['reason'] not in ('deadline', 'qualification_pause'):
            if stop_requested(campaign, trial):
                put(folder / 'STOPPED.json', dict(clean, intentional=True, resume='advance.py resume after the stop marker is removed'))
                return 0
            raise RuntimeError('Signal without a stop marker (manual cancellation or scheduler signal): state saved, no successor')
        assert progress['step'] - before >= policy['minimum_progress_steps_for_successor'], 'No training progress in this allocation: no successor'
        put(folder / 'CLEAN_DEADLINE.json', clean)
        assert wave + 1 < settings['maximum_allocations'], 'Bounded allocation count exhausted; paused state preserved'
        if stop_requested(campaign, trial) or (campaign / 'state' / 'HALT.json').exists():
            put(folder / 'HELD.json', dict(clean, held_because='operator stop, trial stop or campaign halt', resume='advance.py tick resubmits once nothing holds it'))
            return 0
        assert time.time() < exit_at - 40
        with (folder / 'SUBMITTING_SUCCESSOR.json').open('x') as stream:
            json.dump({'time': time.time(), 'wave': wave + 1}, stream)
        result = C.run(C.sbatch_command(campaign, trial, wave + 1, settings, dependency=job), 30)
        put(folder / 'successor-raw.json', result)
        if result['timeout']:
            put(folder / 'AMBIGUOUS_SUCCESSOR.json', {'result': result, 'no_retry': True})
            raise RuntimeError('successor submission timed out')
        identifier = C.job_token(result)
        assert identifier, 'successor submission refused'
        queued.append(identifier)
        put(folder / 'successor.json', {'job': identifier, 'dependency': 'afterok:' + job, 'wave': wave + 1, 'clean_deadline_checkpoint': progress['checkpoint_sha256']})
        if stopping:
            subprocess.run(['scancel', identifier], timeout=10)
            raise RuntimeError('Stop raced with successor submission; queued child cancelled')
        return 0
    except BaseException as error:
        put(folder / 'ATTENTION.json', {'error': repr(error), 'signals': stopping, 'job': job, 'no_automatic_retry': True, 'last_committed_state_preserved': True})
        for identifier in queued:
            subprocess.run(['scancel', identifier], timeout=10)
        if child is not None and child.poll() is None and target:
            if not hard_exit:
                C.run(['scancel', '--signal=TERM', target], 10)
            stop_budget = stopping[0]['time'] + 65 - time.time() if stopping else 60
            remaining = max(0, min(60, exit_at - time.time() - 25, stop_budget))
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                C.run(['scancel', '--signal=KILL', target], 10)
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
    parser.add_argument('--trial', required=True)
    parser.add_argument('--wave', type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.campaign, args.trial, args.wave))
