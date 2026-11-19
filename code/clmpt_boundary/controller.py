"""One GPU allocation of one pretraining trajectory, with bounded exact-resume continuation.

Adapted from the executed three-fold pilot controller (sha256 c3aaf45a...). The query and TEST chaining is
removed: this package only pretrains. The worker receives --allocation-stop-at at StartTime + TimeLimit -
save_margin and commits an exact full state; the controller kills the step and exits before
StartTime + TimeLimit - exit_margin. No site overtime grace is assumed. A successor is submitted only after
a clean deadline pause whose committed state verifies, within the trajectory's allocation bound, and only if
no operator stop exists. SIGTERM (scancel, failure, maintenance) is forwarded to the worker, which commits its
exact state; no successor follows. Errors are never retried.

In-allocation guards sample the allocated GPU every few seconds over the whole wall time, including startup
and checkpoint pauses. A full rolling window below the utilization threshold writes UTILIZATION_WARNING.json;
if utilization stays below the threshold for the configured period after the warning, the worker is stopped
with an exact commit and no successor (UTILIZATION_STOP.json). A gross early shortfall and a stalled worker
heartbeat stop the trajectory the same way. A longer successor time limit is requested only when an operator
file approves it and this allocation's measured utilization passes every configured criterion.
"""
import argparse, datetime, hashlib, json, os, pathlib, re, signal, subprocess, sys, time


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def put(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def journal(campaign, value):
    path = campaign / 'ops-state' / 'jobs.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps({'time': time.time(), **value}, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def seconds(limit):
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d\d):(\d\d)', limit)
    assert match, limit
    days, hours, minutes, secs = (int(x or 0) for x in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + secs


def verified_release(campaign):
    """Verify the approved release, every manifest file and the code checkout pinned by the manifest."""
    release = read(campaign / 'ROOT_RELEASE.json')
    assert release['approved'] is True and release['manifest_sha256'] == sha(campaign / 'frozen-manifest.json')
    manifest = read(campaign / 'frozen-manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(campaign / name) == digest, name
    code = pathlib.Path(manifest['code_dir'])
    for name, digest in manifest['code_files'].items():
        assert sha(code / name) == digest, name
    check_release_request(release)
    return release, manifest


def check_release_request(release):
    """Validate the released Slurm request: allowed time limits and exactly one typed GPU."""
    request = release['slurm_request']
    assert request['time'] in release['allowed_time_limits']
    assert release['utilization_policy']['long_continuation']['time_limit'] in release['allowed_time_limits']
    assert 'gpus' not in request, 'the GPU request lives only in the top-level gpu_request'
    gres = release['gpu_request'].split(':')
    assert len(gres) == 2 and gres[0] and gres[1] == '1' and release['gpu_name_substring'], ('a single typed GPU is required', release['gpu_request'])


def sbatch_arguments(campaign, release, trial, wave, time_limit, dependency=None):
    request = release['slurm_request']
    assert time_limit in release['allowed_time_limits']
    command = ['sbatch', '--parsable', '--no-requeue', '--partition='+ request['partition'], '--qos=' + request['qos'], '--gpus=' + release['gpu_request'], '--time=' + time_limit,
               '--job-name=' + request['job_name_prefix'] + '-' + trial.name,'--output=' + str(campaign / 'logs' / '%x-%j.out')]
    if request.get('account'):
        command.append('--account=' + request['account'])
    if dependency:
        command.append('--dependency=afterok:' + dependency)
    return command + [str(campaign / 'ops/job.sh'), str(campaign), str(trial), str(wave)]


def submit_initial(campaign, release, trial, reason):
    """Submit wave 0 of a trajectory exactly once; an existing marker means reconcile, never resubmit."""
    trial = pathlib.Path(trial).resolve()
    assert campaign in trial.parents and (trial / 'runtime.json').exists() and (trial / 'config.json').exists()
    assert not (campaign / 'OPERATOR_STOP.json').exists() and not (trial / 'OPERATOR_STOP.json').exists(), 'Operator stop present'
    with (trial / 'INITIAL_SUBMITTING.json').open('x') as stream:
        json.dump({'time': time.time(), 'reason': reason, 'manifest_sha256': release['manifest_sha256']}, stream)
    try:
        result = subprocess.run(sbatch_arguments(campaign, release, trial, 0, release['slurm_request']['time']), capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired as error:
        put(trial / 'AMBIGUOUS_SUBMISSION.json', {'error': repr(error), 'no_retry': True})
        raise
    receipt = {'time': time.time(), 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'reason': reason}
    put(trial / 'submission-receipt.json', receipt)
    assert result.returncode == 0, receipt
    identifier = result.stdout.strip().split(';')[0]
    assert identifier.isdigit()
    journal(campaign, {'event': 'submitted', 'trial': str(trial.relative_to(campaign)), 'wave': 0, 'job': identifier, 'reason': reason, 'time_limit': release['slurm_request']['time']})
    return identifier


def signal_worker(child, target, name):
    """Signal the worker step through Slurm when its step ID is known, otherwise through the srun process.

    srun forwards SIGTERM to its task, so the worker can still commit its exact state when step discovery
    failed. SIGKILL of srun ends the launcher; Slurm then removes the step when the allocation ends.
    """
    if target:
        result = subprocess.run(['scancel', '--signal=' + name, target], capture_output=True, text=True, timeout=10)
        return {'method': 'scancel_step', 'target': target, 'signal': name, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    if child.poll() is not None:
        return {'method': 'none_child_exited', 'signal': name, 'returncode': child.returncode}
    if name == 'KILL':
        child.kill()
    else:
        child.send_signal(getattr(signal, 'SIG' + name))
    return {'method': 'srun_process', 'pid': child.pid, 'signal': name}


class Telemetry:
    """Incremental reader of the nvidia-smi CSV written during this allocation."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.rows = []

    def update(self):
        if not self.path.exists():
            return
        with self.path.open('r') as stream:
            stream.seek(self.offset)
            chunk = stream.read()
        complete, _, _ = chunk.rpartition('\n')
        if not complete:
            return
        self.offset += len(complete.encode()) + 1
        for line in complete.split('\n'):
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 4:
                continue
            try:
                stamp = datetime.datetime.strptime(parts[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
                value = float(parts[2].split()[0])
            except ValueError:
                continue
            self.rows.append((stamp, value))

    def window(self, start, end, interval):
        values = [v for t, v in self.rows if start <= t <= end]
        span = max(end - start, 1e-9)
        return {'start': start, 'end': end, 'samples': len(values), 'mean_percent': sum(values) / len(values) if values else None,
                'coverage': min(1., len(values) * interval / span)}


def summarize(telemetry, policy, allocation_start, end):
    """Whole-allocation, allocation-aligned hourly and rolling-hour utilization from all wall samples."""
    interval = policy['sample_interval_seconds']
    width = policy['window_seconds']
    whole = telemetry.window(allocation_start, end, interval)
    hours = []
    begin = allocation_start
    while begin + width <= end:
        hours.append(telemetry.window(begin, begin + width, interval))
        begin += width
    rolling = []
    cursor = allocation_start + width
    while cursor <= end:
        rolling.append(telemetry.window(cursor - width, cursor, interval))
        cursor += 60
    means = [w['mean_percent'] for w in rolling if w['mean_percent'] is not None]
    return {'allocation': whole, 'aligned_hours': hours, 'rolling_hour_minimum_percent': min(means) if means else None,
            'rolling_hour_windows': len(rolling), 'threshold_percent': policy['threshold_percent'], 'includes_startup_and_checkpoint_pauses': True}


def long_continuation_decision(campaign, trial, release, summary):
    """Return the successor time limit and the full reasoning record."""
    policy = release['utilization_policy']['long_continuation']
    threshold = release['utilization_policy']['threshold_percent']
    reasons = {}
    approval = None
    for path in (campaign / policy['operator_file'], trial / policy['operator_file']):
        if path.exists():
            approval = read(path)
    reasons['operator_approved'] = bool(approval and approval.get('approved') is True and approval.get('time_limit') == policy['time_limit'])
    whole = summary['allocation']
    reasons['allocation_mean'] = whole['mean_percent'] is not None and whole['mean_percent'] >= policy['minimum_allocation_mean_percent']
    reasons['allocation_coverage'] = whole['coverage'] >= policy['minimum_wall_coverage']
    reasons['aligned_hours'] = bool(summary['aligned_hours']) and all(h['mean_percent'] is not None and h['mean_percent'] >= threshold for h in summary['aligned_hours'])
    reasons['rolling_hour_minimum'] = (not policy['require_rolling_hour_minimum']) or (summary['rolling_hour_minimum_percent'] is not None and summary['rolling_hour_minimum_percent'] >= threshold)
    reasons['no_warning_in_trajectory'] = not list((trial / 'allocations').glob('wave-*/UTILIZATION_WARNING.json'))
    reasons['no_telemetry_gap_in_trajectory'] = not list((trial / 'allocations').glob('wave-*/TELEMETRY_GAP.json'))
    eligible = all(reasons.values())
    record = {'reasons': reasons, 'approval': approval, 'summary_allocation': whole, 'eligible': eligible}
    if not eligible:
        record['time_limit'] = release['slurm_request']['time']
        return record
    partition = subprocess.run(['scontrol', 'show', 'partition', release['slurm_request']['partition'], '-o'], capture_output=True, text=True, timeout=30)
    match = re.search(r'\bMaxTime=(\S+)', partition.stdout)
    maximum = match.group(1) if match else None
    record['partition_max_time'] = maximum
    record['partition_allows'] = maximum == 'UNLIMITED' or (maximum is not None and seconds(maximum) >= seconds(policy['time_limit']))
    record['time_limit'] = policy['time_limit'] if record['partition_allows'] else release['slurm_request']['time']
    return record


def main(campaign, trial, wave):
    campaign = pathlib.Path(campaign).resolve()
    trial = pathlib.Path(trial).resolve()
    assert campaign in trial.parents
    settings = read(trial / 'runtime.json')
    assert settings['mode'] == 'pretrain' and settings['worker'] == 'train.py'
    assert 1 <= settings['maximum_allocations'] <= 16 and wave < settings['maximum_allocations']
    release, manifest = verified_release(campaign)
    policy = release['utilization_policy']
    if (campaign / 'OPERATOR_STOP.json').exists() or (trial / 'OPERATOR_STOP.json').exists():
        raise SystemExit('Operator stop present: no work in this allocation')
    job = os.environ['SLURM_JOB_ID']
    raw = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=30)
    token = re.search(r'\bStartTime=(\S+)', raw).group(1)
    limit = re.search(r'\bTimeLimit=(\S+)', raw).group(1)
    assert limit in release['allowed_time_limits'], ('unexpected TimeLimit', limit)
    requested = seconds(limit)
    allocation_start = datetime.datetime.fromisoformat(token).timestamp()
    save_at = allocation_start + requested - release['save_margin_seconds']
    exit_at = allocation_start + requested - release['exit_margin_seconds']
    assert allocation_start <= time.time() < save_at
    folder = trial / 'allocations' / ('wave-%03d' % wave)
    folder.mkdir(parents=True, exist_ok=False)
    stopping, queued = [], []
    child, target, hard_exit = None, None, False
    guard = {'warning_at': None, 'stop_reason': None}

    def stop(sig, frame):
        stopping.append({'time': time.time(), 'signal': sig})
        for identifier in queued:
            subprocess.run(['scancel', identifier], timeout=10)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    put(folder / 'allocation.json', {'job': job, 'wave': wave, 'allocation_start': allocation_start, 'slurm_start_time': token, 'slurm_time_limit': limit,
                                     'save_deadline': save_at, 'exit_deadline': exit_at, 'requested_seconds': requested, 'site_overtime_grace_assumed_seconds': 0,
                                     'manual_stop_prevents_successor': True, 'maximum_allocations': settings['maximum_allocations'], 'manifest_sha256': release['manifest_sha256'],
                                     'code_dir': manifest['code_dir'], 'code_revision': manifest.get('code_revision'), 'utilization_policy': policy, 'scontrol': raw})
    journal(campaign, {'event': 'allocation_start', 'trial': str(trial.relative_to(campaign)), 'wave': wave, 'job': job, 'time_limit': limit})
    run = trial / 'training'
    before = read(run / 'progress.json')['step'] if (run / 'progress.json').exists() else 0
    telemetry_process, telemetry_file = None, None
    telemetry = Telemetry(folder / 'gpu-telemetry.csv')
    try:
        inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        visible = [row.split(',') for row in inventory.stdout.splitlines() if row.strip()]
        gpu = visible[0][1].strip() if inventory.returncode == 0 and len(visible) == 1 and len(visible[0]) >= 4 else None
        put(folder / 'gpu.json', {'inventory': inventory.stdout, 'slurm_job_gpus': os.environ.get('SLURM_JOB_GPUS'), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'telemetry_id': gpu})
        assert gpu and release['gpu_name_substring'] in visible[0][2], ('allocated GPU is not the released type', inventory.stdout)
        telemetry_file = (folder / 'gpu-telemetry.csv').open('w')
        telemetry_process = subprocess.Popen(['nvidia-smi', '--id=' + gpu, '--query-gpu=timestamp,uuid,utilization.gpu,memory.used', '--format=csv,noheader',
                                              '--loop=%d' % policy['sample_interval_seconds']], stdout=telemetry_file, stderr=subprocess.STDOUT)
        assert time.time() < save_at and not stopping, 'Allocation startup consumed training budget'
        worker = pathlib.Path(manifest['code_dir']) / settings['worker']
        config = trial / 'config.json'
        launched = time.time()
        with (folder / 'worker.log').open('w') as log:
            command = ['srun', '--exact', '--gpus=1', '--ntasks=1', '--cpus-per-task=' + str(release['worker_cpus']), '--kill-on-bad-exit=0', sys.executable, '-u', str(worker),
                       '--config', str(config), '--output', str(run), '--allocation-stop-at', str(save_at)]
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            forwarded, last_heartbeat, last_step_query, last_guard = False, 0, 0, 0
            while child.poll() is None:
                now = time.time()
                if target is None and now - last_step_query >= 5:
                    last_step_query = now
                    try:
                        rows = subprocess.run(['squeue', '--steps', '-h', '-j', job, '-o', '%i'], capture_output=True, text=True, timeout=10).stdout.splitlines()
                        targets = [x.strip() for x in rows if re.fullmatch(re.escape(job) + r'\.\d+', x.strip())]
                        if len(targets) == 1:
                            target = targets[0]
                    except subprocess.TimeoutExpired as error:
                        put(folder / 'STEP_QUERY_TRANSIENT.json', {'time': now, 'error': repr(error), 'worker_not_interrupted': True})
                if guard['stop_reason'] is None and not stopping and now - last_guard >= 60:
                    last_guard = now
                    telemetry.update()
                    interval = policy['sample_interval_seconds']
                    early = policy['early_guard']
                    if now - allocation_start >= early['after_seconds']:
                        recent = telemetry.window(now - early['window_seconds'], now, interval)
                        if recent['mean_percent'] is not None and recent['coverage'] >= policy['minimum_window_coverage'] and recent['mean_percent'] < early['threshold_percent']:
                            guard['stop_reason'] = {'reason': 'early_gross_underutilization', 'window': recent, 'threshold_percent': early['threshold_percent']}
                    if guard['stop_reason'] is None and now - allocation_start >= policy['window_seconds']:
                        hour = telemetry.window(now - policy['window_seconds'], now, interval)
                        low = hour['mean_percent'] is not None and hour['coverage'] >= policy['minimum_window_coverage'] and hour['mean_percent'] < policy['threshold_percent']
                        if hour['coverage'] < policy['minimum_window_coverage']:
                            put(folder / 'TELEMETRY_GAP.json', {'time': now, 'window': hour, 'long_continuation_blocked': True})
                        if low and guard['warning_at'] is None:
                            guard['warning_at'] = now
                            put(folder / 'UTILIZATION_WARNING.json', {'time': now, 'window': hour, 'threshold_percent': policy['threshold_percent'], 'stop_after_seconds': policy['stop_after_warning_seconds']})
                            journal(campaign, {'event': 'utilization_warning', 'trial': str(trial.relative_to(campaign)), 'wave': wave, 'job': job, 'mean_percent': hour['mean_percent']})
                        elif guard['warning_at'] is not None and now - guard['warning_at'] >= policy['stop_after_warning_seconds']:
                            since = telemetry.window(guard['warning_at'], now, interval)
                            if since['mean_percent'] is None or since['mean_percent'] < policy['threshold_percent']:
                                guard['stop_reason'] = {'reason': 'sustained_low_utilization_after_warning', 'hour': hour, 'since_warning': since, 'threshold_percent': policy['threshold_percent']}
                            else:
                                put(folder / ('UTILIZATION_RECOVERED-%d.json' % now), {'time': now, 'since_warning': since, 'hour': hour})
                                guard['warning_at'] = None
                    heartbeat_path = run / 'heartbeat.json'
                    reference = launched
                    if heartbeat_path.exists():
                        try:
                            reference = max(reference, read(heartbeat_path)['time'])
                        except (ValueError, KeyError):
                            pass
                    if guard['stop_reason'] is None and now - reference > policy['stall_seconds']:
                        guard['stop_reason'] = {'reason': 'worker_heartbeat_stalled', 'seconds_since_heartbeat': now - reference}
                    if guard['stop_reason'] is not None:
                        put(folder / 'UTILIZATION_STOP.json' if 'utilization' in guard['stop_reason']['reason'] else folder / 'STALL_STOP.json', {'time': now, **guard['stop_reason'], 'no_successor': True})
                        journal(campaign, {'event': 'guard_stop', 'trial': str(trial.relative_to(campaign)), 'wave': wave, 'job': job, 'reason': guard['stop_reason']['reason']})
                        stopping.append({'time': now, 'signal': 'guard:' + guard['stop_reason']['reason']})
                if stopping and not forwarded:
                    put(folder / 'TERM-forward.json', {'time': time.time(), 'signals': stopping, **signal_worker(child, target, 'TERM')})
                    forwarded = True
                if time.time() >= exit_at - release['hard_stop_margin_seconds']:
                    hard_exit = True
                    put(folder / 'KILL-at-exit-deadline.json', {'time': time.time(), **signal_worker(child, target, 'KILL')})
                    raise TimeoutError('Allocation exit deadline: last atomic checkpoint retained; no successor')
                if stopping and time.time() - stopping[0]['time'] > release['stop_budget_seconds']:
                    raise TimeoutError('Stop exceeded budget: no successor')
                if time.time() - last_heartbeat >= 10:
                    put(folder / 'heartbeat.json', {'time': time.time(), 'job': job, 'signals': stopping, 'before_step': before, 'utilization_warning_at': guard['warning_at']})
                    last_heartbeat = time.time()
                time.sleep(.25)
        telemetry.update()
        summary = summarize(telemetry, policy, allocation_start, time.time())
        put(folder / 'utilization-summary.json', summary)
        assert not stopping, 'Stop requested (signal or guard): exact state committed by worker; no successor'
        if child.returncode == 0:
            complete = read(run / 'COMPLETE.json')
            put(folder / 'ALLOCATION_COMPLETE.json', {'job': job, 'training_complete': True, 'steps': complete['steps'], 'completed_at': time.time(), 'before_exit_deadline': time.time() < exit_at})
            journal(campaign, {'event': 'complete', 'trial': str(trial.relative_to(campaign)), 'job': job, 'steps': complete['steps']})
            return 0
        assert child.returncode == 75, 'Real training error: no automatic retry'
        pause = read(run / 'PAUSED.json')
        progress = read(run / 'progress.json')
        assert pause['reason'] == 'deadline' and not pause['signals']
        assert pause['step'] == progress['step'] > before
        assert pause['progress'] == progress
        assert sha(run / progress['checkpoint']) == progress['checkpoint_sha256']
        assert (run / 'updates.jsonl').stat().st_size == progress['log_bytes'] and sha(run / 'updates.jsonl') == progress['log_sha256']
        put(folder / 'CLEAN_DEADLINE.json', {'job': job, 'step': progress['step'], 'step_before': before, 'checkpoint_sha256': progress['checkpoint_sha256'], 'pause_sha256': sha(run / 'PAUSED.json'),
                                             'progress_sha256': sha(run / 'progress.json'), 'completed_at': time.time(), 'before_exit_deadline': time.time() < exit_at, 'training_complete': False})
        assert wave + 1 < settings['maximum_allocations'], 'Bounded allocation count exhausted; paused state preserved'
        assert not (campaign / 'OPERATOR_STOP.json').exists() and not (trial / 'OPERATOR_STOP.json').exists(), 'Operator stop: no successor'
        assert not stopping and time.time() < exit_at - 40
        decision = long_continuation_decision(campaign, trial, release, summary)
        put(folder / 'successor-policy.json', decision)
        with (folder / 'SUBMITTING_SUCCESSOR.json').open('x') as stream:
            json.dump({'time': time.time(), 'wave': wave + 1, 'time_limit': decision['time_limit']}, stream)
        try:
            result = subprocess.run(sbatch_arguments(campaign, release, trial, wave + 1, decision['time_limit'], dependency=job), capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired as error:
            put(folder / 'AMBIGUOUS_SUCCESSOR.json', {'error': repr(error), 'no_retry': True})
            raise
        put(folder / 'successor-raw.json', {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        assert result.returncode == 0
        identifier = result.stdout.strip().split(';')[0]
        assert identifier.isdigit()
        queued.append(identifier)
        put(folder / 'successor.json', {'job': identifier, 'dependency': 'afterok:' + job, 'wave': wave + 1, 'time_limit': decision['time_limit'], 'clean_deadline_checkpoint': progress['checkpoint_sha256']})
        journal(campaign, {'event': 'submitted', 'trial': str(trial.relative_to(campaign)), 'wave': wave + 1, 'job': identifier, 'reason': 'clean deadline continuation', 'time_limit': decision['time_limit']})
        if stopping:
            subprocess.run(['scancel', identifier], timeout=10)
            raise RuntimeError('Stop raced with successor submission; queued child cancelled')
        return 0
    except BaseException as error:
        put(folder / 'ATTENTION.json', {'error': repr(error), 'signals': stopping, 'guard': guard, 'job': job, 'no_automatic_retry': True, 'last_committed_state_preserved': True})
        journal(campaign, {'event': 'attention', 'trial': str(trial.relative_to(campaign)), 'wave': wave, 'job': job, 'error': repr(error)})
        for identifier in queued:
            subprocess.run(['scancel', identifier], timeout=10)
        if child is not None and child.poll() is None:
            try:
                if not hard_exit:
                    signal_worker(child, target, 'TERM')
                stop_budget = stopping[0]['time'] + release['stop_budget_seconds'] - time.time() if stopping else 60
                remaining = max(0, min(60, exit_at - time.time() - 25, stop_budget))
                try:
                    child.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    signal_worker(child, target, 'KILL')
            except Exception as cleanup_error:
                put(folder / 'WORKER_STOP_CLEANUP_ERROR.json', {'time': time.time(), 'error': repr(cleanup_error)})
        if not (folder / 'utilization-summary.json').exists():
            try:
                telemetry.update()
                put(folder / 'utilization-summary.json', summarize(telemetry, policy, allocation_start, time.time()))
            except Exception:
                pass
        return 1
    finally:
        try:
            if telemetry_process is not None and telemetry_process.poll() is None:
                telemetry_process.terminate()
                try:
                    telemetry_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    telemetry_process.kill()
                    telemetry_process.wait(timeout=10)
            if telemetry_file is not None:
                telemetry_file.close()
        except Exception as cleanup_error:
            try:
                put(folder / 'TELEMETRY_CLEANUP_WARNING.json', {'time': time.time(), 'error': repr(cleanup_error), 'exit_status_unchanged': True})
            except Exception:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--wave', type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.campaign, args.trial, args.wave))
