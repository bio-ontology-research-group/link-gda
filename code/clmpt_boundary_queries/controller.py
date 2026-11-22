"""One Slurm allocation of one query trajectory: run the worker, save before 3h40, exit before 3h45, chain only clean pauses.

Ported from the executed clmpt-axiom13-hpo controller (ops/controller.py, sha256 6bb57445...), query worker only.
Unchanged: the allocation clock starts at the actual Slurm StartTime; the worker receives --allocation-stop-at at
StartTime+13200 s and commits an exact full state; the controller hard-stops the step at StartTime+13410 s and must
be gone by StartTime+13500 s (site OverTimeLimit of 60 min observed for a 3 h request). A successor is submitted only
after a clean deadline pause whose committed state verifies, within the trajectory's allocation bound. Errors,
manual cancellation and exhausted bounds never submit a successor and never retry.
Changed: the frozen manifest also pins the checkout package files and names the campaign; every submission
carries the trajectory's own job name and log path and --no-requeue (sbatch_command, shared with advance.py); the
StartTime query, every sbatch and the advancer's squeue/sacct calls run with SLURM_TIME_FORMAT removed and TZ=UTC
(slurm_env), and StartTime is accepted only as an ISO 8601 UTC timestamp, so an inherited shell time format cannot
shift the allocation clock.
"""
import argparse, datetime, hashlib, json, os, pathlib, re, signal, subprocess, sys, time


SAVE_AT = 13200
EXIT_AT = 13500
HARD_STOP_MARGIN = 90
WORKER = 'query/code/train.py'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


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


def slurm_env():
    """Environment for Slurm client calls: default ISO time output in UTC, whatever the submitting shell exported."""
    env = {key: value for key, value in os.environ.items() if key != 'SLURM_TIME_FORMAT'}
    env['TZ'] = 'UTC'
    return env


def sbatch_command(campaign, trial, wave, job_name, dependency=None):
    """The only form in which a trajectory allocation is submitted (initial by advance.py, successors here)."""
    command = ['sbatch', '--parsable', '--no-requeue', '--job-name=' + job_name, '--output=' + str(pathlib.Path(campaign) / 'logs' / '%x-%j.out')]
    if dependency:
        command.append('--dependency=afterok:' + dependency)
    return command + [str(pathlib.Path(campaign) / 'ops' / 'job.sh'), str(campaign), str(trial), str(wave)]


def verify_frozen(campaign):
    manifest = json.loads((campaign / 'frozen-manifest.json').read_text())
    release = json.loads((campaign / 'ROOT_RELEASE.json').read_text())
    assert manifest['campaign'] == str(campaign)
    assert release['approved'] is True and release['manifest_sha256'] == sha(campaign / 'frozen-manifest.json')
    for name, digest in manifest['files'].items():
        assert sha(campaign / name) == digest, name
    code_dir = pathlib.Path(manifest['code_dir'])
    for name, digest in manifest['code_files'].items():
        assert sha(code_dir / name) == digest, name
    assert pathlib.Path(__file__).resolve() == code_dir / 'controller.py'
    return manifest, release


STARTUP_BUDGET = 480
STARTUP_DELAYS = (5, 10, 20, 40, 60, 60, 60, 60)


def slurm_start(job):
    """Actual Slurm StartTime of this job by bounded read-only scontrol retries; never the submit or an estimated time."""
    began, attempts = time.time(), []
    for delay in STARTUP_DELAYS:
        try:
            result = subprocess.run(['scontrol', 'show', 'job', job, '-o'], capture_output=True, text=True, timeout=45, env=slurm_env())
            identity = re.search(r'\bJobId=(\d+)\b', result.stdout)
            state = re.search(r'\bJobState=(\S+)', result.stdout)
            token = re.search(r'\bStartTime=(\S+)', result.stdout)
            if result.returncode != 0 or not identity or not token:
                attempts.append({'time': time.time(), 'returncode': result.returncode, 'stderr': result.stderr[-300:]})
            elif identity.group(1) != job:
                raise RuntimeError('scontrol returned job %s for %s' % (identity.group(1), job))
            elif token.group(1) in ('Unknown', 'None', 'N/A') or (state and state.group(1) != 'RUNNING'):
                attempts.append({'time': time.time(), 'start_time': token.group(1), 'state': state.group(1) if state else None})
            elif not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', token.group(1)):
                raise RuntimeError('scontrol StartTime is not ISO 8601 UTC: %r' % token.group(1))
            else:
                start = datetime.datetime.fromisoformat(token.group(1)).replace(tzinfo=datetime.timezone.utc).timestamp()
                if start <= time.time():
                    return start, token.group(1), attempts
                attempts.append({'time': time.time(), 'start_time_in_future': token.group(1)})
        except subprocess.TimeoutExpired as error:
            attempts.append({'time': time.time(), 'timeout': repr(error)})
        if time.time() - began + delay + 45 > STARTUP_BUDGET:
            break
        time.sleep(delay)
    raise RuntimeError('scontrol StartTime unavailable after bounded retries: %r' % attempts)


def main(campaign, trial, wave):
    campaign = pathlib.Path(campaign).resolve()
    trial = pathlib.Path(trial).resolve()
    assert campaign in trial.parents
    settings = json.loads((trial / 'runtime.json').read_text())
    assert settings['mode'] == 'query' and settings['worker'] == WORKER
    assert 1 <= settings['maximum_allocations'] <= 16 and wave < settings['maximum_allocations']
    manifest, release = verify_frozen(campaign)
    if (campaign / 'OPERATOR_STOP.json').exists():
        raise SystemExit('Operator stop present: no work in this allocation')
    job = os.environ['SLURM_JOB_ID']
    allocation_start, token, startup_attempts = slurm_start(job)
    assert allocation_start <= time.time() < allocation_start + EXIT_AT
    save_at = allocation_start + SAVE_AT
    exit_at = allocation_start + EXIT_AT
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
    put(folder / 'allocation.json', {'job': job, 'wave': wave, 'mode': settings['mode'], 'allocation_start': allocation_start, 'slurm_start_time': token, 'save_deadline': save_at, 'exit_deadline': exit_at,
                                     'requested_seconds': 10800, 'observed_site_overtime_minutes': 60, 'manual_stop_prevents_successor': True, 'maximum_allocations': settings['maximum_allocations'],
                                     'manifest_sha256': release['manifest_sha256'], 'code_revision': manifest['code_revision']})
    if startup_attempts:
        put(folder / 'startup-scontrol-retries.json', {'job': job, 'failed_attempts': startup_attempts, 'start_time': token})
    run = trial / 'training'
    before = json.loads((run / 'progress.json').read_text())['step'] if (run / 'progress.json').exists() else 0
    telemetry, telemetry_file = None, None
    try:
        inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total', '--format=csv,noheader'], capture_output=True, text=True, timeout=15)
        visible = [row.split(',') for row in inventory.stdout.splitlines() if row.strip()]
        gpu = visible[0][1].strip() if inventory.returncode == 0 and len(visible) == 1 and len(visible[0]) >= 4 else None
        put(folder / 'gpu.json', {'inventory': inventory.stdout, 'slurm_job_gpus': os.environ.get('SLURM_JOB_GPUS'), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'telemetry_id': gpu})
        if gpu:
            telemetry_file = (folder / 'gpu-telemetry.csv').open('w')
            telemetry = subprocess.Popen(['nvidia-smi', '--id=' + gpu, '--query-gpu=timestamp,uuid,utilization.gpu,memory.used', '--format=csv,noheader', '--loop=5'], stdout=telemetry_file, stderr=subprocess.STDOUT)
        assert time.time() < save_at and not stopping, 'Allocation startup consumed training budget'
        worker = campaign / WORKER
        config = trial / 'config.json'
        with (folder / 'worker.log').open('w') as log:
            command = ['srun', '--exact', '--gres=gpu:1', '--ntasks=1', '--cpus-per-task=4', '--kill-on-bad-exit=0', sys.executable, '-u', str(worker), '--config', str(config), '--output', str(run), '--allocation-stop-at', str(save_at)]
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            forwarded, last_heartbeat, last_step_query = False, 0, 0
            while child.poll() is None:
                if target is None:
                    for execution in sorted(run.glob('execution*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
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
                        try:
                            rows = subprocess.run(['squeue', '--steps', '-h', '-j', job, '-o', '%i'], capture_output=True, text=True, timeout=10).stdout.splitlines()
                            targets = [x.strip() for x in rows if re.fullmatch(re.escape(job) + r'\.\d+', x.strip())]
                            if len(targets) == 1:
                                target = targets[0]
                        except subprocess.TimeoutExpired as error:
                            put(folder / 'STEP_QUERY_TRANSIENT.json', {'time': time.time(), 'error': repr(error), 'worker_not_interrupted': True})
                if stopping and not forwarded and target:
                    result = subprocess.run(['scancel', '--signal=TERM', target], capture_output=True, text=True, timeout=10)
                    put(folder / 'TERM-forward.json', {'target': target, 'time': time.time(), 'signals': stopping, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
                    forwarded = True
                if time.time() >= exit_at - HARD_STOP_MARGIN:
                    hard_exit = True
                    if target:
                        subprocess.run(['scancel', '--signal=KILL', target], timeout=10)
                    raise TimeoutError('Allocation exit deadline: last atomic checkpoint retained; no successor')
                if stopping and time.time() - stopping[0]['time'] > 65:
                    raise TimeoutError('Manual stop exceeded 65 s: no successor')
                if time.time() - last_heartbeat >= 10:
                    put(folder / 'heartbeat.json', {'time': time.time(), 'job': job, 'signals': stopping, 'before_step': before})
                    last_heartbeat = time.time()
                time.sleep(.25)
        assert not stopping, 'Manual cancellation: no successor'
        if child.returncode == 0:
            complete = json.loads((run / 'COMPLETE.json').read_text())
            put(folder / 'ALLOCATION_COMPLETE.json', {'job': job, 'mode': settings['mode'], 'training_complete': True, 'steps': complete['steps'], 'completed_at': time.time(), 'before_exit_deadline': time.time() < exit_at})
            return 0
        assert child.returncode == 75, 'Real training error: no automatic retry'
        pause = json.loads((run / 'PAUSED.json').read_text())
        progress = json.loads((run / 'progress.json').read_text())
        assert pause['reason'] == 'deadline' and not pause['signals']
        assert pause['step'] == progress['step'] > before
        assert pause['progress'] == progress
        assert sha(run / progress['checkpoint']) == progress['checkpoint_sha256']
        assert (run / 'updates.jsonl').stat().st_size == progress['log_bytes'] and sha(run / 'updates.jsonl') == progress['log_sha256']
        put(folder / 'CLEAN_DEADLINE.json', {'job': job, 'step': progress['step'], 'checkpoint_sha256': progress['checkpoint_sha256'], 'pause_sha256': sha(run / 'PAUSED.json'),
                                             'progress_sha256': sha(run / 'progress.json'), 'completed_at': time.time(), 'before_exit_deadline': time.time() < exit_at, 'training_complete': False})
        assert wave + 1 < settings['maximum_allocations'], 'Bounded allocation count exhausted; paused state preserved'
        assert not (campaign / 'OPERATOR_STOP.json').exists(), 'Operator stop: no successor'
        assert not stopping and time.time() < exit_at - 40
        with (folder / 'SUBMITTING_SUCCESSOR.json').open('x') as stream:
            json.dump({'time': time.time(), 'wave': wave + 1}, stream)
        try:
            result = subprocess.run(sbatch_command(campaign, trial, wave + 1, settings['job_name'], dependency=job), capture_output=True, text=True, timeout=30, env=slurm_env())
        except subprocess.TimeoutExpired as error:
            put(folder / 'AMBIGUOUS_SUCCESSOR.json', {'error': repr(error), 'no_retry': True})
            raise
        put(folder / 'successor-raw.json', {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        assert result.returncode == 0
        identifier = result.stdout.strip().split(';')[0]
        assert identifier.isdigit()
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
                subprocess.run(['scancel', '--signal=TERM', target], timeout=10)
            stop_budget = stopping[0]['time'] + 65 - time.time() if stopping else 60
            remaining = max(0, min(60, exit_at - time.time() - 25, stop_budget))
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                subprocess.run(['scancel', '--signal=KILL', target], timeout=10)
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
