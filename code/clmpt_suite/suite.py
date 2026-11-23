"""Shared helpers for the CLMPT suite campaigns: immutable writes, hashing, manifest checks, Slurm commands and trial configs.

Write helpers, checkout verification and Slurm environment handling are ported from code/clmpt_boundary_queries
(campaign.py and controller.py at commit 4518251). Numerical whitelists live only in the campaign protocol file.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).resolve().parent
JOB_NAME_PREFIX = 'lgs-'
PACKAGE_FILES = ('suite.py', 'controller.py', 'advance.py', 'stage_hpo.py', 'select_hpo.py', 'qualify.py', 'capacity.py', 'worker_configs.py',
                 'hpo-protocol.json', 'ops/job-template.sh', 'ops/cpu-job-template.sh')
VIEWS = ('raw', 'calibrated')
STAGES = ('Q', 'S', 'C')
MANIFEST_SCHEMA = 'clmpt-suite-hpo-manifest/1'
WORKER = 'query/code/train.py'
WORKER_PROTOCOL = 'query/protocols/suite-query-v1.json'
INITIALIZER_KEYS = ('taxonomy_mode', 'cohort', 'modality', 'fold', 'pretrain_seed', 'pretrain_learning_rate', 'pretrain_regularization', 'initializer_update_budget')
CONTINUATION_KEYS = ('purpose', 'maximum_steps', 'experiment_id', 'continuation_parent', 'parent_complete_sha256', 'parent_checkpoint_sha256')


class Refusal(Exception):
    pass


def check(condition, message):
    if not condition:
        raise Refusal(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def canonical(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inside(campaign, path):
    """Absolute, normalized, symlink-free path inside the campaign."""
    path = Path(os.path.normpath(str(path)))
    check(path.is_absolute(), 'path must be absolute: %s' % path)
    check(str(path) == str(campaign) or str(path).startswith(str(campaign) + os.sep), 'path outside campaign: %s' % path)
    probe = path
    while str(probe) != str(campaign):
        check(not os.path.islink(str(probe)), 'symlink inside campaign: %s' % probe)
        probe = probe.parent
    return path


def write_exclusive(campaign, path, value):
    """Create a file once; an existing file must hold identical bytes. Returns 'created' or 'identical'."""
    path = inside(campaign, Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else encoded(value)
    if os.path.lexists(str(path)):
        check(path.read_bytes() == data, 'immutable file differs: %s' % path)
        return 'identical'
    temporary = path.with_name('.%s.tmp-%d-%d' % (path.name, os.getpid(), time.time_ns()))
    descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(str(temporary), str(path))
    finally:
        os.unlink(str(temporary))
    fsync_directory(path.parent)
    check(path.read_bytes() == data, 'immutable file differs after racing writer: %s' % path)
    return 'created'


def write_replace(campaign, path, value):
    path = inside(campaign, Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.%s.tmp-%d-%d' % (path.name, os.getpid(), time.time_ns()))
    with temporary.open('wb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def append_line(campaign, path, value):
    path = inside(campaign, Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, 'a') as stream:
        stream.write(json.dumps(dict(value, time=time.time(), pid=os.getpid()), sort_keys=True, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def campaign_root(argument):
    given = Path(argument)
    check(given.is_absolute() and os.path.realpath(str(given)) == str(given), 'campaign must be an absolute symlink-free path: %s' % given)
    return given


def slurm_env():
    """Environment for Slurm client calls: default ISO time output in UTC, whatever the calling shell exported."""
    env = {key: value for key, value in os.environ.items() if key != 'SLURM_TIME_FORMAT'}
    env['TZ'] = 'UTC'
    return env


def run(command, timeout):
    try:
        slurm = command[0] in ('sbatch', 'squeue', 'sacct', 'scontrol', 'scancel')
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=slurm_env() if slurm else None)
        return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'timeout': False}
    except subprocess.TimeoutExpired as error:
        return {'returncode': None, 'stdout': str(error.stdout), 'stderr': str(error.stderr), 'timeout': True}


GRACE_READ_DELAYS = (0, 5, 15, 30)


def read_overtime(run_function, delays=GRACE_READ_DELAYS, sleep=time.sleep):
    """Cluster OverTimeLimit (seconds) from `scontrol show config`, read inside the job with bounded retries.

    Every attempt is recorded with its return code, timeout flag and stderr tail. Returns (0, record) when no attempt
    yields a parsable value, and the caller must then keep its deadlines inside the requested time. A grace is never
    assumed from the login node, the release or a previous allocation.
    """
    attempts = []
    for delay in delays:
        if delay:
            sleep(delay)
        result = run_function(['scontrol', 'show', 'config'], 45)
        match = re.search(r'^OverTimeLimit\s*=\s*(\S+)', result.get('stdout') or '', re.MULTILINE) if result.get('returncode') == 0 else None
        attempts.append({'time': time.time(), 'returncode': result.get('returncode'), 'timeout': result.get('timeout'), 'stderr': (result.get('stderr') or '')[-600:],
                         'line': match.group(0) if match else None})
        if match:
            token = match.group(1)
            value = 10 ** 9 if token.upper() == 'UNLIMITED' else (int(token) * 60 if token.isdigit() else None)
            if value is not None:
                return value, {'read': True, 'line': match.group(0), 'attempts': attempts}
    return 0, {'read': False, 'line': None, 'attempts': attempts, 'fallback': 'deadlines inside the requested time'}


def deadline_safety(allocation, terminal, policy):
    """Whether one allocation used a safe deadline basis and actually ended inside its recorded exit deadline.

    Validated grace: OverTimeLimit was read in the job and covers the required minutes, and the exit deadline is at most
    min(exit_by_allocation_seconds, TimeLimit + OverTimeLimit - overtime_safety_seconds) after the start. Otherwise the
    exit deadline must lie inside the request (TimeLimit - save_margin_seconds). The save deadline must precede the exit
    deadline by the save margin, and the allocation's completion or clean-pause record must be written no later than the
    recorded exit deadline. A missing record, an unsafe bound or a late finish is unsafe.
    """
    reasons = []
    start, save_at, exit_at = allocation.get('allocation_start'), allocation.get('save_deadline'), allocation.get('exit_deadline')
    limit = allocation.get('time_limit_seconds')
    if None in (start, save_at, exit_at, limit):
        return {'safe': False, 'basis': None, 'reasons': ['allocation record lacks deadlines']}
    if limit != policy['requested_seconds']:
        reasons.append('TimeLimit %s differs from the request %s' % (limit, policy['requested_seconds']))
    overtime = allocation.get('overtime_seconds') or 0
    validated = allocation.get('overtime_validated') is True and overtime >= policy['required_overtime_minutes'] * 60
    if validated:
        bound = min(policy['exit_by_allocation_seconds'], limit + min(overtime, 10 ** 6) - policy['overtime_safety_seconds'])
    else:
        bound = limit - policy['save_margin_seconds']
    if exit_at - start > bound + 1e-6:
        reasons.append('exit deadline %.0f s after start exceeds the %s bound %.0f s' % (exit_at - start, 'validated-grace' if validated else 'requested-time', bound))
    if save_at > exit_at - policy['save_margin_seconds'] + 1e-6:
        reasons.append('save deadline is not at least the save margin before the exit deadline')
    finished = (terminal or {}).get('completed_at')
    if finished is None:
        reasons.append('no completion or clean-pause record')
    elif finished > exit_at:
        reasons.append('finished %.1f s after the recorded exit deadline' % (finished - exit_at))
    return {'safe': not reasons, 'basis': 'validated_grace' if validated else 'requested_time_fallback', 'reasons': reasons,
            'exit_after_start_seconds': exit_at - start, 'finished_after_start_seconds': None if finished is None else finished - start}


def job_name(identifier):
    return JOB_NAME_PREFIX + identifier


def sbatch_command(campaign, trial, wave, runtime, dependency=None, test_only=False):
    """The only form in which a trajectory allocation is submitted (initial and resumed by advance.py, successors by controller.py)."""
    command = ['sbatch'] + (['--test-only'] if test_only else ['--parsable']) + ['--no-requeue', '--job-name=' + runtime['job_name'],
               '--output=' + str(Path(campaign) / 'logs' / '%x-%j.out')]
    if runtime.get('constraint'):
        command.append('--constraint=' + runtime['constraint'])
    if dependency:
        command.append('--dependency=afterok:' + dependency)
    return command + [str(Path(campaign) / 'ops' / 'job.sh'), str(campaign), str(trial), str(wave)]


def cpu_command(campaign, action, argument, test_only=False):
    """CPU-only Slurm job running one advance.py subcommand (selection or qualification comparison)."""
    return ['sbatch'] + (['--test-only'] if test_only else ['--parsable']) + ['--no-requeue', '--job-name=' + job_name('%s-%s' % (action, argument)),
            '--output=' + str(Path(campaign) / 'logs' / '%x-%j.out'), str(Path(campaign) / 'ops' / 'cpu_job.sh'), str(campaign), action, argument]


def job_token(result):
    token = str(result.get('stdout', '')).strip().split(';')[0] if result.get('returncode') == 0 else ''
    return token if token.isdigit() else None


def verify_manifest(campaign, bank_full=False):
    """Frozen-manifest context: campaign files and checkout package files by hash; bank by COMPLETE hash and sizes; initializers by size."""
    manifest_path = campaign / 'frozen-manifest.json'
    check(manifest_path.exists(), 'campaign not staged: frozen-manifest.json missing')
    manifest = read_json(manifest_path)
    check(manifest.get('schema') == MANIFEST_SCHEMA and manifest['campaign'] == str(campaign), 'manifest does not describe this campaign')
    for name, digest in manifest['files'].items():
        check(sha(inside(campaign, campaign / name)) == digest, 'frozen campaign file changed: %s' % name)
    code_dir = Path(manifest['code_dir'])
    for name, digest in manifest['code_files'].items():
        check(sha(code_dir / name) == digest, 'checkout package file changed: %s' % name)
    check(HERE == code_dir and sha(HERE / 'suite.py') == manifest['code_files']['suite.py'], 'running helper differs from the frozen checkout')
    bank = manifest['bank']
    check(sha(Path(bank['dir']) / 'COMPLETE.json') == bank['complete_sha256'], 'bank COMPLETE changed')
    for name, record in bank['files'].items():
        path = Path(bank['dir']) / name
        check(path.stat().st_size == record['bytes'], 'bank file size changed: %s' % name)
        if bank_full:
            check(sha(path) == record['sha256'], 'bank file hash changed: %s' % name)
    for record in manifest['initializers'].values():
        check(Path(record['path']).stat().st_size == record['bytes'], 'initializer size changed: %s' % record['path'])
    for name, digest in manifest['worker']['source_hashes'].items():
        check(sha(campaign / 'query' / 'code' / name) == digest, 'worker source changed: %s' % name)
    protocol_path = campaign / 'config' / 'protocol.json'
    protocol = read_json(protocol_path)
    check(protocol['status'] == 'FROZEN' and sha(protocol_path) == manifest['protocol_sha256'], 'protocol not frozen or not pinned')
    worker_protocol_path = campaign / WORKER_PROTOCOL
    check(sha(worker_protocol_path) == manifest['worker']['protocol_sha256'], 'worker protocol changed')
    return {'campaign': campaign, 'manifest': manifest, 'manifest_sha256': sha(manifest_path), 'protocol': protocol, 'protocol_path': str(protocol_path),
            'protocol_sha256': manifest['protocol_sha256'], 'worker': manifest['worker'], 'worker_protocol_path': str(worker_protocol_path),
            'bank': bank, 'code_dir': code_dir, 'python': manifest['python'], 'initializers': manifest['initializers']}


def fmt(value):
    return repr(float(value))


def screen_id(initializer_name, learning_rate, temperature):
    return 'S-%s-qlr%s-t%s' % (initializer_name, fmt(learning_rate), fmt(temperature))


def trial_dir(campaign, stage, identifier):
    return Path(campaign) / 'trials' / stage / identifier


def plan_item(output, protocol_path, purpose, identifier, data_dir, initializer, fixed, learning_rate, temperature, steps, evaluation_every, maximum_seconds,
              continuation_parent=None):
    """One entry of a worker_configs.py plan; the worker's make_config.build_config produces the actual config."""
    return {'output': str(output), 'protocol': str(protocol_path), 'purpose': purpose, 'experiment_id': identifier, 'data_dir': str(data_dir),
            'pretrained': initializer['path'], 'pretrained_sha256': initializer['sha256'], 'initializer': {key: initializer[key] for key in INITIALIZER_KEYS},
            'seed': fixed['seed'], 'encoder': fixed['encoder'], 'curriculum': fixed['curriculum'], 'learning_rate': learning_rate, 'temperature': temperature,
            'maximum_steps': steps, 'evaluation_every_updates': evaluation_every, 'maximum_seconds': maximum_seconds, 'continuation_parent': continuation_parent}


def continuation_item(output, parent_config, parent_training_dir, identifier, total_steps):
    """hpo_continuation plan entry: every setting is the parent screen's; only purpose, budget, id and the parent pins change."""
    check(parent_config['purpose'] == 'hpo_screen' and not parent_config.get('continuation_parent'), 'continuation parent is not a fresh screen')
    check(parent_config['sampler_seed'] == parent_config['seed'], 'parent sampler seed differs from its seed')
    initializer = dict(parent_config['initializer'], path=parent_config['pretrained_checkpoint'], sha256=parent_config['pretrained_sha256'])
    fixed = {key: parent_config[key] for key in ('seed', 'encoder', 'curriculum')}
    return plan_item(output, parent_config['protocol_path'], 'hpo_continuation', identifier, parent_config['data_dir'], initializer, fixed, parent_config['learning_rate'],
                     parent_config['temperature'], total_steps, parent_config['evaluation_every_updates'], parent_config['maximum_seconds'],
                     continuation_parent=str(parent_training_dir))


def continuation_differences(parent_config, config):
    """Keys whose values differ between a screen config and its continuation; must be a subset of CONTINUATION_KEYS."""
    return sorted(key for key in set(config) | set(parent_config) if config.get(key) != parent_config.get(key))


def runtime(protocol, stage, identifier, maximum_allocations, order_key, constraint=None, forced_pause_after_seconds=None, pause_at_step=None):
    check(1 <= maximum_allocations <= protocol['runtime']['maximum_allocations_cap'], 'runtime out of range')
    value = {'mode': 'query', 'worker': WORKER, 'maximum_allocations': maximum_allocations, 'stage': stage, 'trial_id': identifier, 'job_name': job_name(identifier),
             'order_key': list(order_key), 'constraint': constraint or protocol['runtime']['gpu_constraint']}
    if forced_pause_after_seconds is not None or pause_at_step is not None:
        check(stage == 'Q', 'forced pauses are qualification-only')
    if forced_pause_after_seconds is not None:
        value['forced_pause_after_seconds'] = int(forced_pause_after_seconds)
    if pause_at_step is not None:
        value['pause_at_step'] = int(pause_at_step)
    return value


def worker_configs_command(python, worker_code, plan, receipt):
    """Subprocess building and validating configs with the worker's own make_config in worker_code."""
    return [python, '-I', str(HERE / 'worker_configs.py'), '--worker-code', str(worker_code), '--plan', str(plan), '--receipt', str(receipt)]


def git(directory, *arguments):
    return subprocess.run(['git', '-C', str(directory)] + list(arguments), capture_output=True, text=True, timeout=60, check=True).stdout


def verified_checkout(revision, files):
    """Require every EXECUTED file to be committed and unmodified at the pushed upstream revision; return the Git record.

    Only the executed files are checked, so new non-executed work elsewhere in the package (data, pretraining, production
    planning) does not block staging. The unlicensed CLMPT reference copies are never executed from the checkout: the
    stager copies them from --references-dir by hash.
    """
    head = git(HERE, 'rev-parse', 'HEAD').strip()
    check(head == revision, 'git HEAD %s differs from --code-revision %s' % (head, revision))
    upstream = git(HERE, 'rev-parse', '@{upstream}').strip()
    check(upstream == head, 'staged revision must be the pushed and pulled commit: HEAD %s, upstream %s' % (head, upstream))
    status = git(HERE, 'status', '--porcelain', '--untracked-files=all', '--', *files)
    check(not status.strip(), 'executed files not clean: %s' % status.splitlines())
    tracked = {}
    for line in git(HERE, 'ls-files', '-s', '--', '.').splitlines():
        meta, path = line.split('\t', 1)
        tracked[path] = meta.split()[1]
    for name in files:
        check(name in tracked, 'executed file not tracked: %s' % name)
        check(git(HERE, 'hash-object', str(HERE / name)).strip() == tracked[name], 'working file differs from committed blob: %s' % name)
    return {'head': head, 'upstream': upstream, 'branch': git(HERE, 'rev-parse', '--abbrev-ref', 'HEAD').strip(), 'package_status_clean': True,
            'tracked_blob_ids': {name: tracked[name] for name in files}, 'top_level': git(HERE, 'rev-parse', '--show-toplevel').strip()}
