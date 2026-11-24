"""Shared helpers for the production pipeline: hashing, write-once files, locks, Slurm calls and durations.

Standard library only, so every login-node command runs with plain python3.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).resolve().parent
SUITE = HERE.parent


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
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=str) + '\n').encode()


def canonical(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def read_json(path, default=None):
    path = Path(path)
    if default is not None and not path.exists():
        return default
    return json.loads(path.read_text())


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path, value):
    """Create once; an existing file must hold identical bytes. Returns 'created' or 'identical'."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else encoded(value)
    if os.path.lexists(str(path)):
        check(path.read_bytes() == data, 'immutable file differs: %s' % path)
        return 'identical'
    temporary = path.with_name('.%s.tmp-%d-%d' % (path.name, os.getpid(), time.time_ns()))
    with open(str(temporary), 'xb') as stream:
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


def write_replace(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.%s.tmp-%d-%d' % (path.name, os.getpid(), time.time_ns()))
    with temporary.open('wb') as stream:
        stream.write(value if isinstance(value, bytes) else encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def append_line(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), 'a') as stream:
        stream.write(json.dumps(dict(value, time=time.time(), pid=os.getpid()), sort_keys=True, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def exclusive_lock(path):
    """Non-blocking exclusive lock; yields False when another process holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    finally:
        os.close(descriptor)


def slurm_env():
    """Slurm client environment: ISO times in UTC whatever the shell exported (Orix ~/.profile sets SLURM_TIME_FORMAT)."""
    env = {key: value for key, value in os.environ.items() if key != 'SLURM_TIME_FORMAT'}
    env['TZ'] = 'UTC'
    return env


SLURM = ('sbatch', 'squeue', 'sacct', 'scontrol', 'scancel', 'sinfo', 'sacctmgr')


def run(command, timeout, stdin=None):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, input=stdin,
                                env=slurm_env() if Path(command[0]).name in SLURM else None)
        return {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr[-4000:], 'timeout': False}
    except subprocess.TimeoutExpired as error:
        return {'command': command, 'returncode': None, 'stdout': str(error.stdout or ''), 'stderr': str(error.stderr or '')[-4000:], 'timeout': True}
    except FileNotFoundError as error:
        return {'command': command, 'returncode': 127, 'stdout': '', 'stderr': repr(error), 'timeout': False}


GRACE_READ_DELAYS = (0, 5, 15, 30)


def read_overtime(run_function, delays=None, sleep=time.sleep):
    """Cluster OverTimeLimit (seconds) from `scontrol show config`, read inside the job with bounded retries and a
    per-attempt stderr audit; (0, record) when no attempt is parsable, so the caller keeps deadlines inside the request.
    Same rule as the HPO root suite.read_overtime."""
    attempts = []
    for delay in (GRACE_READ_DELAYS if delays is None else delays):
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


def job_token(result):
    token = str(result.get('stdout', '')).strip().split(';')[0] if result.get('returncode') == 0 else ''
    return token if token.isdigit() else None


def seconds(token):
    """Slurm duration [D-]HH:MM:SS, MM:SS or minutes to seconds; None when unlimited or unparsable."""
    match = re.fullmatch(r'(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)', token or '')
    if match:
        days, hours, minutes, secs = (int(part) if part else 0 for part in match.groups())
        return ((days * 24 + hours) * 60 + minutes) * 60 + secs
    if token and token.isdigit():
        return int(token) * 60
    return None


def tree_bytes(path):
    total = 0
    if Path(path).is_dir():
        for root, _, files in os.walk(str(path)):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except FileNotFoundError:
                    pass
    return total


def free_bytes(path, quota_command=None):
    """Smaller of df available bytes and an optional command printing available bytes on its last line."""
    result = run(['df', '-B1', '--output=avail', str(path)], 60)
    if result['returncode'] != 0:
        return None, {'df': result}
    values = {'df_avail': int(result['stdout'].split()[-1])}
    if quota_command:
        quota = run(list(quota_command), 60)
        token = quota['stdout'].strip().splitlines()[-1].strip() if quota['returncode'] == 0 and quota['stdout'].strip() else ''
        if not token.isdigit():
            return None, dict(values, quota=quota)
        values['quota_avail'] = int(token)
    return min(values.values()), values
