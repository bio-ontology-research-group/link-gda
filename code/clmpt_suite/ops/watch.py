"""Workstation watcher: run the campaign's advance.py tick on the cluster login node over SSH at a fixed interval.

It only ever calls `advance.py tick` (and `gate` for the receipt). It never cancels, resumes, releases, acknowledges
or resubmits anything itself; tick resubmits only clean HELD pauses and never revives cancelled or failed trials.
One watcher per state directory (exclusive lock). Every cycle appends its command, return code, output and
classification to cycles.jsonl and rewrites last.json. ATTENTION.json is written (and an optional notify command
run once per new reason) when tick reports a halt or a refusal, unacknowledged trial attention, a storage stall, or
when SSH fails repeatedly. The watcher exits on DONE, on a STOP file in the state directory, or after a halt has
been reported for --halt-exit-cycles consecutive cycles.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def put(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')
    os.replace(temporary, path)


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def remote(args, action):
    command = '%s -I %s %s --campaign %s' % (shlex.quote(args.remote_python), shlex.quote(args.remote_code + '/advance.py'), action, shlex.quote(args.campaign))
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', '-o', 'ServerAliveInterval=30', args.ssh_target, command]
    try:
        result = subprocess.run(ssh, capture_output=True, text=True, timeout=args.timeout)
        return {'command': ssh, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr[-4000:], 'timeout': False}
    except subprocess.TimeoutExpired as error:
        return {'command': ssh, 'returncode': None, 'stdout': str(error.stdout or ''), 'stderr': str(error.stderr or '')[-4000:], 'timeout': True}


def classify(result):
    """Reasons that need a person, from one tick result; empty when healthy."""
    if result['timeout'] or result['returncode'] == 255:
        return ['transport']
    try:
        summary = json.loads(result['stdout'])
    except json.JSONDecodeError:
        return ['unparsable_tick_output_rc_%s' % result['returncode']]
    reasons = []
    if result['returncode'] == 2 or 'refused' in summary:
        reasons.append('refusal')
    if summary.get('status') == 'halted':
        reasons.append('halted:%s' % (summary.get('reason') or (summary.get('halt') or {}).get('reason')))
    if summary.get('unacknowledged_trial_attention'):
        reasons.append('trial_attention')
    if summary.get('budget', {}).get('storage_stall'):
        reasons.append('storage_stall')
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ssh-target', required=True, help='SSH host of the cluster login node')
    parser.add_argument('--remote-python', default='python3')
    parser.add_argument('--remote-code', required=True, help='code/clmpt_suite directory of the frozen checkout on the cluster')
    parser.add_argument('--campaign', required=True, help='campaign directory on the cluster')
    parser.add_argument('--state-dir', required=True, help='local directory for the lock, logs and receipts')
    parser.add_argument('--interval', type=int, default=600)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--transport-failures', type=int, default=6, help='consecutive SSH failures before ATTENTION')
    parser.add_argument('--halt-exit-cycles', type=int, default=6)
    parser.add_argument('--notify-command', help='JSON list; run once per new attention reason with the reason appended')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / 'watch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(json.dumps({'status': 'another watcher holds %s' % (state / 'watch.lock')}))
        return 3
    put(state / 'watcher.json', {'pid': os.getpid(), 'started': time.time(), 'argv': sys.argv, 'host': os.uname().nodename})
    notified, transport_failures, halted_cycles = set(), 0, 0
    while True:
        if (state / 'STOP').exists():
            append(state / 'cycles.jsonl', {'time': time.time(), 'event': 'stop_file'})
            return 0
        began = time.time()
        result = remote(args, 'tick')
        reasons = classify(result)
        transport_failures = transport_failures + 1 if reasons == ['transport'] else 0
        if reasons == ['transport'] and transport_failures < args.transport_failures:
            reasons = []
        halted_cycles = halted_cycles + 1 if any(reason.startswith('halted') or reason == 'refusal' for reason in reasons) else 0
        record = {'time': began, 'seconds': time.time() - began, 'returncode': result['returncode'], 'timeout': result['timeout'], 'reasons': reasons,
                  'stdout': result['stdout'][-20000:], 'stderr': result['stderr']}
        append(state / 'cycles.jsonl', record)
        put(state / 'last.json', record)
        if not reasons:
            gate = remote(args, 'gate')
            put(state / 'last-gate.json', {'time': time.time(), 'returncode': gate['returncode'], 'stdout': gate['stdout'][-20000:], 'stderr': gate['stderr']})
        new = [reason for reason in reasons if reason not in notified]
        if reasons:
            put(state / 'ATTENTION.json', {'time': time.time(), 'reasons': reasons, 'last_cycle': record,
                                           'resolution': 'a person inspects the campaign; the watcher takes no corrective action'})
            if new and args.notify_command:
                subprocess.run(json.loads(args.notify_command) + [', '.join(new)], timeout=60)
            notified.update(new)
        try:
            status = json.loads(result['stdout']).get('status')
        except json.JSONDecodeError:
            status = None
        if status == 'done':
            put(state / 'DONE.json', {'time': time.time(), 'last_cycle': record})
            return 0
        if halted_cycles >= args.halt_exit_cycles:
            append(state / 'cycles.jsonl', {'time': time.time(), 'event': 'exit_after_repeated_halt'})
            return 1
        if args.once:
            return 0 if not reasons else 1
        time.sleep(max(30, args.interval - (time.time() - began)))


if __name__ == '__main__':
    raise SystemExit(main())
