"""Wake one exact, already-authorized agent session once when a watched campaign is DONE.

Reads the HPO watcher's DONE.json / ATTENTION.json (code/clmpt_suite/ops/watch.py) or an explicitly labelled
production completion file, and when the campaign is genuinely done sends a prepared message to ONE ccmux
session, at most once per completion. It never starts, restarts or creates sessions, never changes jobs,
never acknowledges ATTENTION, never reads credentials or transcripts.

Completion. HPO DONE: {time, last_cycle} with last_cycle.returncode 0, timeout false, reasons [] and
json(last_cycle.stdout).status == 'done'. Production DONE: {schema: 'clmpt-suite-completion-v1', status:
'done', campaign: str}. Anything else is INVALID_DONE (recorded, user notified once, no wake). A failed
or exited job is never a completion.

Target. Before every send, `ccmux show --json` is read twice (decision and immediately before send); the
entry whose id equals --session-id must also have nativeSessionId == --native-session-id and
cwd == --cwd, and its status field must be exactly 'idle'. Busy, approval-waiting or unknown status:
READY (wait, retry next interval). Missing or mismatched: NEEDS_SESSION_MATCH (user notified once; never
retargeted). Send: an intent marker is fsynced first; return code 0 gives SENT; anything else, or a marker
found on startup, gives SEND_AMBIGUOUS, which is never retried (user notified once).

ATTENTION.json: recorded and the user notified once per file content; no wake, no other action.

State: --state-dir holds watch.lock (exclusive flock), state.json (per-completion outcome keyed by the
sha256 of the completion file, with message and helper hashes) and events.jsonl. --dry-run never invokes
ccmux send or notify (would-be calls are recorded); with --mock-show it never invokes ccmux at all.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

CCMUX = '/home/zhapacfp/.local/bin/ccmux'
PRODUCTION_SCHEMA = 'clmpt-suite-completion-v1'
TERMINAL = ('SENT', 'SEND_AMBIGUOUS', 'DRY_RUN_WOULD_SEND', 'INVALID_DONE')
MESSAGE_LIMIT = 8192


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def put(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def completion(path):
    """Return (key, kind, detail) for a valid completion, or raise ValueError with the reason."""
    data = path.read_bytes()
    value = json.loads(data)
    if value.get('schema') == PRODUCTION_SCHEMA:
        if value.get('status') != 'done' or not isinstance(value.get('campaign'), str):
            raise ValueError('production completion without status done and campaign')
        return sha_bytes(data), 'production', {'campaign': value['campaign']}
    cycle = value.get('last_cycle')
    if not isinstance(cycle, dict):
        raise ValueError('DONE without last_cycle')
    if cycle.get('returncode') != 0 or cycle.get('timeout') is not False or cycle.get('reasons') != []:
        raise ValueError('last cycle failed: returncode=%r timeout=%r reasons=%r' % (cycle.get('returncode'), cycle.get('timeout'), cycle.get('reasons')))
    status = json.loads(cycle['stdout']).get('status')
    if status != 'done':
        raise ValueError('campaign status is %r, not done' % status)
    summary = json.loads(cycle['stdout'])
    return sha_bytes(data), 'hpo', {'code_revision': summary.get('code_revision'), 'tick_time': summary.get('time')}


class Watcher:
    def __init__(self, args, invoke=None):
        self.args = args
        self.state_dir = Path(args.state_dir)
        self.invoke = invoke or self.run_ccmux
        self.message = Path(args.message_file).read_bytes()
        assert 0 < len(self.message) <= MESSAGE_LIMIT, 'message file must be non-empty and at most %d bytes' % MESSAGE_LIMIT
        self.text = self.message.decode('utf-8')
        self.helper_sha256 = sha_bytes(Path(__file__).read_bytes())

    def run_ccmux(self, argv):
        result = subprocess.run([CCMUX] + argv, capture_output=True, text=True, timeout=60)
        return result.returncode, result.stdout

    def log(self, event):
        event = dict(event, time=time.time())
        with (self.state_dir / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def load(self):
        path = self.state_dir / 'state.json'
        return json.loads(path.read_text()) if path.exists() else {'completions': {}, 'attention': {}, 'notified': []}

    def notify(self, state, key, text):
        """At most one user notification per key; never in dry-run."""
        if key in state['notified']:
            return
        state['notified'].append(key)
        if self.args.dry_run:
            self.log({'event': 'would_notify', 'key': key, 'text': text})
        else:
            code, _ = self.invoke(['notify', text])
            self.log({'event': 'notified', 'key': key, 'returncode': code})

    def show(self):
        if self.args.mock_show:
            return json.loads(Path(self.args.mock_show).read_text())
        code, stdout = self.invoke(['show', '--json'])
        if code != 0:
            raise RuntimeError('ccmux show failed with %r' % code)
        return json.loads(stdout)

    def target(self):
        """Classify the configured session: 'idle', 'busy:<status>' or 'mismatch:<reason>'."""
        inventory = self.show()
        rows = inventory if isinstance(inventory, list) else next((inventory[key] for key in ('sessions', 'agents', 'items') if isinstance(inventory.get(key), list)), [inventory])
        matches = [row for row in rows if isinstance(row, dict) and row.get('id') == self.args.session_id]
        if len(matches) != 1:
            return 'mismatch:%d entries with id %s' % (len(matches), self.args.session_id)
        row = matches[0]
        if row.get('nativeSessionId') != self.args.native_session_id or row.get('cwd') != self.args.cwd:
            return 'mismatch:nativeSessionId or cwd differs'
        if self.args.tmux_pane and row.get('tmuxPane') != self.args.tmux_pane:
            return 'mismatch:tmuxPane differs'
        status = row.get('status', row.get('state'))
        return 'idle' if status == 'idle' else 'busy:%r' % (status,)

    def cycle(self):
        state = self.load()
        attention = Path(self.args.attention_file)
        if attention.exists():
            key = 'attention:' + sha_bytes(attention.read_bytes())
            if key not in state['attention']:
                try:
                    reasons = json.loads(attention.read_text()).get('reasons')
                except json.JSONDecodeError:
                    reasons = 'unreadable'
                state['attention'][key] = {'reasons': reasons, 'seen': time.time()}
                self.log({'event': 'attention', 'key': key, 'reasons': reasons})
                self.notify(state, key, 'LinkGDA campaign watcher ATTENTION (%s): human review needed; no automatic action taken.' % (reasons,))
        outcome = 'WAITING_FOR_DONE'
        done = Path(self.args.done_file)
        if done.exists():
            try:
                key, kind, detail = completion(done)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                key, kind, detail = 'invalid:' + sha_bytes(done.read_bytes()), 'invalid', {'reason': str(error)}
            record = state['completions'].setdefault(key, {'kind': kind, 'detail': detail, 'status': None, 'source': str(done),
                                                           'message_sha256': sha_bytes(self.message), 'helper_sha256': self.helper_sha256})
            if record['status'] == 'SENDING':
                record['status'] = 'SEND_AMBIGUOUS'
                self.log({'event': 'stale_intent_marker', 'key': key})
                self.notify(state, key + ':ambiguous', 'LinkGDA completion wake may or may not have been delivered; not retried. Check the session.')
            if record['status'] in TERMINAL:
                outcome = record['status']
            elif kind == 'invalid':
                record['status'] = outcome = 'INVALID_DONE'
                self.log({'event': 'invalid_done', 'key': key, 'reason': detail['reason']})
                self.notify(state, key, 'LinkGDA DONE file is not a valid completion (%s); no wake sent.' % detail['reason'])
            else:
                outcome = self.wake(state, key, record)
        put(self.state_dir / 'state.json', state)
        return outcome

    def wake(self, state, key, record):
        verdict = self.target()
        record['last_target_check'] = {'verdict': verdict, 'time': time.time()}
        if verdict.startswith('mismatch'):
            record['status'] = 'NEEDS_SESSION_MATCH'
            self.log({'event': 'needs_session_match', 'key': key, 'verdict': verdict})
            self.notify(state, key + ':session', 'LinkGDA campaign is done but the authorized session %s was not found with matching identity; no wake sent.' % self.args.session_id)
            return 'NEEDS_SESSION_MATCH'
        if verdict != 'idle':
            record['status'] = 'READY'
            self.log({'event': 'ready_session_busy', 'key': key, 'verdict': verdict})
            return 'READY'
        if self.target() != 'idle':
            record['status'] = 'READY'
            return 'READY'
        argv = ['send', self.args.session_id, self.text]
        if self.args.dry_run:
            record['status'] = 'DRY_RUN_WOULD_SEND'
            self.log({'event': 'would_send', 'key': key, 'argv_without_text': argv[:2], 'message_sha256': sha_bytes(self.message)})
            return record['status']
        record['status'] = 'SENDING'
        record['intent'] = {'time': time.time(), 'session_id': self.args.session_id, 'message_sha256': sha_bytes(self.message)}
        put(self.state_dir / 'state.json', state)
        try:
            code, _ = self.invoke(argv)
        except Exception as error:
            code = repr(error)
        record['send_returncode'] = code
        record['status'] = 'SENT' if code == 0 else 'SEND_AMBIGUOUS'
        self.log({'event': 'sent' if code == 0 else 'send_ambiguous', 'key': key, 'returncode': code})
        if code != 0:
            self.notify(state, key + ':ambiguous', 'LinkGDA completion wake send was ambiguous (%r); not retried.' % (code,))
        return record['status']


def watch(args, invoke=None):
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / 'watch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({'status': 'another completion watcher holds %s' % (state_dir / 'watch.lock')}))
        return 2
    watcher = Watcher(args, invoke)
    while True:
        outcome = watcher.cycle()
        print(json.dumps({'time': time.time(), 'outcome': outcome}), flush=True)
        if args.once or outcome in TERMINAL:
            return 0 if outcome != 'SEND_AMBIGUOUS' else 1
        time.sleep(args.interval)


def self_check(output):
    """Mocked inventories and DONE files; any attempted real ccmux call fails the check."""
    calls = []

    def forbidden(argv):
        calls.append(argv)
        raise AssertionError('ccmux must not be invoked in the self-check')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='selfcheck-fixtures-', dir=Path(output).parent))
    session = {'id': 'codex_pane15', 'nativeSessionId': '01a11cdf-be0c-7782-a92e-ecfe75666db1', 'cwd': '/home/zhapacfp/projects/link-gda', 'tmuxPane': '%15'}
    done_stdout = json.dumps({'status': 'done', 'code_revision': 'abc', 'time': 1.0})
    good_done = {'time': 1.0, 'last_cycle': {'returncode': 0, 'timeout': False, 'reasons': [], 'stdout': done_stdout}}
    cases = {
        'idle_match_would_send_once': ([dict(session, status='idle')], good_done, None, 2, ['DRY_RUN_WOULD_SEND', 'DRY_RUN_WOULD_SEND'], 1, 0),
        'working_session_ready_no_send': ([dict(session, status='working')], good_done, None, 2, ['READY', 'READY'], 0, 0),
        'awaiting_approval_ready_no_send': ({'sessions': [dict(session, status='awaiting_approval')]}, good_done, None, 1, ['READY'], 0, 0),
        'native_mismatch_notify_once': ([dict(session, nativeSessionId='other', status='idle')], good_done, None, 2, ['NEEDS_SESSION_MATCH', 'NEEDS_SESSION_MATCH'], 0, 1),
        'cwd_mismatch': ([dict(session, cwd='/elsewhere', status='idle')], good_done, None, 1, ['NEEDS_SESSION_MATCH'], 0, 1),
        'missing_session': ([dict(session, id='codex_pane16', status='idle')], good_done, None, 1, ['NEEDS_SESSION_MATCH'], 0, 1),
        'not_done_status_invalid': ([dict(session, status='idle')], dict(good_done, last_cycle=dict(good_done['last_cycle'], stdout=json.dumps({'status': 'ok'}))), None, 2, ['INVALID_DONE', 'INVALID_DONE'], 0, 1),
        'failed_cycle_invalid': ([dict(session, status='idle')], dict(good_done, last_cycle=dict(good_done['last_cycle'], returncode=1)), None, 1, ['INVALID_DONE'], 0, 1),
        'no_done_attention_notify_only': ([dict(session, status='idle')], None, {'reasons': ['halted']}, 2, ['WAITING_FOR_DONE', 'WAITING_FOR_DONE'], 0, 1),
        'production_schema_accepted': ([dict(session, status='idle')], {'schema': PRODUCTION_SCHEMA, 'status': 'done', 'campaign': 'prod'}, None, 1, ['DRY_RUN_WOULD_SEND'], 1, 0),
        'unlabelled_production_rejected': ([dict(session, status='idle')], {'status': 'done', 'campaign': 'prod'}, None, 1, ['INVALID_DONE'], 0, 1),
        'stale_intent_not_retried': ([dict(session, status='idle')], good_done, None, 1, ['SEND_AMBIGUOUS'], 0, 1),
    }
    results = {}
    for name, (inventory, done, attention, cycles, expected, sends, notifications) in cases.items():
        case = root / name
        case.mkdir()
        (case / 'show.json').write_text(json.dumps(inventory))
        (case / 'message.txt').write_text('mock message')
        if done is not None:
            (case / 'DONE.json').write_text(json.dumps(done))
        if attention is not None:
            (case / 'ATTENTION.json').write_text(json.dumps(attention))
        args = argparse.Namespace(done_file=str(case / 'DONE.json'), attention_file=str(case / 'ATTENTION.json'), state_dir=str(case / 'state'),
                                  session_id=session['id'], native_session_id=session['nativeSessionId'], cwd=session['cwd'], tmux_pane='%15',
                                  message_file=str(case / 'message.txt'), interval=30, once=True, dry_run=True, mock_show=str(case / 'show.json'))
        (case / 'state').mkdir()
        if name == 'stale_intent_not_retried':
            key = sha_bytes((case / 'DONE.json').read_bytes())
            put(case / 'state' / 'state.json', {'completions': {key: {'kind': 'hpo', 'detail': {}, 'status': 'SENDING'}}, 'attention': {}, 'notified': []})
        watcher = Watcher(args, forbidden)
        outcomes = [watcher.cycle() for _ in range(cycles)]
        events = [json.loads(line) for line in (case / 'state' / 'events.jsonl').read_text().splitlines()] if (case / 'state' / 'events.jsonl').exists() else []
        observed_sends = sum(event['event'] == 'would_send' for event in events)
        observed_notifications = sum(event['event'] == 'would_notify' for event in events)
        results[name] = {'outcomes': outcomes, 'would_send': observed_sends, 'would_notify': observed_notifications,
                         'passed': outcomes == expected and observed_sends == sends and observed_notifications == notifications}
    lock_dir = root / 'lock'
    lock_dir.mkdir()
    holder = (lock_dir / 'watch.lock').open('a')
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_args = argparse.Namespace(**dict(vars(args), state_dir=str(lock_dir), once=True))
    results['second_watcher_refused_by_flock'] = {'returncode': watch(lock_args, forbidden)}
    results['second_watcher_refused_by_flock']['passed'] = results['second_watcher_refused_by_flock']['returncode'] == 2
    receipt = {'helper': str(Path(__file__).resolve()), 'helper_sha256': sha_bytes(Path(__file__).read_bytes()), 'python': sys.version.split()[0],
               'temporary_root': str(root), 'cases': results, 'real_ccmux_invocations': len(calls),
               'passed': all(row['passed'] for row in results.values()) and not calls,
               'scope': 'mocked ccmux inventory and DONE/ATTENTION files; no real send or notification was possible (any invocation fails the check)'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'receipt': str(output), 'passed': receipt['passed'], 'real_ccmux_invocations': len(calls)}))
    return 0 if receipt['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-check', metavar='RECEIPT', help='run mocked scenarios only and write a receipt')
    parser.add_argument('--done-file')
    parser.add_argument('--attention-file')
    parser.add_argument('--state-dir')
    parser.add_argument('--session-id')
    parser.add_argument('--native-session-id')
    parser.add_argument('--cwd')
    parser.add_argument('--tmux-pane', help='optional additional exact match')
    parser.add_argument('--message-file')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--mock-show', help='dry-run only: JSON file used instead of ccmux show --json')
    args = parser.parse_args()
    if args.self_check:
        return self_check(args.self_check)
    for name in ('done_file', 'attention_file', 'state_dir', 'session_id', 'native_session_id', 'cwd', 'message_file'):
        assert getattr(args, name), '--%s is required' % name.replace('_', '-')
    for name in ('done_file', 'attention_file', 'state_dir', 'cwd', 'message_file'):
        assert Path(getattr(args, name)).is_absolute(), '--%s must be absolute' % name.replace('_', '-')
    assert not args.mock_show or args.dry_run, '--mock-show requires --dry-run'
    assert args.interval >= 30
    return watch(args)


if __name__ == '__main__':
    raise SystemExit(main())
