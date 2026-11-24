"""Wake the one authorized research session when production TEST readouts complete, on genuine attention, and on
genuine DONE. Read-only over the production bridge's persisted state; never touches jobs, the bridge or remotes.

Sources (all local files written by production_ops/bridge.py; nothing here runs ssh):
  bridge-state/last-cycle.json            counts (orix, ibex, ibex_inference, archived), attention, done
  bridge-state/archives/<unit>.json       per-unit archive receipt verified on Orix; files include inference/* hashes
  bridge-state/transfers/archive-<unit>.item.json   IBEX inventory item at archive launch (inference verified on IBEX)
  bridge-state/DONE.json                  written by `bridge.py loop` only when its cycle reports done
  --inventory-file (optional)             a persisted `bridge.py status` snapshot; ibex.test_states / ibex.archivable

Wake. A completed TEST readout is a unit with a hash of inference/INFERENCE_COMPLETE.json. New units, a rise in the
counted completions beyond the identified ones, new genuine attention reasons and a genuine DONE are batched; one
batch is sent at most every --min-gap seconds (default 900). Each batch writes a manifest (all current unit ids and
hashes, frozen recipe hashes, storage decision, counts) before sending, and is never resent. The target check, idle-only
rule, the twice-read inventory, intent marker and SEND_AMBIGUOUS handling are completion_watch.Watcher's, reused
unchanged; the authorized identity is fixed below and cannot be changed outside --dry-run.

Attention is genuine when it is in the current last-cycle.json (inventory_unavailable only after two consecutive
cycles), or the bridge's last cycle is older than --stale-seconds. Each reason notifies the user once and is put in
the next wake. Nothing is acknowledged, resubmitted, fixed or deleted.

DONE is genuine only when DONE.json is byte-identical to last-cycle.json, has no wrapper schema, done true, dry_run
false, empty attention, and exact counts against prod/bundle/BUNDLE.json: every pretrain unit complete or reused,
every query unit complete, TEST complete and archived, and an Orix-verified archive receipt with an inference hash
held locally for every query unit. Anything else is INVALID_DONE (user notified once, no wake).
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import completion_watch as base  # noqa: E402
from completion_watch import MESSAGE_LIMIT, Watcher, put, sha_bytes  # noqa: E402

COMPLETION = Path('/home/zhapacfp/projects/link-gda/notes/experiment-roadmap-20261123/execution/completion-20261124')
BRIDGE_STATE = COMPLETION / 'production' / 'bridge-state'
PROD_DIR = COMPLETION / 'production' / 'prod'
MONITOR = COMPLETION / 'monitor'
STATE_DIR = MONITOR / 'state'
TEMPLATE = MONITOR / 'wake-template.txt'
INVENTORY = MONITOR / 'inventory' / 'status-latest.json'
SESSION = {'session_id': 'codex_pane15', 'native_session_id': '01a11cdf-be0c-7782-a92e-ecfe75666db1', 'cwd': '/home/zhapacfp/projects/link-gda'}
FROZEN = ('hpo/recipe.frozen.json', 'hpo/final.frozen.json', 'hpo/frozen-manifest.json', 'hpo/selection-decision.json', 'bundle/BUNDLE.json')
READOUT = 'inference/INFERENCE_COMPLETE.json'
CONFIRM_TWICE = ('inventory_unavailable',)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None, None
    data = path.read_bytes()
    return json.loads(data), sha_bytes(data)


def file_sha(path):
    path = Path(path)
    return sha_bytes(path.read_bytes()) if path.exists() else None


def totals(prod_dir):
    bundle, _ = load_json(Path(prod_dir) / 'bundle' / 'BUNDLE.json')
    return {'pretrain': len(bundle['units']['pretrain']), 'query': len(bundle['units']['query'])}


def readouts(bridge_state, inventory_file):
    """unit -> {sha256 of INFERENCE_COMPLETE.json, inference file hashes, where it was seen}; plus units the inventory
    calls complete without a hash yet."""
    bridge_state = Path(bridge_state)
    found, unhashed = {}, set()
    for path in sorted((bridge_state / 'transfers').glob('archive-*.item.json')):
        value, _ = load_json(path)
        if value.get('kind') == 'archive' and READOUT in value['item']['files']:
            found[value['unit']] = {'sha256': value['item']['files'][READOUT], 'source': 'ibex_verified_archive_item', 'location': value['item']['sources'].get('inference'),
                                    'files': {name: digest for name, digest in value['item']['files'].items() if name.startswith('inference/')}}
    archived = set()
    for path in sorted((bridge_state / 'archives').glob('*.json')):
        value, _ = load_json(path)
        verified = value.get('verified') or {}
        if verified.get('verified_on_archive') is True and READOUT in verified.get('files', {}):
            archived.add(path.stem)
            found[path.stem] = {'sha256': verified['files'][READOUT], 'source': 'orix_verified_archive', 'location': verified.get('archive'),
                                'files': {name: digest for name, digest in verified['files'].items() if name.startswith('inference/')}}
    if inventory_file and Path(inventory_file).exists():
        snapshot, _ = load_json(inventory_file)
        ibex = (snapshot or {}).get('ibex') or {}
        for unit, item in (ibex.get('archivable') or {}).items():
            if unit not in found and READOUT in item.get('files', {}):
                found[unit] = {'sha256': item['files'][READOUT], 'source': 'status_snapshot_archivable', 'location': item['sources'].get('inference'),
                               'files': {name: digest for name, digest in item['files'].items() if name.startswith('inference/')}}
        unhashed = {unit for unit, state in (ibex.get('test_states') or {}).items() if state == 'complete' and unit not in found}
    return found, sorted(unhashed), archived


def genuine_done(bridge_state, prod_dir, archived):
    """(True, detail) or (False, reason) for bridge-state/DONE.json."""
    bridge_state = Path(bridge_state)
    data = (bridge_state / 'DONE.json').read_bytes()
    value = json.loads(data)
    expected = totals(prod_dir)
    if not isinstance(value, dict) or 'schema' in value:
        return False, 'DONE.json is a wrapper, not the bridge summary'
    if not (bridge_state / 'last-cycle.json').exists() or data != (bridge_state / 'last-cycle.json').read_bytes():
        return False, 'DONE.json differs from last-cycle.json'
    if value.get('done') is not True or value.get('dry_run') is not False or value.get('attention') != []:
        return False, 'done=%r dry_run=%r attention=%r' % (value.get('done'), value.get('dry_run'), value.get('attention'))
    counts = value.get('counts') or {}
    orix, ibex, test = counts.get('orix') or {}, counts.get('ibex') or {}, counts.get('ibex_inference') or {}
    checks = {'orix_total': sum(orix.values()) == expected['pretrain'],
              'orix_complete_or_reused': orix.get('complete', 0) + orix.get('reused', 0) == expected['pretrain'],
              'ibex_complete': ibex.get('complete') == sum(ibex.values()) == expected['query'],
              'test_complete': test.get('complete') == sum(test.values()) == expected['query'],
              'archived_count': counts.get('archived') == expected['query'],
              'archive_receipts_local': len(archived) == expected['query']}
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        return False, 'counts not exact: %s (expected %s)' % (', '.join(failed), expected)
    return True, {'done_sha256': sha_bytes(data), 'counts': counts, 'expected': expected}


def empty_state():
    return {'announced': {}, 'pending': {'units': {}, 'attention': [], 'count_gap': None, 'done': None}, 'batches': [], 'sending': None,
            'attention_seen': {}, 'attention_candidate': {}, 'invalid_done': {}, 'notified': [], 'last_send_time': 0, 'last_count_reported': 0,
            'started': None, 'finished': None}


class ProductionWatcher(Watcher):
    def __init__(self, args, invoke=None, clock=time.time):
        self.args = args
        self.state_dir = Path(args.state_dir)
        self.invoke = invoke or self.run_ccmux
        self.clock = clock
        self.template = Path(args.template).read_text()
        assert 0 < len(self.template.encode()) <= MESSAGE_LIMIT // 2, 'wake template must be non-empty and at most %d bytes' % (MESSAGE_LIMIT // 2)
        self.helper_sha256 = {'production_watch': file_sha(__file__), 'completion_watch': file_sha(base.__file__)}

    def load(self):
        path = self.state_dir / 'state.json'
        state = json.loads(path.read_text()) if path.exists() else empty_state()
        if state['started'] is None:
            state['started'] = self.clock()
        return state

    def attention(self, state, last, last_sha):
        bridge_state = Path(self.args.bridge_state)
        now = self.clock()
        reasons = []
        if last is None:
            fallback, _ = load_json(bridge_state / 'ATTENTION.json')
            reasons += ['attention_file:%s' % reason for reason in (fallback or {}).get('reasons', [])]
            if now - state['started'] > self.args.stale_seconds:
                reasons.append('bridge_state_missing')
            return reasons
        if now - last['time'] > self.args.stale_seconds:
            reasons.append('bridge_stale_since:%d' % int(last['time']))
        for reason in last.get('attention', []):
            if reason not in CONFIRM_TWICE:
                reasons.append(reason)
                continue
            first = state['attention_candidate'].setdefault(reason, last_sha)
            if first != last_sha:
                reasons.append('%s:confirmed_since:%s' % (reason, first[:12]))
        for reason in CONFIRM_TWICE:
            if reason not in last.get('attention', []):
                state['attention_candidate'].pop(reason, None)
        return reasons

    def observe(self, state):
        bridge_state = Path(self.args.bridge_state)
        last, last_sha = load_json(bridge_state / 'last-cycle.json')
        found, unhashed, archived = readouts(bridge_state, self.args.inventory_file)
        pending = state['pending']
        for reason in self.attention(state, last, last_sha):
            if reason not in state['attention_seen']:
                state['attention_seen'][reason] = self.clock()
                pending['attention'].append(reason)
                self.log({'event': 'attention', 'reason': reason})
                self.notify(state, 'attention:' + reason, 'LinkGDA production ATTENTION (%s): human review needed; the watcher takes no corrective action.' % reason)
        for unit, info in sorted(found.items()):
            known = state['announced'].get(unit) or pending['units'].get(unit)
            if known is None:
                pending['units'][unit] = info
                self.log({'event': 'new_readout', 'unit': unit, 'sha256': info['sha256'], 'source': info['source']})
            elif known['sha256'] != info['sha256']:
                reason = 'test_readout_hash_changed:%s' % unit
                if reason not in state['attention_seen']:
                    state['attention_seen'][reason] = self.clock()
                    pending['attention'].append(reason)
                    self.notify(state, 'attention:' + reason, 'LinkGDA TEST readout hash changed for %s; no action taken.' % unit)
        counted = (((last or {}).get('counts') or {}).get('ibex_inference') or {}).get('complete') or 0
        identified = len(set(state['announced']) | set(pending['units']))
        if counted > identified and counted > state['last_count_reported']:
            pending['count_gap'] = {'counted_complete': counted, 'identified': identified, 'complete_without_hash_in_snapshot': unhashed}
        done = bridge_state / 'DONE.json'
        if done.exists():
            key = file_sha(done)
            if key not in state['invalid_done'] and pending['done'] is None and not state.get('finished'):
                try:
                    ok, detail = genuine_done(bridge_state, self.args.prod_dir, archived)
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    ok, detail = False, 'unreadable DONE: %r' % (error,)
                if ok:
                    pending['done'] = detail
                    self.log({'event': 'genuine_done', 'sha256': key})
                else:
                    state['invalid_done'][key] = detail
                    self.log({'event': 'invalid_done', 'sha256': key, 'reason': detail})
                    self.notify(state, 'invalid_done:' + key, 'LinkGDA bridge DONE.json is not a genuine completion (%s); no wake sent.' % detail)
        return {'last_cycle_sha256': last_sha, 'last_cycle_time': (last or {}).get('time'), 'counts': (last or {}).get('counts'), 'found': found, 'unhashed': unhashed}

    def frozen(self):
        return {name: file_sha(Path(self.args.prod_dir) / name) for name in FROZEN}

    def manifest(self, state, observation, batch_id):
        pending = state['pending']
        current = dict(observation['found'])
        return {'batch': batch_id, 'time': self.clock(), 'session': SESSION if not self.args.dry_run else {k: getattr(self.args, k) for k in SESSION},
                'new_units': {unit: pending['units'][unit] for unit in sorted(pending['units'])},
                'all_identified_units': {unit: {'sha256': info['sha256'], 'source': info['source'], 'location': info['location']} for unit, info in sorted(current.items())},
                'count_gap': pending['count_gap'], 'attention': pending['attention'], 'done': pending['done'],
                'counts': observation['counts'], 'last_cycle_sha256': observation['last_cycle_sha256'], 'frozen_sha256': self.frozen(),
                'storage_decision_at_launch': 'null; every reportable-deletion flag off (production/BRIDGE-SERVICE.md)',
                'helper_sha256': self.helper_sha256, 'template_sha256': sha_bytes(self.template.encode())}

    def message(self, batch_id, manifest_path, manifest_sha, manifest):
        kinds = [name for name, present in (('test_readouts', manifest['new_units']), ('count_gap', manifest['count_gap']),
                                            ('attention', manifest['attention']), ('DONE', manifest['done'])) if present]
        head = ['LinkGDA production watch batch %s (%s).' % (batch_id, ', '.join(kinds)),
                'Manifest: %s sha256 %s (all ids, full hashes, frozen-recipe hashes, counts).' % (manifest_path, manifest_sha)]
        if manifest['done']:
            head.append('Bridge DONE verified with exact counts %s.' % json.dumps(manifest['done']['expected'], sort_keys=True))
        if manifest['attention']:
            joined = '; '.join(manifest['attention'])
            head.append('Attention (%d): %s.' % (len(manifest['attention']), joined if len(joined) <= 1200 else joined[:1200] + ' ... see manifest'))
        if manifest['count_gap']:
            head.append('Counted TEST complete %d, identified locally %d.' % (manifest['count_gap']['counted_complete'], manifest['count_gap']['identified']))
        head.append('Identified TEST readouts in total: %d.' % len(manifest['all_identified_units']))
        tail = '\n' + self.template.strip() + '\n'
        units = ['%s=%s' % (unit, info['sha256'][:12]) for unit, info in manifest['new_units'].items()]
        budget = MESSAGE_LIMIT - len(('\n'.join(head) + tail).encode()) - 200
        listed = []
        for entry in units:
            if len(' '.join(listed + [entry]).encode()) > budget:
                break
            listed.append(entry)
        if units:
            head.append('New TEST readouts (%d; INFERENCE_COMPLETE sha256 prefix): %s%s' % (len(units), ' '.join(listed), '' if len(listed) == len(units) else ' ... %d more in the manifest' % (len(units) - len(listed))))
        text = '\n'.join(head) + tail
        assert len(text.encode()) <= MESSAGE_LIMIT
        return text

    def settle(self, state, record, status):
        pending = state['pending']
        for unit in record['units']:
            info = pending['units'].pop(unit, None) or {'sha256': record['unit_sha256'][unit]}
            state['announced'][unit] = {'sha256': info['sha256'], 'batch': record['id'], 'status': status}
        for reason in record['attention']:
            if reason in pending['attention']:
                pending['attention'].remove(reason)
        if record['count_gap'] is not None:
            state['last_count_reported'] = max(state['last_count_reported'], record['count_gap']['counted_complete'])
            pending['count_gap'] = None
        if record['done']:
            pending['done'] = None
            state['finished'] = status
        record['status'] = status
        for batch in state['batches']:
            if batch['id'] == record['id']:
                batch.update(record)
        state['last_send_time'] = record.get('sent_time') or self.clock()
        state['sending'] = None

    def cycle(self):
        state = self.load()
        if state['sending'] is not None:
            record = state['sending']
            self.log({'event': 'stale_intent_marker', 'batch': record['id']})
            self.settle(state, record, 'SEND_AMBIGUOUS')
            self.notify(state, record['id'] + ':ambiguous', 'LinkGDA production wake batch %s may or may not have been delivered; not retried.' % record['id'])
        observation = self.observe(state)
        outcome = self.wake(state, observation) if any(state['pending'][key] for key in ('units', 'attention', 'count_gap', 'done')) else 'WAITING'
        if state.get('finished') and outcome not in ('READY', 'RATE_LIMITED', 'NEEDS_SESSION_MATCH'):
            outcome = 'DONE_' + state['finished']
        put(self.state_dir / 'state.json', state)
        return outcome

    def wake(self, state, observation):
        if self.clock() - state['last_send_time'] < self.args.min_gap:
            return 'RATE_LIMITED'
        verdict = self.target()
        if verdict.startswith('mismatch'):
            self.log({'event': 'needs_session_match', 'verdict': verdict})
            self.notify(state, 'session:' + verdict, 'LinkGDA production results are waiting but session %s was not found with matching identity (%s); no wake sent.' % (SESSION['session_id'], verdict))
            return 'NEEDS_SESSION_MATCH'
        if verdict != 'idle' or self.target() != 'idle':
            self.log({'event': 'ready_session_busy', 'verdict': verdict})
            return 'READY'
        pending = state['pending']
        batch_id = '%04d' % (len(state['batches']) + 1)
        manifest = self.manifest(state, observation, batch_id)
        data = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
        manifest_path = self.state_dir / 'batches' / ('batch-%s-%s.json' % (batch_id, sha_bytes(data)[:12]))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(data)
        text = self.message(batch_id, manifest_path, sha_bytes(data), manifest)
        record = {'id': batch_id, 'manifest': str(manifest_path), 'manifest_sha256': sha_bytes(data), 'message_sha256': sha_bytes(text.encode()),
                  'units': sorted(pending['units']), 'unit_sha256': {unit: info['sha256'] for unit, info in pending['units'].items()},
                  'attention': list(pending['attention']), 'count_gap': pending['count_gap'], 'done': pending['done'] is not None, 'status': 'SENDING'}
        state['batches'].append(record)
        if self.args.dry_run:
            (manifest_path.with_suffix('.message.txt')).write_text(text)
            self.log({'event': 'would_send', 'batch': batch_id, 'argv_without_text': ['send', self.args.session_id], 'message_sha256': record['message_sha256']})
            self.settle(state, record, 'DRY_RUN_WOULD_SEND')
            return 'DRY_RUN_WOULD_SEND'
        record['intent_time'] = self.clock()
        state['sending'] = record
        put(self.state_dir / 'state.json', state)
        try:
            code, _ = self.invoke(['send', SESSION['session_id'], text])
        except Exception as error:
            code = repr(error)
        record['send_returncode'] = code
        record['sent_time'] = self.clock()
        status = 'SENT' if code == 0 else 'SEND_AMBIGUOUS'
        self.log({'event': 'sent' if code == 0 else 'send_ambiguous', 'batch': batch_id, 'returncode': code})
        self.settle(state, record, status)
        state['batches'][-1] = record
        if code != 0:
            self.notify(state, batch_id + ':ambiguous', 'LinkGDA production wake batch %s send was ambiguous (%r); not retried.' % (batch_id, code))
        return status


def status(args):
    state, _ = load_json(Path(args.state_dir) / 'state.json')
    last, last_sha = load_json(Path(args.bridge_state) / 'last-cycle.json')
    found, unhashed, archived = readouts(args.bridge_state, args.inventory_file)
    print(json.dumps({'bridge_last_cycle': {'sha256': last_sha, 'time': (last or {}).get('time'), 'done': (last or {}).get('done'),
                                            'attention': (last or {}).get('attention'), 'counts': (last or {}).get('counts')},
                      'bridge_done_present': (Path(args.bridge_state) / 'DONE.json').exists(), 'identified_readouts': len(found), 'orix_verified_archives': len(archived),
                      'complete_without_hash_in_snapshot': unhashed,
                      'watcher': None if state is None else {key: state[key] for key in ('last_send_time', 'finished', 'sending', 'attention_seen', 'invalid_done')} |
                      {'announced': len(state['announced']), 'pending_units': sorted(state['pending']['units']), 'pending_attention': state['pending']['attention'],
                       'batches': [{key: batch.get(key) for key in ('id', 'status', 'manifest', 'manifest_sha256', 'message_sha256')} for batch in state['batches']]}},
                     indent=2, sort_keys=True, default=str))
    return 0


def watch(args, invoke=None, clock=time.time, sleep=time.sleep):
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / 'watch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({'status': 'another production watcher holds %s' % (state_dir / 'watch.lock')}))
        return 2
    watcher = ProductionWatcher(args, invoke, clock)
    deadline = clock() + args.max_hours * 3600
    while True:
        outcome = watcher.cycle()
        print(json.dumps({'time': clock(), 'outcome': outcome}), flush=True)
        if outcome.startswith('DONE_'):
            return 0
        if args.once:
            return 0 if outcome != 'SEND_AMBIGUOUS' else 1
        if clock() > deadline:
            print(json.dumps({'status': 'deadline reached', 'max_hours': args.max_hours}), flush=True)
            return 3
        sleep(args.interval)


def self_check(output):
    """Mocked bridge states, inventories and a fake clock; any attempted real ccmux call fails the check."""
    calls = []

    def forbidden(argv):
        calls.append(argv)
        raise AssertionError('ccmux must not be invoked in the self-check')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='selfcheck-fixtures-', dir=output.parent))
    row = dict(SESSION, id=SESSION['session_id'], nativeSessionId=SESSION['native_session_id'])
    del row['session_id'], row['native_session_id']
    query = ['q%d' % index for index in range(3)]
    results = {}

    class Clock:
        def __init__(self):
            self.now = 1_000_000.0

        def __call__(self):
            return self.now

    def fixture(name, status='idle', session=None):
        case = root / name
        (case / 'bridge' / 'transfers').mkdir(parents=True)
        (case / 'bridge' / 'archives').mkdir()
        (case / 'prod' / 'bundle').mkdir(parents=True)
        (case / 'prod' / 'bundle' / 'BUNDLE.json').write_text(json.dumps({'units': {'pretrain': {'p0': {}, 'p1': {}}, 'query': {unit: {} for unit in query}}}))
        (case / 'show.json').write_text(json.dumps([dict(row, status=status, **(session or {}))]))
        (case / 'template.txt').write_text(TEMPLATE.read_text())
        args = argparse.Namespace(bridge_state=str(case / 'bridge'), prod_dir=str(case / 'prod'), state_dir=str(case / 'state'), template=str(case / 'template.txt'),
                                  inventory_file=str(case / 'inventory.json'), session_id=SESSION['session_id'], native_session_id=SESSION['native_session_id'],
                                  cwd=SESSION['cwd'], tmux_pane=None, mock_show=str(case / 'show.json'), dry_run=True, once=True, interval=300, min_gap=900,
                                  stale_seconds=3600, max_hours=1)
        (case / 'state').mkdir()
        return case, args

    def item(case, unit, digest='a' * 64):
        files = {'training/COMPLETE.json': 'b' * 64, READOUT: digest, 'inference/predictions.npz': 'c' * 64}
        put(case / 'bridge' / 'transfers' / ('archive-%s.item.json' % unit), {'kind': 'archive', 'unit': unit, 'item': {'sources': {'inference': '/ibex/x/' + unit}, 'files': files}})

    def archive(case, unit, digest='a' * 64):
        files = {READOUT: digest, 'inference/predictions.npz': 'c' * 64}
        put(case / 'bridge' / 'archives' / (unit + '.json'), {'unit': unit, 'verified': {'unit': unit, 'verified_on_archive': True, 'files': files, 'archive': '/orix/a/' + unit}})

    def cycle_record(case, clock, attention=(), test_complete=0, done=False, archived=0, ibex_complete=0):
        summary = {'time': clock(), 'dry_run': False, 'attention': list(attention), 'done': done,
                   'counts': {'orix': {'complete': 1, 'reused': 1}, 'ibex': {'complete': ibex_complete, 'active': 3 - ibex_complete} if ibex_complete < 3 else {'complete': 3},
                              'ibex_inference': {'complete': test_complete, 'not_ready': 3 - test_complete} if test_complete < 3 else {'complete': 3}, 'archived': archived}}
        put(case / 'bridge' / 'last-cycle.json', summary)
        return summary

    def run(case, args, clock, cycles=1):
        watcher = ProductionWatcher(args, forbidden, clock)
        return [watcher.cycle() for _ in range(cycles)]

    def events(case):
        path = case / 'state' / 'events.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return sum(row['event'] == 'would_send' for row in rows), sum(row['event'] == 'would_notify' for row in rows)

    def record(name, case, outcomes, expected, sends, notifications, extra=True):
        observed = events(case)
        results[name] = {'outcomes': outcomes, 'would_send': observed[0], 'would_notify': observed[1],
                         'passed': outcomes == expected and observed == (sends, notifications) and bool(extra)}

    clock = Clock()
    case, args = fixture('empty_bridge_waits')
    record('empty_bridge_waits', case, run(case, args, clock, 2), ['WAITING', 'WAITING'], 0, 0)

    clock = Clock()
    case, args = fixture('new_readout_wakes_once_then_dedups')
    cycle_record(case, clock, test_complete=1)
    item(case, 'q0')
    outcomes = run(case, args, clock, 1)
    clock.now += 1000
    outcomes += run(case, args, clock, 1)
    state = json.loads((case / 'state' / 'state.json').read_text())
    message = Path(state['batches'][0]['manifest']).with_suffix('.message.txt').read_text()
    record('new_readout_wakes_once_then_dedups', case, outcomes, ['DRY_RUN_WOULD_SEND', 'WAITING'], 1, 0,
           'q0=aaaaaaaaaaaa' in message and 'Do not tune' in message and len(message.encode()) <= MESSAGE_LIMIT)

    clock = Clock()
    case, args = fixture('second_readout_rate_limited_then_batched')
    cycle_record(case, clock, test_complete=1)
    item(case, 'q0')
    outcomes = run(case, args, clock, 1)
    clock.now += 300
    item(case, 'q1')
    outcomes += run(case, args, clock, 1)
    clock.now += 300
    item(case, 'q2')
    outcomes += run(case, args, clock, 1)
    clock.now += 301
    outcomes += run(case, args, clock, 1)
    state = json.loads((case / 'state' / 'state.json').read_text())
    record('second_readout_rate_limited_then_batched', case, outcomes, ['DRY_RUN_WOULD_SEND', 'RATE_LIMITED', 'RATE_LIMITED', 'DRY_RUN_WOULD_SEND'], 2, 0,
           [batch['units'] for batch in state['batches']] == [['q0'], ['q1', 'q2']])

    clock = Clock()
    case, args = fixture('busy_session_ready_no_send', status='working')
    item(case, 'q0')
    outcomes = run(case, args, clock, 2)
    (case / 'show.json').write_text(json.dumps([dict(row, status='idle')]))
    outcomes += run(case, args, clock, 1)
    record('busy_session_ready_then_idle_sends', case, outcomes, ['READY', 'READY', 'DRY_RUN_WOULD_SEND'], 1, 0)

    clock = Clock()
    case, args = fixture('awaiting_approval_ready', status='awaiting_approval')
    item(case, 'q0')
    record('awaiting_approval_ready', case, run(case, args, clock, 1), ['READY'], 0, 0)

    for name, session in (('native_mismatch_notify_once', {'nativeSessionId': 'other'}), ('cwd_mismatch_notify_once', {'cwd': '/elsewhere'}),
                          ('missing_session_notify_once', {'id': 'codex_pane16'})):
        clock = Clock()
        case, args = fixture(name, session=session)
        item(case, 'q0')
        record(name, case, run(case, args, clock, 2), ['NEEDS_SESSION_MATCH', 'NEEDS_SESSION_MATCH'], 0, 1)

    clock = Clock()
    case, args = fixture('attention_notifies_once_and_wakes_once')
    cycle_record(case, clock, attention=['ibex_storage_permission_required'])
    outcomes = run(case, args, clock, 1)
    clock.now += 60
    cycle_record(case, clock, attention=['ibex_storage_permission_required'])
    outcomes += run(case, args, clock, 1)
    record('attention_notifies_once_and_wakes_once', case, outcomes, ['DRY_RUN_WOULD_SEND', 'WAITING'], 1, 1)

    clock = Clock()
    case, args = fixture('transient_inventory_unavailable_needs_two_cycles')
    cycle_record(case, clock, attention=['inventory_unavailable'])
    outcomes = run(case, args, clock, 2)
    clock.now += 300
    cycle_record(case, clock, attention=['inventory_unavailable'])
    outcomes += run(case, args, clock, 1)
    record('transient_inventory_unavailable_needs_two_cycles', case, outcomes, ['WAITING', 'WAITING', 'DRY_RUN_WOULD_SEND'], 1, 1)

    clock = Clock()
    case, args = fixture('stale_bridge_is_attention')
    cycle_record(case, clock)
    clock.now += 3601
    record('stale_bridge_is_attention', case, run(case, args, clock, 2), ['DRY_RUN_WOULD_SEND', 'WAITING'], 1, 1)

    clock = Clock()
    case, args = fixture('count_gap_wakes_with_snapshot_ids')
    cycle_record(case, clock, test_complete=2)
    item(case, 'q0')
    put(case / 'inventory.json', {'ibex': {'test_states': {'q0': 'complete', 'q1': 'complete', 'q2': 'active'}, 'archivable': {}}})
    outcomes = run(case, args, clock, 1)
    state = json.loads((case / 'state' / 'state.json').read_text())
    manifest = json.loads(Path(state['batches'][0]['manifest']).read_text())
    record('count_gap_wakes_with_snapshot_ids', case, outcomes, ['DRY_RUN_WOULD_SEND'], 1, 0,
           manifest['count_gap'] == {'counted_complete': 2, 'identified': 1, 'complete_without_hash_in_snapshot': ['q1']})

    clock = Clock()
    case, args = fixture('hash_change_is_attention')
    item(case, 'q0')
    outcomes = run(case, args, clock, 1)
    clock.now += 1000
    archive(case, 'q0', digest='d' * 64)
    outcomes += run(case, args, clock, 1)
    record('hash_change_is_attention', case, outcomes, ['DRY_RUN_WOULD_SEND', 'DRY_RUN_WOULD_SEND'], 2, 1)

    def done_fixture(name, archived_units):
        clock = Clock()
        case, args = fixture(name)
        for unit in archived_units:
            archive(case, unit)
        put(case / 'bridge' / 'DONE.json', cycle_record(case, clock, test_complete=3, done=True, archived=3, ibex_complete=3))
        return case, args, clock

    case, args, clock = done_fixture('genuine_done_wakes_and_finishes', query)
    outcomes = run(case, args, clock, 1)
    clock.now += 1000
    outcomes += run(case, args, clock, 1)
    state = json.loads((case / 'state' / 'state.json').read_text())
    record('genuine_done_wakes_and_finishes', case, outcomes, ['DONE_DRY_RUN_WOULD_SEND', 'DONE_DRY_RUN_WOULD_SEND'], 1, 0, state['batches'][0]['done'] and state['batches'][0]['units'] == query)

    case, args, clock = done_fixture('wrapper_done_rejected', query)
    put(case / 'bridge' / 'DONE.json', {'schema': base.PRODUCTION_SCHEMA, 'status': 'done', 'campaign': 'prod'})
    outcomes = run(case, args, clock, 2)
    record('wrapper_done_rejected', case, outcomes, ['DRY_RUN_WOULD_SEND', 'WAITING'], 1, 1, not json.loads((case / 'state' / 'state.json').read_text())['finished'])

    case, args, clock = done_fixture('short_archive_count_rejected', query[:2])
    record('short_archive_count_rejected', case, run(case, args, clock, 1), ['DRY_RUN_WOULD_SEND'], 1, 1, not json.loads((case / 'state' / 'state.json').read_text())['finished'])

    case, args, clock = done_fixture('done_differs_from_last_cycle_rejected', query)
    put(case / 'bridge' / 'last-cycle.json', dict(json.loads((case / 'bridge' / 'DONE.json').read_text()), time=clock() + 1))
    record('done_differs_from_last_cycle_rejected', case, run(case, args, clock, 1), ['DRY_RUN_WOULD_SEND'], 1, 1, not json.loads((case / 'state' / 'state.json').read_text())['finished'])

    clock = Clock()
    case, args = fixture('stale_intent_not_retried')
    item(case, 'q0')
    state = empty_state()
    state['pending']['units']['q0'] = {'sha256': 'a' * 64, 'source': 'x', 'location': None, 'files': {}}
    state['sending'] = {'id': '0001', 'units': ['q0'], 'unit_sha256': {'q0': 'a' * 64}, 'attention': [], 'count_gap': None, 'done': False, 'status': 'SENDING'}
    state['batches'] = [state['sending']]
    put(case / 'state' / 'state.json', state)
    outcomes = run(case, args, clock, 1)
    state = json.loads((case / 'state' / 'state.json').read_text())
    record('stale_intent_not_retried', case, outcomes, ['WAITING'], 0, 1, state['announced']['q0']['status'] == 'SEND_AMBIGUOUS')

    clock = Clock()
    case, args = fixture('message_bounded_with_many_units')
    for index in range(185):
        item(case, 'main-PFS-f%d-qs%d-full13-independent-%03d' % (index % 10, index % 10, index), digest='%064x' % index)
    outcomes = run(case, args, clock, 1)
    state = json.loads((case / 'state' / 'state.json').read_text())
    text = Path(state['batches'][0]['manifest']).with_suffix('.message.txt').read_bytes()
    manifest = json.loads(Path(state['batches'][0]['manifest']).read_text())
    record('message_bounded_with_many_units', case, outcomes, ['DRY_RUN_WOULD_SEND'], 1, 0, len(text) <= MESSAGE_LIMIT and len(manifest['new_units']) == 185 and b'more in the manifest' in text)

    holder_dir = root / 'lock'
    holder_dir.mkdir()
    holder = (holder_dir / 'watch.lock').open('a')
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    results['second_watcher_refused_by_flock'] = {'returncode': watch(argparse.Namespace(**dict(vars(args), state_dir=str(holder_dir))), forbidden)}
    results['second_watcher_refused_by_flock']['passed'] = results['second_watcher_refused_by_flock']['returncode'] == 2

    receipt = {'helper': str(Path(__file__).resolve()), 'helper_sha256': file_sha(__file__), 'completion_watch_sha256': file_sha(base.__file__),
               'template_sha256': file_sha(TEMPLATE), 'python': sys.version.split()[0], 'temporary_root': str(root), 'cases': results,
               'real_ccmux_invocations': len(calls), 'passed': all(row['passed'] for row in results.values()) and not calls,
               'scope': 'mocked bridge-state, inventory snapshots, ccmux inventory and clock; no real send or notification was possible (any invocation fails the check)'}
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'receipt': str(output), 'passed': receipt['passed'], 'failed': sorted(name for name, row in results.items() if not row['passed']),
                      'real_ccmux_invocations': len(calls)}))
    return 0 if receipt['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['watch', 'once', 'status', 'self-check'])
    parser.add_argument('--receipt', help='self-check: receipt path')
    parser.add_argument('--bridge-state', default=str(BRIDGE_STATE))
    parser.add_argument('--prod-dir', default=str(PROD_DIR))
    parser.add_argument('--state-dir', default=str(STATE_DIR))
    parser.add_argument('--template', default=str(TEMPLATE))
    parser.add_argument('--inventory-file', default=str(INVENTORY), help='optional persisted `bridge.py status` snapshot')
    parser.add_argument('--dry-run', action='store_true', help='never send or notify; requires a --state-dir other than the production one')
    parser.add_argument('--mock-show', help='dry-run only: JSON used instead of `ccmux show --json`')
    parser.add_argument('--session-id', default=SESSION['session_id'])
    parser.add_argument('--native-session-id', default=SESSION['native_session_id'])
    parser.add_argument('--cwd', default=SESSION['cwd'])
    parser.add_argument('--tmux-pane', help='optional additional exact match')
    parser.add_argument('--interval', type=float, default=300)
    parser.add_argument('--min-gap', type=float, default=900)
    parser.add_argument('--stale-seconds', type=float, default=3600)
    parser.add_argument('--max-hours', type=float, default=336)
    args = parser.parse_args()
    if args.command == 'self-check':
        assert args.receipt, '--receipt is required'
        return self_check(args.receipt)
    if args.command == 'status':
        return status(args)
    args.once = args.command == 'once'
    for name in ('bridge_state', 'prod_dir', 'state_dir', 'template'):
        assert Path(getattr(args, name)).is_absolute(), '--%s must be absolute' % name.replace('_', '-')
    if args.dry_run:
        assert Path(args.state_dir).resolve() != STATE_DIR.resolve(), 'dry-run must use its own --state-dir'
    else:
        assert {key: getattr(args, key) for key in SESSION} == SESSION, 'the authorized session identity cannot be changed outside --dry-run'
        assert not args.mock_show, '--mock-show requires --dry-run'
        assert args.min_gap >= 900 and args.interval >= 60
    return watch(args)


if __name__ == '__main__':
    raise SystemExit(main())
