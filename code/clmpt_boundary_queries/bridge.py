"""Bridge between the pretraining campaign and the query campaign, run on a machine that reaches both (standard library only).

One cycle, under an exclusive lock held for the life of the process:
  1. Read-only inventory of the pretraining campaign: per trajectory the committed step, the milestone model
     artifacts recorded in progress.json, completion and attention markers, and the user's queued jobs.
  2. For each expected milestone that is committed and not yet delivered: download only the model-only file, the
     trajectory identity, config and commit journal; verify the model hash against progress.json and the full
     lineage with the same verify_delivery the query advancer runs (pins: pretraining protocol, sources, graph,
     trajectory config, journaled artifact).
  3. Deliver it into the staged query campaign: upload to incoming/<name>.partial, re-hash remotely, then one atomic
     rename to initializers/<name> (read-only files). Nothing else is copied; the bank is used in place.
  4. Run one advancer tick on the query cluster login node (light: metadata, squeue, sbatch) and record its summary.
  5. Gates: stop on HALT, operator stop, DONE (after saving the decisions and report inputs locally), repeated
     transport failure, an invalid delivered milestone, or no progress anywhere for the configured time.
Transport errors (ssh/rsync exit 255 or timeouts) are retried a bounded number of times for read-only calls only;
remote-mutating calls are never repeated within a cycle (the tick and the rename are idempotent across cycles).
Host names, accounts and paths come from a local JSON config, never from this file.
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import shlex
import subprocess
import time


HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location('boundary_advance', str(HERE / 'advance.py'))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
C = A.C
TRANSPORT_CODES = (255,)

PRETRAIN_INVENTORY = r'''
import glob, json, os, subprocess, sys, time
campaign = sys.argv[1]
def load(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None
out = {'time': time.time(), 'campaign': campaign, 'operator_stop': os.path.exists(os.path.join(campaign, 'OPERATOR_STOP.json')), 'trajectories': {}}
for name in sorted(os.listdir(os.path.join(campaign, 'trials'))):
    trial = os.path.join(campaign, 'trials', name)
    training = os.path.join(trial, 'training')
    progress = load(os.path.join(training, 'progress.json')) or {}
    heartbeat = load(os.path.join(training, 'heartbeat.json')) or {}
    waves = sorted(glob.glob(os.path.join(trial, 'allocations', 'wave-*')))
    stops = sorted(os.path.relpath(path, trial) for marker in ('ATTENTION.json', 'STALL_STOP.json', 'UTILIZATION_STOP.json', 'AMBIGUOUS_SUCCESSOR.json')
                   for path in glob.glob(os.path.join(trial, 'allocations', 'wave-*', marker)))
    out['trajectories'][name] = {'step': progress.get('step'), 'identity_sha256': progress.get('identity_sha256'),
        'models': {key: value for key, value in (progress.get('artifacts') or {}).items() if key.startswith('models/')},
        'complete': os.path.exists(os.path.join(training, 'COMPLETE.json')), 'training_attention': os.path.exists(os.path.join(training, 'ATTENTION.json')),
        'heartbeat_time': heartbeat.get('time'), 'waves': len(waves), 'last_wave_files': sorted(os.listdir(waves[-1])) if waves else [], 'stop_markers': stops,
        'submitted': os.path.exists(os.path.join(trial, 'submission-receipt.json'))}
try:
    result = subprocess.run(['squeue', '--me', '-h', '-o', '%i|%T|%j'], capture_output=True, text=True, timeout=60)
    out['jobs'] = [line.split('|', 2) for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else None
except Exception as error:
    out['jobs'] = None
    out['jobs_error'] = repr(error)
print(json.dumps(out))
'''

QUERY_FINALIZE = r'''
import hashlib, json, os, sys
campaign, name, admission_sha = sys.argv[1], sys.argv[2], sys.argv[3]
def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()
assert os.path.exists(os.path.join(campaign, 'frozen-manifest.json'))
partial = os.path.join(campaign, 'incoming', name + '.partial')
final = os.path.join(campaign, 'initializers', name)
source = final if os.path.isdir(final) else partial
assert os.path.realpath(source) == source
assert sha(os.path.join(source, 'ADMISSION.json')) == admission_sha
with open(os.path.join(source, 'ADMISSION.json')) as stream:
    admission = json.load(stream)
assert sorted(os.listdir(source)) == sorted(list(admission['files']) + ['ADMISSION.json'])
for file_name, digest in admission['files'].items():
    assert sha(os.path.join(source, file_name)) == digest, file_name
if source == final:
    print(json.dumps({'state': 'already_delivered', 'name': name}))
    raise SystemExit(0)
for file_name in os.listdir(partial):
    os.chmod(os.path.join(partial, file_name), 0o444)
os.rename(partial, final)
descriptor = os.open(os.path.dirname(final), os.O_RDONLY)
os.fsync(descriptor)
os.close(descriptor)
print(json.dumps({'state': 'delivered', 'name': name}))
'''


class Stop(Exception):
    def __init__(self, code, reason, detail=None):
        super().__init__(reason)
        self.code, self.reason, self.detail = code, reason, detail


def now():
    return time.time()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.%s.tmp-%d' % (path.name, os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Bridge:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text())
        self.state = Path(self.config['state_dir'])
        self.store = Path(self.config['store_dir'])
        self.protocol = json.loads((HERE / 'protocol.json').read_text())
        self.pretrain, self.query = self.config['pretrain'], self.config['query']
        self.state.mkdir(parents=True, exist_ok=True)
        self.identity = {'bridge_sha256': C.sha(Path(__file__).resolve()), 'protocol_sha256': C.sha(HERE / 'protocol.json'), 'code_dir': str(HERE),
                         'config_sha256': C.sha(self.config_path)}
        try:
            self.identity['git_head'] = C.git(HERE, 'rev-parse', 'HEAD').strip()
            self.identity['package_dirty'] = bool(C.git(HERE, 'status', '--porcelain', '--', '.').strip())
        except (subprocess.SubprocessError, OSError):
            self.identity['git_head'] = None

    def event(self, value):
        with (self.state / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(value, time=now()), sort_keys=True, default=str) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def call(self, command, read_only, timeout, stdin=None):
        """Run one transport command; read-only calls get bounded retries on transport failure only."""
        attempts = self.config['transport_attempts'] if read_only else 1
        history = []
        for attempt in range(attempts):
            try:
                result = subprocess.run(command, input=stdin, capture_output=True, text=True, timeout=timeout)
                record = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr[-4000:], 'timeout': False}
            except subprocess.TimeoutExpired:
                record = {'returncode': None, 'stdout': '', 'stderr': 'timeout after %d s' % timeout, 'timeout': True}
            history.append({key: value for key, value in record.items() if key != 'stdout'})
            transport = record['timeout'] or record['returncode'] in TRANSPORT_CODES
            if not transport:
                record['transport_error'] = False
                record['attempts'] = history
                return record
            if attempt + 1 < attempts:
                time.sleep(self.config['transport_backoff_seconds'] * (2 ** attempt))
        record['transport_error'] = True
        record['attempts'] = history
        return record

    def remote(self, side, command_parts, read_only, timeout, stdin=None):
        return self.call(list(side['ssh']) + [' '.join(shlex.quote(str(part)) for part in command_parts)], read_only, timeout, stdin)

    def rsync(self, side, source, destination, upload):
        shell = side['rsync_shell']
        remote = side['host'] + ':'
        arguments = ['rsync', '-s', '--partial', '--times', '-e', shell]
        arguments += [source, remote + destination] if upload else [remote + source, destination]
        return self.call(arguments, True, self.config['transfer_timeout_seconds'])

    def transport_failed(self, record, what):
        failures = self.read('transport-failures.json', {'consecutive': 0})
        failures = {'consecutive': failures['consecutive'] + 1, 'last': what, 'record': {key: value for key, value in record.items() if key != 'stdout'}, 'time': now()}
        put(self.state / 'transport-failures.json', failures)
        self.event({'event': 'transport_error', 'what': what, 'consecutive': failures['consecutive']})
        if failures['consecutive'] >= self.config['transport_failure_limit']:
            raise Stop(5, 'transport_failure_limit', failures)

    def transport_ok(self):
        put(self.state / 'transport-failures.json', {'consecutive': 0, 'time': now()})

    def read(self, name, default):
        path = self.state / name
        return json.loads(path.read_text()) if path.exists() else default

    def pretrain_inventory(self):
        record = self.remote(self.pretrain, [self.pretrain['python'], '-', self.pretrain['campaign']], True, 120, PRETRAIN_INVENTORY)
        if record['transport_error']:
            self.transport_failed(record, 'pretrain_inventory')
            return None
        if record['returncode'] != 0:
            raise Stop(6, 'pretrain_inventory_script_error', record['stderr'])
        return json.loads(record['stdout'].strip().splitlines()[-1])

    def delivered(self):
        return self.read('delivered.json', {})

    def fetch(self, expected, inventory):
        """Download and verify one committed milestone; returns the local delivery folder or None if not ready."""
        name = A.delivery_name(expected)
        budget = expected['initializer_update_budget']
        artifact = self.protocol['pretraining_lineage']['milestone_model_name'] % budget
        row = inventory['trajectories'].get(expected['trajectory'])
        if row is None or artifact not in row['models']:
            return None
        folder = self.store / name
        if (folder / 'ADMISSION.json').exists():
            return folder
        incoming = self.store / (name + '.incoming')
        incoming.mkdir(parents=True, exist_ok=True)
        trial = self.pretrain['campaign'].rstrip('/') + '/trials/' + expected['trajectory']
        model = 'update-%07d.pt' % budget
        for source, target in ((trial + '/training/' + artifact, model), (trial + '/training/identity.json', 'identity.json'),
                               (trial + '/config.json', 'config.json'), (trial + '/training/commits.jsonl', 'commits.jsonl')):
            if target == 'commits.jsonl' and (incoming / target).exists():
                (incoming / target).unlink()
            record = self.rsync(self.pretrain, source, str(incoming / target), upload=False)
            if record['transport_error']:
                self.transport_failed(record, 'pretrain_download:%s' % name)
                return None
            if record['returncode'] != 0:
                raise Stop(6, 'pretrain_download_failed', {'name': name, 'file': target, 'stderr': record['stderr']})
        digest = C.sha(incoming / model)
        if digest != row['models'][artifact]:
            attempts = self.read('fetch-mismatch.json', {})
            attempts[name] = attempts.get(name, 0) + 1
            put(self.state / 'fetch-mismatch.json', attempts)
            (incoming / model).unlink()
            self.event({'event': 'model_hash_mismatch', 'name': name, 'local': digest, 'progress': row['models'][artifact], 'attempt': attempts[name]})
            if attempts[name] >= 2:
                raise Stop(7, 'milestone_hash_mismatch_after_redownload', {'name': name, 'local': digest, 'progress': row['models'][artifact]})
            return None
        admission = {'schema': 'clmpt-boundary-delivery/1', 'trajectory': expected['trajectory'], 'initializer_update_budget': budget,
                     'files': {target: C.sha(incoming / target) for target in (model, 'identity.json', 'config.json', 'commits.jsonl')},
                     'lineage': {'source_campaign': self.pretrain['campaign'], 'source_trial': expected['trajectory'], 'artifact': artifact,
                                 'progress_artifact_sha256': row['models'][artifact], 'progress_step_at_fetch': row['step'],
                                 'progress_identity_sha256': row['identity_sha256'], 'fetched_at': now(), 'bridge': self.identity}}
        put(incoming / 'ADMISSION.json', admission)
        try:
            record = A.verify_delivery({'protocol': self.protocol}, expected, incoming)
        except A.Refusal as error:
            raise Stop(7, 'milestone_lineage_refused', {'name': name, 'error': str(error)})
        os.replace(incoming, folder)
        put(self.state / 'lineage' / (name + '.json'), {'record': record, 'admission_sha256': C.sha(folder / 'ADMISSION.json')})
        self.event({'event': 'milestone_fetched_and_verified', 'name': name, 'sha256': record['sha256']})
        return folder

    def query_staged(self):
        record = self.remote(self.query, ['test', '-f', self.query['campaign'].rstrip('/') + '/ROOT_RELEASE.json'], True, 60)
        if record['transport_error']:
            self.transport_failed(record, 'query_stage_check')
            return None
        return record['returncode'] == 0

    def deliver(self, name, folder):
        campaign = self.query['campaign'].rstrip('/')
        partial = campaign + '/incoming/' + name + '.partial/'
        record = self.remote(self.query, ['mkdir', '-p', partial], True, 60)
        if record['transport_error']:
            self.transport_failed(record, 'query_mkdir:%s' % name)
            return False
        for path in sorted(folder.iterdir()):
            record = self.rsync(self.query, str(path), partial + path.name, upload=True)
            if record['transport_error']:
                self.transport_failed(record, 'query_upload:%s' % name)
                return False
            if record['returncode'] != 0:
                raise Stop(6, 'query_upload_failed', {'name': name, 'file': path.name, 'stderr': record['stderr']})
        record = self.remote(self.query, [self.query['python'], '-I', '-', campaign, name, C.sha(folder / 'ADMISSION.json')], False, 600, QUERY_FINALIZE)
        if record['transport_error']:
            self.transport_failed(record, 'query_finalize:%s' % name)
            return False
        if record['returncode'] != 0:
            raise Stop(7, 'query_delivery_verification_failed', {'name': name, 'stderr': record['stderr']})
        delivered = self.delivered()
        delivered[name] = {'time': now(), 'admission_sha256': C.sha(folder / 'ADMISSION.json'), 'remote': json.loads(record['stdout'].strip().splitlines()[-1])}
        put(self.state / 'delivered.json', delivered)
        self.event({'event': 'milestone_delivered', 'name': name})
        return True

    def tick(self):
        record = self.remote(self.query, [self.query['python'], '-I', self.query['code_dir'].rstrip('/') + '/advance.py', 'tick', '--campaign', self.query['campaign']], False, 1800)
        if record['transport_error']:
            self.transport_failed(record, 'query_tick')
            return None
        try:
            summary = json.loads(record['stdout'])
        except ValueError:
            raise Stop(6, 'query_tick_unparseable', {'returncode': record['returncode'], 'stdout': record['stdout'][-2000:], 'stderr': record['stderr']})
        put(self.state / 'last-query-tick.json', summary)
        return summary

    def pretrain_gates(self, inventory):
        """Expected milestones whose trajectory has stopped without completing and has no queued or running job."""
        jobs = inventory.get('jobs')
        prefix = self.pretrain['job_name_prefix']
        stalled = {}
        for name, row in inventory['trajectories'].items():
            live = jobs is None or any(job[2].startswith(prefix) and job[2].endswith(name) for job in jobs)
            stopped = row['training_attention'] or row['stop_markers'] or (row['submitted'] and not live and not row['complete'])
            if stopped and not live and not row['complete']:
                missing = [A.delivery_name(expected) for expected in self.protocol['expected_initializers'] if expected['trajectory'] == name
                           and self.protocol['pretraining_lineage']['milestone_model_name'] % expected['initializer_update_budget'] not in row['models']]
                if missing:
                    stalled[name] = {'step': row['step'], 'markers': row['stop_markers'], 'training_attention': row['training_attention'], 'missing_milestones': missing}
        if stalled:
            put(self.state / 'PRETRAIN_ATTENTION.json', {'time': now(), 'trajectories': stalled,
                                                         'meaning': 'these milestones may never exist; the query grid cannot complete and no selection runs until a person resolves pretraining'})
        elif (self.state / 'PRETRAIN_ATTENTION.json').exists():
            os.replace(self.state / 'PRETRAIN_ATTENTION.json', self.state / ('PRETRAIN_ATTENTION.resolved-%d.json' % int(now())))
        return stalled

    def progress_gate(self, inventory, summary):
        fingerprint = {'pretrain_steps': {name: row['step'] for name, row in inventory['trajectories'].items()} if inventory else None,
                       'delivered': sorted(self.delivered()), 'query_states': (summary or {}).get('trial_states'), 'query_gate': (summary or {}).get('gate'),
                       'query_submitted': bool((summary or {}).get('submitted'))}
        previous = self.read('progress-fingerprint.json', None)
        live_pretrain = inventory is not None and bool([job for job in (inventory.get('jobs') or []) if job[2].startswith(self.pretrain['job_name_prefix'])])
        live_query = any(state in ('active', 'unknown', 'suspect') for stage in ((summary or {}).get('trial_states') or {}).values() for state in stage)
        pending_selection = str(((summary or {}).get('gate') or {}).get('gate', '')).endswith('selection_pending')
        if previous is None or previous['fingerprint'] != fingerprint or live_pretrain or live_query or pending_selection:
            put(self.state / 'progress-fingerprint.json', {'fingerprint': fingerprint, 'since': now()})
            return
        if now() - previous['since'] > self.config['no_progress_hours'] * 3600:
            raise Stop(3, 'no_progress', {'since': previous['since'], 'fingerprint': fingerprint})

    def results(self):
        campaign = self.query['campaign'].rstrip('/')
        target = self.state / 'results'
        target.mkdir(parents=True, exist_ok=True)
        for source in ('decisions/', 'state/DONE.json', 'state/initializers/', 'state/events.jsonl', 'frozen-manifest.json', 'ROOT_RELEASE.json'):
            record = self.call(['rsync', '-s', '-r', '--times', '-e', self.query['rsync_shell'], self.query['host'] + ':' + campaign + '/' + source,
                                str(target / source.rstrip('/').replace('/', '__'))], True, 600)
            if record['returncode'] != 0:
                raise Stop(6, 'results_download_failed', {'source': source, 'stderr': record['stderr']})
        final = json.loads((target / 'decisions' / 'final.json').read_text())
        screen = json.loads((target / 'decisions' / 'screen.json').read_text())
        put(self.state / 'REPORT-INPUTS.json', {'screen_decision': str(target / 'decisions' / 'screen.json'), 'final_decision': str(target / 'decisions' / 'final.json'),
                                                'promoted': screen['promoted'], 'reuse_control_long': screen['reuse_control_long'], 'winners': final['winners'],
                                                'screen_leaderboards': screen['leaderboards'], 'final_leaderboards': final['leaderboards'],
                                                'scope': 'VALIDATION only, fold 0 seed 0, 545 queries / 586 associations, 13625 candidates; no TEST',
                                                'hardware_note': self.protocol['pretraining_lineage']['hardware_note']})

    def cycle(self):
        if (self.state / 'OPERATOR_STOP').exists():
            raise Stop(0, 'operator_stop')
        inventory = self.pretrain_inventory()
        staged = self.query_staged()
        if inventory is not None:
            self.transport_ok()
            put(self.state / 'last-pretrain-inventory.json', inventory)
            self.pretrain_gates(inventory)
            for expected in self.protocol['expected_initializers']:
                name = A.delivery_name(expected)
                if name in self.delivered():
                    continue
                folder = self.fetch(expected, inventory)
                if folder is not None and staged:
                    self.deliver(name, folder)
        summary = self.tick() if staged else None
        status = {'time': now(), 'identity': self.identity, 'delivered': sorted(self.delivered()), 'query_staged': staged, 'query_tick': summary}
        put(self.state / 'status.json', status)
        if summary is not None:
            if summary.get('code_revision') and summary['code_revision'] != self.identity.get('git_head'):
                raise Stop(8, 'bridge_revision_differs_from_query_campaign', {'bridge': self.identity.get('git_head'), 'query_campaign': summary['code_revision']})
            if summary.get('status') == 'done':
                self.results()
                raise Stop(0, 'done')
            if summary.get('status') in ('halted', 'operator_stop'):
                raise Stop(4, 'query_' + summary['status'], summary)
        self.progress_gate(inventory, summary)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['once', 'loop', 'inventory'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    bridge = Bridge(args.config)
    if args.command == 'inventory':
        print(json.dumps({'pretrain': bridge.pretrain_inventory(), 'delivered': bridge.delivered(), 'identity': bridge.identity}, indent=2, sort_keys=True, default=str))
        return 0
    lock = (bridge.state / 'bridge.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(json.dumps({'status': 'busy', 'reason': 'another bridge holds the lock'}))
        return 1
    bridge.event({'event': 'start', 'command': args.command, 'identity': bridge.identity, 'pid': os.getpid()})
    try:
        while True:
            try:
                bridge.cycle()
            except Stop as stop:
                put(bridge.state / 'STOPPED.json', {'code': stop.code, 'reason': stop.reason, 'detail': stop.detail, 'time': now(), 'identity': bridge.identity})
                bridge.event({'event': 'stop', 'code': stop.code, 'reason': stop.reason})
                print(json.dumps({'stopped': stop.reason, 'code': stop.code}, default=str))
                return stop.code
            if args.command == 'once':
                return 0
            time.sleep(bridge.config['poll_seconds'])
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
