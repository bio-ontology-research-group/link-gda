"""Workstation bridge and watcher for the production campaigns (Orix pretraining, IBEX query training).

fetch-hpo   Copy the HPO campaign's decision files to <prod_dir>/hpo with the hashes computed on IBEX (FETCH-RECEIPT.json);
            refuses until the HPO campaign is DONE. freeze.py check-hpo validates them.
once        One short scheduling cycle: Orix tick, IBEX tick, both inventories, then transfer reconciliation. Transfers
            never run inside the cycle: each delivery (model-only export, COMPLETE, identity, config, progress; never
            optimizer states; hashed against Orix COMPLETE, re-hashed by `campaign.py admit`, atomic rename) and each
            archive (hashed locally, re-hashed on Orix by `verify-archive`, acknowledged on IBEX) runs in its own
            detached worker process (`transfer`), one per unit, holding that unit's flock for the whole transfer and
            writing only that unit's state file. Pools are bounded (max_delivery_workers, max_archive_workers); deliveries
            have their own pool and go first, ordered by how many query units they unblock. A failed attempt backs off
            exponentially; after transfer_attempt_limit failures the unit raises attention and is not relaunched. A dead
            worker releases its lock, and the next cycle relaunches it, resuming from the transit and .partial copies.
loop        `once` every poll_seconds while holding the scheduler lock (only one scheduler; transfers never hold it),
            until everything is complete and archived or a STOP file appears. Workers also stop at the next file on STOP.
drain       `once` repeatedly until no transfer is running or launchable (operator convenience, rehearsals).
transfer    The worker entry point (launched by the scheduler; safe to run by hand: a duplicate exits on the lock).
status      Read-only inventories of both campaigns plus the bridge state.
dry-run     Read-only: inventories, `tick --dry-run` on both hosts, and the deliveries and archives a cycle would make.
The bridge never cancels, resumes, releases or acknowledges attention; it calls only tick, inventory, admit,
verify-archive and ack-archive. Transit copies are removed only after the downstream host verified them.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import exclusive_lock, read_json, sha, write_exclusive, write_replace, append_line  # noqa: E402

HPO_REMOTE = {'DONE.json': 'state/DONE.json', 'frozen-manifest.json': 'frozen-manifest.json'}
TRANSFER_KINDS = {'delivery': {'receipts': 'deliveries'},
                  'archive': {'receipts': 'archives', 'ack': 'ack-archive'},
                  'hpo-archive': {'receipts': 'hpo-archives', 'ack': 'ack-hpo-archive', 'root_suffix': '/hpo-campaign'}}
HPO_DECISIONS = ('recipe.json', 'recipe.frozen.json', 'recipe.crosscheck.json', 'recipe.crosscheck-agreement.json', 'final.json', 'final.frozen.json',
                 'selection-decision.json', 'selection-decision.receipt.json', 'screen.json')


class Bridge:
    def __init__(self, path, dry_run=False, failures_file=None):
        self.config_path = str(Path(path).resolve())
        self.config = read_json(path)
        self.state = Path(self.config['state_dir'])
        self.transit = Path(self.config['transit_dir'])
        self.dry_run = dry_run
        self.failures_file = Path(failures_file) if failures_file else self.state / 'transport-failures.json'
        self.state.mkdir(parents=True, exist_ok=True)
        self.transit.mkdir(parents=True, exist_ok=True)

    def log(self, value):
        append_line(self.state / 'events.jsonl', value)

    def call(self, command, timeout, stdin=None, read_only=True):
        attempts = self.config.get('transport_attempts', 3) if read_only else 1
        for attempt in range(attempts):
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, input=stdin)
                record = {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr[-4000:], 'transport_error': result.returncode == 255}
            except subprocess.TimeoutExpired:
                record = {'command': command, 'returncode': None, 'stdout': '', 'stderr': 'timeout', 'transport_error': True}
            if not record['transport_error']:
                return record
            time.sleep(self.config.get('transport_backoff_seconds', 30) * (2 ** attempt))
        failures = read_json(self.failures_file, {'consecutive': 0})
        failures = {'consecutive': failures['consecutive'] + 1, 'last': command[:3], 'time': time.time()}
        write_replace(self.failures_file, failures)
        return record

    def remote(self, side, parts, timeout=600, stdin=None, read_only=True):
        host = self.config[side]
        return self.call(list(host['ssh']) + [' '.join(shlex.quote(str(part)) for part in parts)], timeout, stdin, read_only)

    def campaign(self, side, command, *extra, stdin=None, read_only=True):
        host = self.config[side]
        return self.remote(side, [host['python'], str(Path(host['code_dir']) / 'campaign.py'), command, '--campaign', host['campaign']] + list(extra),
                           timeout=self.config.get('transfer_timeout_seconds', 3600), stdin=stdin, read_only=read_only)

    def rsync(self, side, source, destination, upload, directory=False):
        host = self.config[side]
        flags = ['rsync', '-s', '--partial', '--times'] + (['-r'] if directory else []) + (['--compress'] if self.config.get('rsync_compress') else []) + ['-e', host['rsync_shell']]
        remote = host['host'] + ':'
        command = flags + ([source, remote + destination] if upload else [remote + source, destination])
        return self.call(command, self.config.get('transfer_timeout_seconds', 3600), read_only=not upload)

    def inventory(self, side):
        record = self.campaign(side, 'inventory')
        if record['returncode'] != 0:
            return None, record
        return json.loads(record['stdout']), record

    # ------------------------------------------------------------ HPO decision files

    def fetch_hpo(self):
        hpo = self.config['hpo']['campaign']
        target = Path(self.config['prod_dir']) / 'hpo'
        if (target / 'FETCH-RECEIPT.json').exists():
            return {'status': 'already_fetched', 'receipt': str(target / 'FETCH-RECEIPT.json')}
        names = dict({name: 'decisions/' + name for name in HPO_DECISIONS}, **HPO_REMOTE)
        listing = self.remote('ibex', ['ls', hpo + '/decisions'])
        hashes = self.remote('ibex', ['sha256sum'] + [hpo + '/' + path for path in names.values()])
        if hashes['returncode'] != 0:
            return {'status': 'waiting', 'detail': 'HPO decision files not all present (campaign not DONE)', 'stderr': hashes['stderr'][-2000:]}
        remote = {}
        for line in hashes['stdout'].splitlines():
            digest, path = line.split(None, 1)
            remote[next(name for name, relative in names.items() if path.strip() == hpo + '/' + relative)] = digest
        partial = target.with_name('.hpo-partial')
        partial.mkdir(parents=True, exist_ok=True)
        for name, relative in names.items():
            record = self.rsync('ibex', hpo + '/' + relative, str(partial / name), upload=False)
            if record['returncode'] != 0 or sha(partial / name) != remote[name]:
                return {'status': 'transfer_failed', 'file': name, 'record': record}
        write_exclusive(partial / 'FETCH-RECEIPT.json', {'campaign': hpo, 'files': remote, 'remote_decisions_listing': listing['stdout'].split(), 'time': time.time()})
        os.replace(str(partial), str(target))
        return {'status': 'fetched', 'receipt': str(target / 'FETCH-RECEIPT.json')}

    def fetch_reuse_evidence(self):
        """After a passed check-hpo: fetch the original delivery ADMISSION/identity/config of the calibrated winner's
        initializer and the winner trajectory's identity/COMPLETE, each with its IBEX-computed hash."""
        hpo_dir = Path(self.config['prod_dir']) / 'hpo'
        target = hpo_dir / 'reuse'
        if (target / 'FETCH-RECEIPT.json').exists():
            return {'status': 'already_fetched'}
        recipe, manifest, final = read_json(hpo_dir / 'recipe.json'), read_json(hpo_dir / 'frozen-manifest.json'), read_json(hpo_dir / 'final.json')
        exported = recipe['common_winner']['identities']['pretrained_sha256']
        record = next(value for value in manifest['initializers'].values() if value['sha256'] == exported)
        delivery = Path(record['delivery_admission']['path']).parent
        winner = final['winners'][recipe['common_leaderboard']]
        remote = {'ADMISSION.json': str(delivery / 'ADMISSION.json'), 'identity.json': str(delivery / 'identity.json'), 'config.json': str(delivery / 'config.json'),
                  'trajectory-identity.json': winner['training_dir'] + '/identity.json', 'trajectory-COMPLETE.json': winner['training_dir'] + '/COMPLETE.json'}
        hashes = self.remote('ibex', ['sha256sum'] + list(remote.values()))
        if hashes['returncode'] != 0:
            return {'status': 'failed', 'stderr': hashes['stderr'][-1000:]}
        by_path = {line.split(None, 1)[1].strip(): line.split(None, 1)[0] for line in hashes['stdout'].splitlines()}
        partial = target.with_name('.reuse-partial')
        partial.mkdir(parents=True, exist_ok=True)
        for name, path in remote.items():
            record_ = self.rsync('ibex', path, str(partial / name), upload=False)
            if record_['returncode'] != 0 or sha(partial / name) != by_path[path]:
                return {'status': 'transfer_failed', 'file': name}
        write_exclusive(partial / 'FETCH-RECEIPT.json', {'remote': remote, 'files': {name: by_path[path] for name, path in remote.items()},
                                                         'initializer_record': record, 'winner_training_dir': winner['training_dir'], 'time': time.time()})
        os.replace(str(partial), str(target))
        return {'status': 'fetched'}

    # ------------------------------------------------------------ Orix -> IBEX deliveries

    def stop_requested(self):
        return (self.state / 'STOP').exists()

    def deliver(self, unit, export):
        """Fetch one completed export to transit, verify, upload to IBEX incoming/.partial-<unit>, admit there.
        Runs inside a transfer worker that holds this unit's lock; attempts and backoff are the worker's business."""
        receipt = self.state / 'deliveries' / (unit + '.json')
        if receipt.exists():
            return {'unit': unit, 'status': 'already_delivered'}
        folder = self.transit / ('delivery-' + unit)
        folder.mkdir(parents=True, exist_ok=True)
        for name, relative in export['paths'].items():
            if self.stop_requested():
                return {'unit': unit, 'status': 'stopped'}
            if (folder / name).exists() and sha(folder / name) == export['files'][name]:
                continue
            record = self.rsync('orix', str(Path(export['training_dir']) / relative), str(folder / name), upload=False)
            if record['returncode'] != 0:
                return {'unit': unit, 'status': 'fetch_failed', 'file': name, 'stderr': record['stderr'][-500:]}
        wrong = [name for name, digest in export['files'].items() if sha(folder / name) != digest]
        if wrong:
            return {'unit': unit, 'status': 'fetched_hash_mismatch', 'files': wrong}
        write_replace(folder / 'DELIVERY.json', {'unit': unit, 'files': export['files'], 'source': {'site': 'orix', 'campaign': self.config['orix']['campaign'],
                                                                                                    'training_dir': export['training_dir']}, 'time': time.time()})
        destination = self.config['ibex']['campaign'] + '/incoming/.partial-' + unit + '/'
        mkdir = self.remote('ibex', ['mkdir', '-p', destination], read_only=False)
        upload = self.rsync('ibex', str(folder) + '/', destination, upload=True, directory=True)
        if mkdir['returncode'] != 0 or upload['returncode'] != 0:
            return {'unit': unit, 'status': 'upload_failed', 'stderr': upload['stderr'][-500:]}
        admitted = self.campaign('ibex', 'admit', '--parent', unit, read_only=False)
        if admitted['returncode'] != 0:
            return {'unit': unit, 'status': 'admission_refused', 'stderr': admitted['stderr'][-2000:]}
        write_exclusive(receipt, {'unit': unit, 'files': export['files'], 'admit': admitted['stdout'][-2000:], 'time': time.time()})
        shutil.rmtree(str(folder))
        return {'unit': unit, 'status': 'delivered'}

    # ------------------------------------------------------------ IBEX -> Orix archive

    def archive(self, unit, item, root=None, ack_command='ack-archive', kind='archives'):
        """Archive one item {sources: {prefix: remote dir}, files: {'prefix/relative': sha256}} from IBEX to Orix."""
        receipt = self.state / kind / (unit + '.json')
        if receipt.exists():
            ack = self.campaign('ibex', ack_command, '--unit', unit, stdin=json.dumps(read_json(receipt)['verified']), read_only=False)
            return {'unit': unit, 'status': 'reacknowledged' if ack['returncode'] == 0 else 'ack_failed', 'stderr': ack['stderr'][-500:]}
        folder = self.transit / (kind + '-' + unit)
        folder.mkdir(parents=True, exist_ok=True)
        for name, digest in item['files'].items():
            if self.stop_requested():
                return {'unit': unit, 'status': 'stopped'}
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            if (folder / name).exists() and sha(folder / name) == digest:
                continue
            prefix, relative = name.split('/', 1)
            record = self.rsync('ibex', item['sources'][prefix] + '/' + relative, str(folder / name), upload=False)
            if record['returncode'] != 0:
                return {'unit': unit, 'status': 'fetch_failed', 'file': name}
        wrong = [name for name, digest in item['files'].items() if sha(folder / name) != digest]
        if wrong:
            return {'unit': unit, 'status': 'fetched_hash_mismatch', 'files': wrong[:10]}
        root = root or self.config['orix']['archive_root']
        mkdir = self.remote('orix', ['mkdir', '-p', root + '/.partial-' + unit], read_only=False)
        upload = self.rsync('orix', str(folder) + '/', root + '/.partial-' + unit + '/', upload=True, directory=True)
        if mkdir['returncode'] != 0 or upload['returncode'] != 0:
            return {'unit': unit, 'status': 'upload_failed', 'stderr': upload['stderr'][-500:]}
        expected = {'files': item['files'], 'source': {'site': 'ibex', 'campaign': self.config['ibex']['campaign'], 'sources': item['sources']}}
        verify = self.remote('orix', [self.config['orix']['python'], str(Path(self.config['orix']['code_dir']) / 'campaign.py'), 'verify-archive', '--root', root, '--unit', unit],
                             timeout=self.config.get('transfer_timeout_seconds', 3600), stdin=json.dumps(expected), read_only=False)
        if verify['returncode'] != 0:
            return {'unit': unit, 'status': 'archive_verification_failed', 'stderr': verify['stderr'][-1000:]}
        verified = dict(json.loads(verify['stdout']), unit=unit, archive=root + '/' + unit, sources=item['sources'])
        write_exclusive(receipt, {'unit': unit, 'verified': verified, 'time': time.time()})
        ack = self.campaign('ibex', ack_command, '--unit', unit, stdin=json.dumps(verified), read_only=False)
        if ack['returncode'] != 0:
            return {'unit': unit, 'status': 'ack_failed', 'stderr': ack['stderr'][-1000:]}
        shutil.rmtree(str(folder))
        return {'unit': unit, 'status': 'archived'}

    # ------------------------------------------------------------ transfer workers (separate processes, one per unit)

    def transfer_paths(self, kind, unit):
        base = self.state / 'transfers' / ('%s-%s' % (kind, unit))
        return {'lock': base.with_suffix('.lock'), 'item': base.with_suffix('.item.json'), 'state': base.with_suffix('.state.json'),
                'log': base.with_suffix('.log'), 'transport': base.with_suffix('.transport.json')}

    @staticmethod
    def lock_held(path):
        """True while a worker holds the unit's flock (released by the kernel if the worker dies)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        finally:
            os.close(descriptor)
        return False

    def receipt_path(self, kind, unit):
        return self.state / TRANSFER_KINDS[kind]['receipts'] / (unit + '.json')

    def transfer_status(self, kind, unit):
        paths = self.transfer_paths(kind, unit)
        if self.receipt_path(kind, unit).exists() and kind == 'delivery':
            return 'done'
        if self.lock_held(paths['lock']):
            return 'running'
        state = read_json(paths['state'], {})
        if state.get('status') == 'done':
            return 'done'
        if state.get('status') == 'failed':
            if state.get('attempts', 0) >= self.config.get('transfer_attempt_limit', 3):
                return 'exhausted'
            if time.time() < state.get('next_after', 0):
                return 'backoff'
        return 'pending'

    def launch(self, kind, unit, item):
        """Start one detached worker process for one unit. A second worker for the same unit exits on the lock."""
        paths = self.transfer_paths(kind, unit)
        write_replace(paths['item'], {'kind': kind, 'unit': unit, 'item': item})
        log = open(str(paths['log']), 'a')
        try:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'transfer', '--config', self.config_path, '--kind', kind, '--unit', unit],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        finally:
            log.close()
        self.log({'event': 'transfer_launched', 'kind': kind, 'unit': unit, 'pid': process.pid})
        return process.pid

    def run_transfer(self, kind, unit):
        """Worker body: hold the unit lock for the whole transfer, count the attempt, write only this unit's state."""
        paths = self.transfer_paths(kind, unit)
        with exclusive_lock(paths['lock']) as locked:
            if not locked:
                return {'unit': unit, 'status': 'busy_other_worker'}
            state = read_json(paths['state'], {})
            if state.get('status') == 'done':
                return {'unit': unit, 'status': 'already_done'}
            spec = read_json(paths['item'])
            attempt = state.get('attempts', 0) + 1
            write_replace(paths['state'], dict(state, status='running', attempts=attempt, pid=os.getpid(), started=time.time()))
            worker = Bridge(self.config_path, failures_file=paths['transport'])
            try:
                if kind == 'delivery':
                    result = worker.deliver(unit, spec['item'])
                else:
                    rules = TRANSFER_KINDS[kind]
                    root = self.config['orix']['archive_root'].rstrip('/') + rules.get('root_suffix', '')
                    result = worker.archive(unit, spec['item'], root=root, ack_command=rules['ack'], kind=rules['receipts'])
            except Exception as error:
                result = {'unit': unit, 'status': 'worker_exception', 'error': repr(error)}
            done = result['status'] in ('delivered', 'already_delivered', 'archived', 'reacknowledged')
            stopped = result['status'] == 'stopped'
            status = 'done' if done else 'stopped' if stopped else 'failed'
            backoff = self.config.get('transfer_backoff_seconds', 300) * (2 ** (attempt - 1))
            write_replace(paths['state'], {'status': status, 'attempts': attempt - 1 if stopped else attempt, 'result': result, 'finished': time.time(),
                                           'next_after': 0 if done or stopped else time.time() + backoff, 'pid': os.getpid()})
            self.log({'event': 'transfer_finished', 'kind': kind, 'unit': unit, 'status': status, 'detail': result.get('status')})
            return result

    def delivery_order(self, units):
        """Initializers that unblock the most query units first (PFS parents feed five conditions)."""
        bundle = Path(self.config['prod_dir']) / 'bundle' / 'BUNDLE.json'
        children = {}
        if bundle.exists():
            for child in read_json(bundle)['units']['query'].values():
                children[child['parent']] = children.get(child['parent'], 0) + 1
        return sorted(units, key=lambda unit: (-children.get(unit, 0), unit))

    def schedule_transfers(self, orix, ibex):
        """Reconcile worker states and launch new workers within bounded pools; never waits for a transfer."""
        report = {'running': [], 'launched': [], 'exhausted': [], 'backoff': [], 'planned': []}
        if self.stop_requested():
            report['stop'] = True
            return report
        pools = {'delivery': self.config.get('max_delivery_workers', 2), 'archive': self.config.get('max_archive_workers', 2)}
        candidates = {'delivery': [], 'archive': []}
        missing = [unit for unit in orix['exports'] if unit not in ibex['admitted_parents']]
        for unit in self.delivery_order(missing):
            candidates['delivery'].append(('delivery', unit, orix['exports'][unit]))
        for unit in sorted(ibex['archivable']):
            candidates['archive'].append(('archive', unit, ibex['archivable'][unit]))
        hpo_items = {}
        if ibex.get('storage_decision', {}).get('archive_current_hpo_reportables'):
            listing = self.campaign('ibex', 'hpo-archive-list', '--hpo-campaign', self.config['hpo']['campaign'])
            if listing['returncode'] == 0:
                hpo_items = json.loads(listing['stdout'])['items']
        for key in sorted(hpo_items):
            candidates['archive'].append(('hpo-archive', key, hpo_items[key]))
        for pool, entries in candidates.items():
            statuses = [(kind, unit, item, self.transfer_status(kind, unit)) for kind, unit, item in entries]
            running = [unit for kind, unit, item, status in statuses if status == 'running']
            report['running'] += ['%s:%s' % (kind, unit) for kind, unit, item, status in statuses if status == 'running']
            report['exhausted'] += ['%s:%s' % (kind, unit) for kind, unit, item, status in statuses if status == 'exhausted']
            report['backoff'] += ['%s:%s' % (kind, unit) for kind, unit, item, status in statuses if status == 'backoff']
            free = pools[pool] - len(running)
            if pool == 'delivery':
                free = min(free, ibex['delivery_slots'] - len(running))
            for kind, unit, item, status in statuses:
                if status != 'pending' or free <= 0:
                    continue
                report['planned'].append('%s:%s' % (kind, unit))
                if not self.dry_run:
                    self.launch(kind, unit, item)
                    report['launched'].append('%s:%s' % (kind, unit))
                free -= 1
        return report

    def transfers_overview(self):
        folder = self.state / 'transfers'
        value = {}
        for path in sorted(folder.glob('*.state.json')) if folder.is_dir() else []:
            record = read_json(path)
            value[path.name[:-len('.state.json')]] = {key: record.get(key) for key in ('status', 'attempts', 'finished', 'next_after')}
        return value

    # ------------------------------------------------------------ cycle

    def cycle(self):
        """One short scheduling pass: both ticks, both inventories, transfer reconciliation. Never waits for a transfer."""
        summary = {'time': time.time(), 'dry_run': self.dry_run, 'attention': []}
        tick = self.campaign('orix', 'tick', *(['--dry-run'] if self.dry_run else []), read_only=self.dry_run)
        summary['orix_tick'] = {'returncode': tick['returncode'], 'stdout': tick['stdout'][-6000:], 'stderr': tick['stderr'][-1000:]}
        tick = self.campaign('ibex', 'tick', *(['--dry-run'] if self.dry_run else []), read_only=self.dry_run)
        summary['ibex_tick'] = {'returncode': tick['returncode'], 'stdout': tick['stdout'][-6000:], 'stderr': tick['stderr'][-1000:]}
        orix, record = self.inventory('orix')
        ibex, record_ibex = self.inventory('ibex')
        if orix is None or ibex is None:
            summary['attention'].append('inventory_unavailable')
            summary['inventory_errors'] = [record['stderr'][-500:], record_ibex['stderr'][-500:]]
            if not self.dry_run:
                write_replace(self.state / 'last-cycle.json', summary)
                self.alert(summary['attention'])
            return summary
        summary['transfers'] = transfers = self.schedule_transfers(orix, ibex)
        for name in transfers['exhausted']:
            summary['attention'].append('transfer_exhausted:%s' % name)
        for side, value, tick_record in (('orix', orix, summary['orix_tick']), ('ibex', ibex, summary['ibex_tick'])):
            if value.get('halt'):
                summary['attention'].append('%s_halt:%s' % (side, value['halt'].get('reason')))
            if value.get('attention'):
                summary['attention'].append('%s_unit_attention:%d' % (side, len(value['attention'])))
            if tick_record['returncode'] not in (0, None):
                summary['attention'].append('%s_tick_rc_%s' % (side, tick_record['returncode']))
            last = value.get('last_tick') or {}
            if (last.get('skipped') or {}).get('storage_stall'):
                summary['attention'].append('%s_storage_stall' % side)
            if (last.get('skipped') or {}).get('storage_permission_required'):
                summary['attention'].append('%s_storage_permission_required' % side)
        if orix.get('utilization_warnings'):
            summary['utilization_warnings'] = orix['utilization_warnings']
            summary['attention'].append('orix_utilization_warnings:%d' % len(orix['utilization_warnings']))
        failures = read_json(self.state / 'transport-failures.json', {'consecutive': 0})
        if failures['consecutive'] >= self.config.get('transport_failure_limit', 12):
            summary['attention'].append('transport_failures')
        summary['counts'] = {'orix': orix['counts'], 'ibex': ibex['counts'], 'ibex_inference': ibex.get('test_counts'), 'archived': len(ibex['archived']),
                             'reused': {'orix': orix.get('reused', []), 'ibex': ibex.get('reused', [])}}
        orix_done = sum(orix['counts'].get(state, 0) for state in ('complete', 'reused')) == sum(orix['counts'].values())
        summary['done'] = (orix_done and ibex['counts'].get('complete') == sum(ibex['counts'].values())
                           and (ibex.get('test_counts') or {}).get('complete') == sum(ibex['counts'].values()) and len(ibex['archived']) == sum(ibex['counts'].values()))
        if not self.dry_run:
            write_replace(self.state / 'last-cycle.json', summary)
            append_line(self.state / 'cycles.jsonl', {'attention': summary['attention'], 'done': summary['done'],
                                                      'transfers': {key: transfers[key] for key in ('running', 'launched', 'exhausted')}})
            if summary['attention']:
                self.alert(summary['attention'])
        return summary

    def alert(self, reasons):
        notified = set(read_json(self.state / 'notified.json', {'reasons': []})['reasons'])
        write_replace(self.state / 'ATTENTION.json', {'time': time.time(), 'reasons': reasons, 'resolution': 'a person inspects; the bridge takes no corrective action'})
        new = [reason for reason in reasons if reason not in notified]
        if new and self.config.get('notify_command'):
            subprocess.run(list(self.config['notify_command']) + [', '.join(new)], timeout=60)
        write_replace(self.state / 'notified.json', {'reasons': sorted(notified | set(new))})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['fetch-hpo', 'fetch-reuse-evidence', 'once', 'loop', 'drain', 'status', 'dry-run', 'transfer'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--kind', choices=sorted(TRANSFER_KINDS), help='transfer worker only')
    parser.add_argument('--unit', help='transfer worker only')
    parser.add_argument('--timeout', type=float, default=7200, help='drain: give up after this many seconds')
    args = parser.parse_args()
    bridge = Bridge(args.config, dry_run=args.command == 'dry-run')
    if args.command == 'transfer':
        result = bridge.run_transfer(args.kind, args.unit)
        print(json.dumps(result, default=str), flush=True)
        return 0
    if args.command in ('fetch-hpo', 'fetch-reuse-evidence'):
        result = bridge.fetch_hpo() if args.command == 'fetch-hpo' else bridge.fetch_reuse_evidence()
        print(json.dumps(result, indent=2, default=str))
        return 0 if result['status'] in ('fetched', 'already_fetched') else 1
    if args.command == 'status':
        value = {side: bridge.inventory(side)[0] for side in ('orix', 'ibex')}
        value['bridge'] = {'last_cycle': read_json(bridge.state / 'last-cycle.json', {}) or None, 'attention': read_json(bridge.state / 'ATTENTION.json', {}) or None,
                           'transfers': bridge.transfers_overview()}
        print(json.dumps(value, indent=2, default=str))
        return 0
    with exclusive_lock(bridge.state / 'bridge.lock') as locked:
        if not locked:
            print(json.dumps({'status': 'another scheduler holds the bridge lock'}))
            return 3
        started = time.time()
        while True:
            if (bridge.state / 'STOP').exists():
                return 0
            began = time.time()
            summary = bridge.cycle()
            transfers = summary.get('transfers') or {}
            print(json.dumps({key: summary.get(key) for key in ('attention', 'done', 'counts')}, default=str), json.dumps({key: transfers.get(key) for key in ('running', 'launched')}), flush=True)
            if args.command in ('once', 'dry-run'):
                return 0 if not summary['attention'] else 1
            if args.command == 'drain':
                if not transfers.get('running') and not transfers.get('launched'):
                    return 0 if not summary['attention'] else 1
                if time.time() - started > args.timeout:
                    return 4
                time.sleep(max(1, bridge.config.get('drain_poll_seconds', 2)))
                continue
            if summary.get('done'):
                write_replace(bridge.state / 'DONE.json', summary)
                return 0
            time.sleep(max(30, bridge.config.get('poll_seconds', 300) - (time.time() - began)))


if __name__ == '__main__':
    raise SystemExit(main())
