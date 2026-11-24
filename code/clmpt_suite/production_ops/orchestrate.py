"""Workstation entrypoint from the real HPO freeze to a running production bridge. Each step is idempotent and writes a
receipt in <prod_dir>/orchestrate/; re-running continues where it stopped.

  1 wait-hpo       bridge fetch-hpo until the HPO campaign is DONE (polls; --once checks once)
  2 check-hpo      freeze.py check-hpo against the user's selection decision (calibrated); a failure stops with ATTENTION
  3 protocols      freeze.py protocols (make_protocol generate FROZEN + validate)
  4 reuse          bridge fetch-reuse-evidence, freeze.py reuse-audit (a failed audit is recorded: no reuse, not an error)
  5 bundle         freeze.py bundle at the local HEAD (refuses uncommitted or unpushed executed source)
  6 source gate    both host checkouts at the bundle revision and every staged input present; otherwise GATE.json with
                   the exact commands and exit code 3. Source synchronization is NOT performed here.
  7 stage          copy bundle and DEPLOY files to each host, campaign.py stage (a staged campaign must hold this bundle)
  8 release        only if the reviewed config says release.approve true; the storage decision file, if given, is
                   recorded; otherwise GATE.json 'release_approval_required'
  9 bridge         exec bridge.py loop (or print the command with --no-bridge)
--dry-run reports which steps are done and which gate would stop, without remote mutations.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import read_json, sha, write_replace  # noqa: E402
import bridge as B  # noqa: E402

EXIT_GATE = 3


class Gate(Exception):
    def __init__(self, name, detail):
        super().__init__(name)
        self.name, self.detail = name, detail


def local(command, timeout=7200):
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True, timeout=timeout)
    return {'command': [str(part) for part in command], 'returncode': result.returncode, 'stdout': result.stdout[-8000:], 'stderr': result.stderr[-8000:]}


class Orchestrator:
    def __init__(self, path, dry_run):
        self.config = read_json(path)
        self.dry_run = dry_run
        self.bridge = B.Bridge(self.config['bridge_config'], dry_run=dry_run)
        self.prod = Path(self.config['prod_dir'])
        self.folder = self.prod / 'orchestrate'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.python = self.config['python']
        self.source = Path(self.config['source_root'])

    def done(self, step):
        return (self.folder / (step + '.json')).exists()

    def record(self, step, value):
        if not self.dry_run:
            write_replace(self.folder / (step + '.json'), dict(value, step=step, time=time.time()))

    def freeze(self, *arguments):
        return local([self.python, HERE / 'freeze.py'] + list(arguments))

    # ------------------------------------------------------------ steps

    def wait_hpo(self, once):
        while True:
            if (self.prod / 'hpo' / 'FETCH-RECEIPT.json').exists():
                return {'status': 'fetched'}
            if self.dry_run:
                raise Gate('hpo_not_fetched', 'dry run: the HPO decision files have not been fetched')
            result = self.bridge.fetch_hpo()
            if result['status'] in ('fetched', 'already_fetched'):
                return result
            if once or result['status'] not in ('waiting',):
                raise Gate('hpo_not_done', result)
            time.sleep(self.config.get('poll_seconds', 900))

    def check_hpo(self):
        result = self.freeze('check-hpo', '--hpo-dir', self.prod / 'hpo', '--user-decision', self.config['user_decision'], '--output', self.prod / 'hpo-check.json')
        if result['returncode'] != 0 or read_json(self.prod / 'hpo-check.json', {}).get('passed') is not True:
            raise Gate('hpo_check_failed', result)
        return result

    def protocols(self):
        result = self.freeze('protocols', '--prod', self.prod, '--hpo-dir', self.prod / 'hpo', '--hpo-check', self.prod / 'hpo-check.json', '--source-root', self.source, '--python', self.python)
        if result['returncode'] != 0:
            raise Gate('protocols_refused', result)
        return result

    def reuse(self):
        fetched = self.bridge.fetch_reuse_evidence()
        if fetched['status'] not in ('fetched', 'already_fetched'):
            return {'status': 'no_reuse', 'reason': 'evidence not fetched', 'fetch': fetched}
        result = self.freeze('reuse-audit', '--prod', self.prod, '--source-root', self.source, '--python', self.python)
        return {'status': 'audited' if result['returncode'] == 0 else 'no_reuse', 'audit': result}

    def bundle(self):
        head = local(['git', '-C', self.source, 'rev-parse', 'HEAD'])['stdout'].strip()
        if self.config.get('revision') and self.config['revision'] != head:
            raise Gate('local_revision_differs', {'expected': self.config['revision'], 'head': head})
        result = self.freeze('bundle', '--prod', self.prod, '--source-root', self.source, '--revision', head)
        if result['returncode'] != 0:
            raise Gate('bundle_refused_commit_and_push_production_source', result)
        return dict(result, revision=head, bundle_sha256=sha(self.prod / 'bundle' / 'BUNDLE.json'))

    def source_gate(self):
        bundle = read_json(self.prod / 'bundle' / 'BUNDLE.json')
        problems, commands = {}, []
        for side in ('orix', 'ibex'):
            deploy = read_json(self.config['deploy'][side])
            head = self.bridge.remote(side, ['git', '-C', deploy['source_checkout'], 'rev-parse', 'HEAD'])
            if head['returncode'] != 0 or head['stdout'].strip() != bundle['source']['revision']:
                problems[side + '_checkout'] = {'expected': bundle['source']['revision'], 'found': head['stdout'].strip(), 'stderr': head['stderr'][-300:]}
            probe = self.bridge.remote(side, ['test', '-f', deploy['inputs_root'] + '/MANIFEST-final-20261123.json'])
            if probe['returncode'] != 0:
                problems[side + '_inputs'] = deploy['inputs_root']
        if problems:
            commands = (self.prod / 'bundle' / 'REQUIRED-COMMANDS.md').read_text()
            raise Gate('source_or_inputs_not_synchronized', {'problems': problems, 'required_commands': commands})
        return {'status': 'synchronized', 'revision': bundle['source']['revision']}

    def stage(self):
        out = {}
        for side in ('orix', 'ibex'):
            deploy = read_json(self.config['deploy'][side])
            remote_dir = self.config['remote_bundle_dir'][side]
            listing = self.bridge.remote(side, ['test', '-f', deploy['campaign'] + '/frozen-manifest.json'])
            if listing['returncode'] == 0:
                check = self.bridge.remote(side, ['sha256sum', deploy['campaign'] + '/bundle/BUNDLE.json'])
                staged = check['stdout'].split()[0] if check['returncode'] == 0 else None
                if staged != sha(self.prod / 'bundle' / 'BUNDLE.json'):
                    raise Gate('%s_campaign_holds_another_bundle' % side, {'campaign': deploy['campaign'], 'staged_bundle_sha256': staged})
                out[side] = 'already_staged'
                continue
            self.bridge.remote(side, ['mkdir', '-p', remote_dir], read_only=False)
            for name in ('BUNDLE.json', 'suite-query-v2.json', 'suite-pretrain-v2.json'):
                record = self.bridge.rsync(side, str(self.prod / 'bundle' / name), remote_dir + '/' + name, upload=True)
                if record['returncode'] != 0:
                    raise Gate('%s_bundle_copy_failed' % side, record)
            record = self.bridge.rsync(side, self.config['deploy'][side], remote_dir + '/DEPLOY-%s.json' % side, upload=True)
            staged = self.bridge.remote(side, [self.bridge.config[side]['python'], self.bridge.config[side]['code_dir'] + '/campaign.py', 'stage', '--campaign', deploy['campaign'],
                                               '--bundle', remote_dir + '/BUNDLE.json', '--deploy', remote_dir + '/DEPLOY-%s.json' % side], timeout=7200, read_only=False)
            if staged['returncode'] != 0:
                raise Gate('%s_stage_refused' % side, staged)
            out[side] = 'staged'
        return out

    def release(self):
        rules = self.config.get('release') or {}
        if rules.get('approve') is not True or not str(rules.get('authority', '')).strip():
            raise Gate('release_approval_required', 'set release.approve true and release.authority in the reviewed orchestrate config')
        out = {}
        for side in ('orix', 'ibex'):
            deploy = read_json(self.config['deploy'][side])
            exists = self.bridge.remote(side, ['test', '-f', deploy['campaign'] + '/ROOT_RELEASE.json'])
            if exists['returncode'] == 0:
                out[side] = 'already_released'
                continue
            command = [self.bridge.config[side]['python'], self.bridge.config[side]['code_dir'] + '/campaign.py', 'release', '--campaign', deploy['campaign'],
                       '--approve', '--authority', rules['authority']]
            if side == 'orix':
                command += ['--partition-choice', rules.get('orix_partition_choice', 'batch')]
                if rules.get('orix_max_active'):
                    command += ['--max-active', str(rules['orix_max_active'])]
            if side == 'ibex' and self.config.get('storage_decision'):
                remote = self.config['remote_bundle_dir']['ibex'] + '/STORAGE-DECISION.json'
                self.bridge.rsync('ibex', self.config['storage_decision'], remote, upload=True)
                command += ['--storage-decision', remote]
            if side == 'ibex' and rules.get('ibex_quota_command'):
                command += ['--quota-command', json.dumps(rules['ibex_quota_command'])]
            result = self.bridge.remote(side, command, timeout=600, read_only=False)
            if result['returncode'] != 0:
                raise Gate('%s_release_refused' % side, result)
            out[side] = 'released'
        return out

    def run(self, once, no_bridge):
        steps = [('wait-hpo', lambda: self.wait_hpo(once)), ('check-hpo', self.check_hpo), ('protocols', self.protocols), ('reuse', self.reuse),
                 ('bundle', self.bundle), ('source-gate', self.source_gate), ('stage', self.stage), ('release', self.release)]
        report = {'dry_run': self.dry_run, 'steps': {}}
        try:
            for name, function in steps:
                if self.done(name):
                    report['steps'][name] = 'done'
                    continue
                if self.dry_run and name not in ('wait-hpo',):
                    report['steps'][name] = 'pending'
                    continue
                value = function()
                self.record(name, {'result': value})
                report['steps'][name] = 'done_now'
        except Gate as gate:
            report.update({'gate': gate.name, 'detail': gate.detail})
            if not self.dry_run:
                write_replace(self.folder / 'GATE.json', {'gate': gate.name, 'detail': gate.detail, 'time': time.time(),
                                                          'resume': 're-run this same command after resolving the gate; completed steps are skipped'})
            print(json.dumps(report, indent=2, default=str))
            return EXIT_GATE
        if (self.folder / 'GATE.json').exists() and not self.dry_run:
            os.replace(str(self.folder / 'GATE.json'), str(self.folder / ('GATE-resolved-%d.json' % time.time())))
        command = [sys.executable, str(HERE / 'bridge.py'), 'loop', '--config', self.config['bridge_config']]
        report['bridge_command'] = command
        print(json.dumps(report, indent=2, default=str), flush=True)
        if self.dry_run or no_bridge:
            return 0
        os.execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='reviewed orchestrate config (see notes/.../production/orchestrate.example.json)')
    parser.add_argument('--once', action='store_true', help='check the HPO once instead of polling until DONE')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-bridge', action='store_true', help='stop after release instead of exec-ing the bridge loop')
    args = parser.parse_args()
    return Orchestrator(args.config, args.dry_run).run(args.once, args.no_bridge)


if __name__ == '__main__':
    raise SystemExit(main())
