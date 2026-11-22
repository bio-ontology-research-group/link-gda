"""Stage the boundary query campaign directory once (login node; no GPU, no submission, no release).

Requires this package to be the committed, clean, pushed revision. Verifies the frozen protocol, the executed query
sources and the pinned CLMPT reference files by hash, the admitted bank file by file, the control screen and
control continuation by their pinned hashes, and the interpreter versions. Copies the worker sources and references
into the campaign (the worker resolves references relative to its own directory), renders the two Slurm scripts and
writes a write-once frozen manifest that pins every campaign file and every package file in the checkout.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location('boundary_campaign', str(Path(__file__).resolve().parent / 'campaign.py'))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def verified_control(protocol, bank_complete, root):
    pins = protocol['control']
    check, sha = C.check, C.sha
    check(sha(root / pins['protocol_relpath']) == pins['protocol_sha256'], 'control protocol hash differs')
    records = {}
    for part in ('screen', 'long'):
        training = root / pins[part]['relpath']
        check(training.name == 'training' and training.parent.name == pins[part]['trial_id'], 'control %s directory does not name its trial' % part)
        check(sha(training.parent / 'config.json') == pins[part]['config_sha256'], 'control %s config differs' % part)
        check(sha(training / 'COMPLETE.json') == pins[part]['complete_sha256'], 'control %s COMPLETE differs' % part)
        complete = C.read_json(training / 'COMPLETE.json')
        for name in ('identity.json', 'progress.json', 'curve.json', 'selection.json'):
            check(sha(training / name) == complete['files'][name], 'control %s %s differs from its COMPLETE' % (part, name))
        config = C.read_json(training.parent / 'config.json')
        check(config['protocol_sha256'] == pins['protocol_sha256'] and config['pretrained_sha256'] == pins['pretrained_sha256'], 'control %s protocol or initializer differs' % part)
        check(config['data_complete_sha256'] == bank_complete, 'control %s bank differs' % part)
        check(config['source_hashes'] == protocol['query_source_hashes'], 'control %s sources differ from the executed sources' % part)
        records[part] = {'steps': complete['steps'], 'identity_sha256': complete['identity_sha256'], 'config': config}
    check(records['screen']['steps'] == protocol['continuation_parent_steps'] and records['long']['steps'] == protocol['continuation_total_steps'], 'control step counts differ')
    check(records['long']['config']['continuation_parent'] == str(root / pins['screen']['relpath']), 'control continuation parent differs')
    screen = protocol['stages']['S2_screen']
    config = records['screen']['config']
    check((config['learning_rate'], config['temperature'], config['mixture_id'], config['initializer_update_budget'], config['pretrain_learning_rate'])
          == (screen['learning_rate'], screen['temperature'], screen['mixture_id'], 633450, 0.2), 'control screen recipe differs from the frozen screen')
    check(config['mixture'] == protocol['mixtures'][screen['mixture_id']], 'control mixture differs')
    value = {part: {'trial_id': pins[part]['trial_id'], 'training_dir': str(root / pins[part]['relpath']), 'complete_sha256': pins[part]['complete_sha256'],
                    'identity_sha256': records[part]['identity_sha256'], 'steps': records[part]['steps']} for part in ('screen', 'long')}
    value.update({'campaign': str(root), 'protocol_path': str(root / pins['protocol_relpath']), 'protocol_sha256': pins['protocol_sha256']})
    return value


def interpreter(python):
    probe = 'import json, sys, torch, numpy; print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": numpy.__version__}))'
    result = subprocess.run([python, '-I', '-c', probe], capture_output=True, text=True, timeout=300)
    C.check(result.returncode == 0, 'interpreter probe failed: %s' % result.stderr[-1000:])
    versions = json.loads(result.stdout.strip().splitlines()[-1])
    C.check(versions['torch'].split('+')[0] == '2.6.0' and versions['numpy'] == '1.26.4', 'interpreter versions differ from requirements-query.txt: %s' % versions)
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--bank-dir', required=True, help='admitted original bank, used read-only in place')
    parser.add_argument('--control-campaign', required=True, help='completed campaign holding the control screen and continuation, used read-only')
    parser.add_argument('--references-dir', required=True, help='directory holding the four pinned official CLMPT reference files')
    args = parser.parse_args()
    try:
        here = C.HERE
        checkout = C.verified_checkout(args.code_revision)
        campaign = C.campaign_root(args.campaign)
        C.check(not (campaign / 'frozen-manifest.json').exists(), 'manifest exists: staging is write-once')
        C.check(not campaign.exists() or not any(campaign.iterdir()), 'campaign directory exists and is not empty')
        protocol_file = here / 'protocol.json'
        protocol = C.read_json(protocol_file)
        C.check(protocol['status'] == 'FROZEN' and protocol['test_access'] is False and protocol['clinical']['test_evaluation'] is False, 'protocol not frozen or admits TEST')
        old_root = Path(args.control_campaign)
        C.check(old_root.is_absolute() and os.path.realpath(str(old_root)) == str(old_root), 'control campaign must be an absolute symlink-free path')
        for root in (old_root, here):
            C.check(not (str(campaign) == str(root) or str(campaign).startswith(str(root) + os.sep) or str(root).startswith(str(campaign) + os.sep)), 'campaign overlaps protected tree %s' % root)
        for name in C.QUERY_SOURCES:
            C.check(C.sha(here / 'query' / 'code' / name) == protocol['query_source_hashes'][name], 'query source differs from the executed source: %s' % name)
        C.check(C.sha(here / 'analysis' / 'select_stage.py') == protocol['selector_sha256'], 'selector differs from the frozen selector')
        references = Path(args.references_dir)
        for name in C.REFERENCES:
            C.check(C.sha(references / name) == protocol['reference_hashes'][name], 'reference differs from its pin: %s' % name)
        bank_dir = Path(args.bank_dir)
        C.check(os.path.realpath(str(bank_dir)) == str(bank_dir), 'bank path contains a symlink')
        bank_complete = C.sha(bank_dir / 'COMPLETE.json')
        C.check(protocol['admitted_bank_complete_sha256'].get(bank_complete) == 'original', 'bank COMPLETE not admitted: %s' % bank_complete)
        bank_record = C.read_json(bank_dir / 'COMPLETE.json')
        C.check(bank_record.get('mode') == 'original', 'bank mode differs')
        bank_files = {}
        for name, record in bank_record['files'].items():
            digest = record['sha256'] if isinstance(record, dict) else record
            C.check(C.sha(bank_dir / name) == digest, 'bank file differs: %s' % name)
            bank_files[name] = {'sha256': digest, 'bytes': (bank_dir / name).stat().st_size}
        control = verified_control(protocol, bank_complete, old_root)
        versions = interpreter(args.python)
        for folder in ('config', 'query/code', 'query/references', 'ops', 'logs', 'trials', 'initializers', 'incoming', 'decisions', 'state'):
            (campaign / folder).mkdir(parents=True, exist_ok=True)
        files = {}
        C.write_exclusive(campaign, campaign / 'config' / 'protocol.json', protocol_file.read_bytes())
        files['config/protocol.json'] = C.sha(campaign / 'config' / 'protocol.json')
        for name in C.QUERY_SOURCES:
            C.write_exclusive(campaign, campaign / 'query' / 'code' / name, (here / 'query' / 'code' / name).read_bytes())
            files['query/code/' + name] = C.sha(campaign / 'query' / 'code' / name)
        for name in C.REFERENCES:
            C.write_exclusive(campaign, campaign / 'query' / 'references' / name, (references / name).read_bytes())
            files['query/references/' + name] = C.sha(campaign / 'query' / 'references' / name)
        for template, target in (('job-template.sh', 'ops/job.sh'), ('select-job-template.sh', 'ops/select_job.sh')):
            text = (here / template).read_text().replace('@PYTHON@', shlex.quote(args.python)).replace('@CODE_DIR@', shlex.quote(str(here)))
            C.check('@' not in text.replace('$@', ''), 'unrendered placeholder in %s' % template)
            C.write_exclusive(campaign, campaign / target, text.encode())
            os.chmod(str(campaign / target), 0o755)
            files[target] = C.sha(campaign / target)
        manifest = {'schema': 'clmpt-boundary-queries-manifest/1', 'campaign': str(campaign), 'code_dir': str(here), 'code_revision': args.code_revision, 'git': checkout,
                    'code_files': {name: C.sha(here / name) for name in C.CODE_FILES}, 'files': files, 'protocol_sha256': C.sha(protocol_file),
                    'bank': {'dir': str(bank_dir), 'complete_sha256': bank_complete, 'files': bank_files, 'use': 'read-only in place'},
                    'control': control, 'references_source': str(references), 'python': args.python, 'interpreter': versions}
        C.write_exclusive(campaign, campaign / 'frozen-manifest.json', manifest)
        print(json.dumps({'campaign': str(campaign), 'manifest_sha256': C.sha(campaign / 'frozen-manifest.json'), 'code_revision': args.code_revision,
                          'protocol_sha256': manifest['protocol_sha256'], 'interpreter': versions}, indent=2))
        return 0
    except C.Refusal as error:
        print(json.dumps({'refused': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
