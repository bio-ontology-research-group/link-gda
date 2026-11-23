"""Stage the suite HPO campaign directory once (login node; no GPU, no submission, no release).

Inputs come from a coordinator-written JSON (schema clmpt-suite-hpo-inputs/1) naming the admitted bank, the two
initializer exports, the CLMPT reference directory, the query interpreter and the historical trees the campaign must
not overlap. Every input is read in place and never modified. Verifies the committed, clean, pushed package; the
campaign protocol against the query worker's frozen protocol (grid, budgets, initializer recipes, bank admission);
the reference hashes; the bank file by file; each initializer by size and SHA-256 (and against its delivery
ADMISSION.json when one sits next to it); and the interpreter versions. Copies the worker, its protocol and the
references into the campaign, builds the three qualification and eighteen screen configs, validates every one with
the copied worker's own check_config, renders the two Slurm scripts and writes a write-once frozen manifest.
"""
import argparse
import ast
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('suite_common', str(HERE / 'suite.py'))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

INPUTS_SCHEMA = 'clmpt-suite-hpo-inputs/1'


def plain_path(value, label):
    path = Path(value)
    C.check(path.is_absolute() and os.path.realpath(str(path)) == str(path), '%s must be an absolute symlink-free path: %s' % (label, value))
    return path


def worker_files():
    folder = HERE / 'query' / 'code'
    C.check((folder / 'train.py').is_file(), 'query worker train.py missing')
    names = sorted(path.name for path in folder.iterdir() if path.is_file() and not path.name.startswith('.') and path.suffix != '.pyc')
    constants = {}
    for node in ast.parse((folder / 'train.py').read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], 'id', None) in ('SOURCE_FILES', 'REFERENCE_FILES'):
            constants[node.targets[0].id] = list(ast.literal_eval(node.value))
    C.check(set(constants) == {'SOURCE_FILES', 'REFERENCE_FILES'} and set(constants['SOURCE_FILES']) <= set(names), 'worker SOURCE_FILES/REFERENCE_FILES not found or incomplete')
    return names, constants['SOURCE_FILES'], constants['REFERENCE_FILES']


def check_worker_protocol(protocol, worker_protocol):
    """The campaign grid must be admitted by the worker protocol, not merely resemble it."""
    C.check(worker_protocol['status'] == 'FROZEN' and worker_protocol['protocol_id'] == protocol['worker_protocol']['protocol_id'] and worker_protocol['test_access'] is False, 'worker protocol not frozen, not the pinned id, or admits TEST')
    C.check(set(protocol['query_learning_rates']) <= set(worker_protocol['query_learning_rates']) and set(protocol['temperatures']) <= set(worker_protocol['temperatures']), 'campaign grid outside the worker protocol')
    recipes = [{key: row[key] for key in ('pretrain_learning_rate', 'pretrain_regularization', 'initializer_update_budget')} for row in protocol['initializer_candidates']]
    C.check(sorted(map(json.dumps, recipes)) == sorted(map(json.dumps, worker_protocol['hpo_initializer_recipes'])), 'initializer candidates differ from the worker hpo_initializer_recipes')
    purposes = worker_protocol['purposes']
    C.check(purposes['hpo_screen']['maximum_steps'] == [protocol['stages']['S_screen']['steps']] == [protocol['continuation_parent_steps']], 'screen budget differs')
    C.check(purposes['hpo_continuation']['maximum_steps'] == [protocol['continuation_total_steps']] and purposes['hpo_continuation']['parent_steps'] == protocol['continuation_parent_steps'], 'continuation budget differs')
    C.check(protocol['qualification']['steps'] in purposes['qualification']['maximum_steps'], 'qualification budget not admitted')
    for purpose in ('hpo_screen', 'hpo_continuation', 'qualification'):
        cadence = purposes[purpose]['evaluation_every_updates']
        C.check(protocol['evaluation_every_updates'] in (cadence if isinstance(cadence, list) else [cadence]), 'evaluation cadence not admitted for %s' % purpose)
    for key, value in protocol['fixed_query_settings'].items():
        for purpose in ('hpo_screen', 'hpo_continuation'):
            C.check(value in purposes[purpose].get('fixed', {}).get(key, [value]), 'fixed setting %s not admitted for %s' % (key, purpose))
    bank = worker_protocol['admitted_banks'].get(protocol['admitted_bank_complete_sha256'])
    C.check(bank is not None and {'qualification', 'hpo'} <= set(bank['purpose_admitted']), 'bank not admitted by the worker protocol for qualification and HPO')
    C.check(bank['validation_cases'] == protocol['clinical']['validation_cases'] and bank['validation_associations'] == protocol['clinical']['validation_associations']
            and bank['candidate_genes'] == protocol['clinical']['candidate_genes'], 'bank validation counts differ')
    C.check(protocol['runtime']['worker_maximum_seconds'] <= worker_protocol['maximum_seconds'], 'worker maximum seconds not admitted')


def interpreter(python, protocol, rehearsal=False):
    probe = 'import json, sys, torch, numpy; print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": numpy.__version__}))'
    result = subprocess.run([python, '-I', '-c', probe], capture_output=True, text=True, timeout=300)
    C.check(result.returncode == 0, 'interpreter probe failed: %s' % result.stderr[-1000:])
    versions = json.loads(result.stdout.strip().splitlines()[-1])
    matches = versions['torch'].split('+')[0] == protocol['interpreter']['torch'] and versions['numpy'] == protocol['interpreter']['numpy']
    C.check(matches or rehearsal, 'interpreter versions differ from the protocol: %s' % versions)
    return dict(versions, matches_protocol=matches)


def initializer_record(candidate, path):
    C.check(path.is_file(), 'initializer missing: %s' % path)
    size = path.stat().st_size
    C.check(size == candidate['expected_bytes'], 'initializer %s has %d bytes, protocol expects %d' % (candidate['name'], size, candidate['expected_bytes']))
    digest = C.sha(path)
    record = {key: candidate[key] for key in ('name', 'trajectory') + C.INITIALIZER_KEYS}
    record.update({'path': str(path), 'sha256': digest, 'bytes': size, 'delivery_admission': None})
    admission_path = path.parent / 'ADMISSION.json'
    if admission_path.exists():
        admission = C.read_json(admission_path)
        C.check(admission.get('files', {}).get(path.name) == digest, 'initializer hash differs from its delivery ADMISSION.json: %s' % path)
        C.check(admission.get('initializer_update_budget') in (None, candidate['initializer_update_budget']), 'delivery budget differs: %s' % path)
        C.check(admission.get('trajectory') in (None, candidate['trajectory']), 'delivery trajectory differs: %s' % path)
        record['delivery_admission'] = {'path': str(admission_path), 'sha256': C.sha(admission_path)}
        for name in ('identity.json', 'config.json'):
            if (path.parent / name).exists() and name in admission.get('files', {}):
                C.check(C.sha(path.parent / name) == admission['files'][name], 'delivery %s differs from ADMISSION.json' % name)
        if (path.parent / 'config.json').exists():
            config = C.read_json(path.parent / 'config.json')
            C.check(config.get('learning_rate') == candidate['pretrain_learning_rate'] and config.get('regularization') == candidate['pretrain_regularization'], 'delivery pretraining config differs from the candidate: %s' % path)
            record['pretrain_config_sha256'] = C.sha(path.parent / 'config.json')
            record['pretrain_graph_complete_sha256'] = config.get('data_complete_sha256')
    return record


def planned_items(protocol, worker_protocol_path, bank_dir, initializers, folder):
    """Plan for the three qualification and eighteen screen configs; worker_configs.py builds them with the worker's make_config."""
    seconds, every = protocol['runtime']['worker_maximum_seconds'], protocol['evaluation_every_updates']
    fixed = protocol['fixed_query_settings']
    plan = protocol['qualification']
    items = {}
    for pair in plan['pairs']:
        for identifier in (pair['reference'], pair['resumed']):
            items[identifier] = C.plan_item(Path(folder) / ('%s.json' % identifier), worker_protocol_path, 'qualification', identifier, bank_dir,
                                            initializers[plan['initializer']], fixed, plan['learning_rate'], plan['temperature'], plan['steps'], every, seconds)
    screen = protocol['stages']['S_screen']
    for candidate, learning_rate, temperature in itertools.product(protocol['initializer_candidates'], protocol['query_learning_rates'], protocol['temperatures']):
        identifier = C.screen_id(candidate['name'], learning_rate, temperature)
        items[identifier] = C.plan_item(Path(folder) / ('%s.json' % identifier), worker_protocol_path, screen['purpose'], identifier, bank_dir,
                                        initializers[candidate['name']], fixed, learning_rate, temperature, screen['steps'], every, seconds)
    C.check(len(items) == 3 + 18, 'planned config count differs: %d' % len(items))
    return items


def check_planned(config, identifier, worker, bank_complete, protocol):
    """A worker-built config must pin exactly this campaign's worker, protocol, bank and fixed settings."""
    C.check(config['experiment_id'] == identifier and config['source_hashes'] == worker['source_hashes'] and config['reference_hashes'] == worker['reference_hashes'],
            'planned config %s pins other worker sources or references' % identifier)
    C.check(config['protocol_path'] == worker['protocol_path'] and config['protocol_sha256'] == worker['protocol_sha256'] and config['data_complete_sha256'] == bank_complete,
            'planned config %s pins another protocol or bank' % identifier)
    C.check(all(config[key] == value for key, value in protocol['fixed_query_settings'].items()) and config.get('test_evaluation', False) is False,
            'planned config %s differs in a fixed setting' % identifier)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--inputs', required=True, help='coordinator JSON, schema clmpt-suite-hpo-inputs/1')
    parser.add_argument('--local-check-uncommitted', action='store_true',
                        help='skip the Git check to rehearse staging; the manifest is marked unverified and advance.py release refuses it')
    args = parser.parse_args()
    try:
        inputs_path = Path(args.inputs)
        inputs = C.read_json(inputs_path)
        C.check(inputs.get('schema') == INPUTS_SCHEMA, 'inputs schema differs')
        names, source_files, reference_files = worker_files()
        protocol_file = HERE / 'hpo-protocol.json'
        protocol = C.read_json(protocol_file)
        worker_protocol_file = HERE / protocol['worker_protocol']['package_relpath']
        package = list(C.PACKAGE_FILES) + ['query/code/' + name for name in names] + [protocol['worker_protocol']['package_relpath']]
        if args.local_check_uncommitted:
            checkout = {'package_status_clean': False, 'unverified_local_rehearsal': True}
        else:
            checkout = C.verified_checkout(args.code_revision, package)
        campaign = C.campaign_root(args.campaign)
        C.check(not (campaign / 'frozen-manifest.json').exists(), 'manifest exists: staging is write-once')
        C.check(not campaign.exists() or not any(campaign.iterdir()), 'campaign directory exists and is not empty')
        C.check(protocol['status'] == 'FROZEN' and protocol['test_access'] is False and protocol['clinical']['test_evaluation'] is False, 'campaign protocol not frozen or admits TEST')
        worker_protocol = C.read_json(worker_protocol_file)
        check_worker_protocol(protocol, worker_protocol)
        protected = [plain_path(value, 'protected tree') for value in inputs.get('protected_trees', [])] + [HERE]
        for root in protected:
            C.check(not (str(campaign) == str(root) or str(campaign).startswith(str(root) + os.sep) or str(root).startswith(str(campaign) + os.sep)), 'campaign overlaps protected tree %s' % root)
        references = plain_path(inputs['references_dir'], 'references_dir')
        C.check(sorted(reference_files) == sorted(worker_protocol['reference_hashes']), 'worker REFERENCE_FILES differ from the worker protocol')
        for name, digest in worker_protocol['reference_hashes'].items():
            C.check(C.sha(references / name) == digest, 'reference differs from its pin: %s' % name)
        bank_dir = plain_path(inputs['bank_dir'], 'bank_dir')
        bank_complete = C.sha(bank_dir / 'COMPLETE.json')
        C.check(bank_complete == protocol['admitted_bank_complete_sha256'], 'bank COMPLETE not admitted: %s' % bank_complete)
        bank_files = {}
        for name, record in C.read_json(bank_dir / 'COMPLETE.json')['files'].items():
            digest = record['sha256'] if isinstance(record, dict) else record
            C.check(C.sha(bank_dir / name) == digest, 'bank file differs: %s' % name)
            bank_files[name] = {'sha256': digest, 'bytes': (bank_dir / name).stat().st_size}
        bank = {'dir': str(bank_dir), 'complete_sha256': bank_complete, 'files': bank_files, 'use': 'read-only in place'}
        candidates = {row['name']: row for row in protocol['initializer_candidates']}
        C.check(set(inputs['initializers']) == set(candidates), 'inputs must name exactly the protocol initializer candidates %s' % sorted(candidates))
        initializers = {name: initializer_record(candidates[name], plain_path(inputs['initializers'][name], 'initializer')) for name in sorted(candidates)}
        C.check(len({record['sha256'] for record in initializers.values()}) == len(initializers), 'two initializer candidates have identical weights')
        python = inputs['python']
        versions = interpreter(python, protocol, rehearsal=args.local_check_uncommitted)
        for folder in ('config/planned', 'query/code', 'query/protocols', 'query/references', 'ops', 'logs', 'trials', 'decisions', 'state'):
            (campaign / folder).mkdir(parents=True, exist_ok=True)
        files = {}

        def place(relative, data):
            C.write_exclusive(campaign, campaign / relative, data)
            files[relative] = C.sha(campaign / relative)
        place('config/protocol.json', protocol_file.read_bytes())
        place('config/inputs.json', inputs_path.read_bytes())
        place(C.WORKER_PROTOCOL, worker_protocol_file.read_bytes())
        for name in names:
            place('query/code/' + name, (HERE / 'query' / 'code' / name).read_bytes())
        for name in reference_files:
            place('query/references/' + name, (references / name).read_bytes())
        worker = {'protocol_path': str(campaign / C.WORKER_PROTOCOL), 'protocol_sha256': files[C.WORKER_PROTOCOL], 'protocol_id': worker_protocol['protocol_id'],
                  'source_files': source_files, 'reference_files': reference_files, 'copied_files': names,
                  'source_hashes': {name: files['query/code/' + name] for name in source_files},
                  'reference_hashes': {name: files['query/references/' + name] for name in reference_files}}
        items = planned_items(protocol, worker['protocol_path'], bank_dir, initializers, campaign / 'config' / 'planned')
        place('config/planned-plan.json', C.encoded({'configs': [items[identifier] for identifier in sorted(items)]}))
        built = subprocess.run(C.worker_configs_command(python, campaign / 'query' / 'code', campaign / 'config' / 'planned-plan.json', campaign / 'config' / 'planned-receipt.json'),
                               capture_output=True, text=True, timeout=1800)
        C.check(built.returncode == 0, 'worker make_config refused a planned config: %s' % built.stderr[-3000:])
        receipt = C.read_json(campaign / 'config' / 'planned-receipt.json')
        C.check(receipt['passed'] is True and sorted(row['experiment_id'] for row in receipt['configs']) == sorted(items), 'planned config receipt incomplete')
        files['config/planned-receipt.json'] = C.sha(campaign / 'config' / 'planned-receipt.json')
        configs = {}
        for identifier in sorted(items):
            relative = 'config/planned/%s.json' % identifier
            check_planned(C.read_json(campaign / relative), identifier, worker, bank_complete, protocol)
            configs[identifier] = files[relative] = C.sha(campaign / relative)
        for template, target in (('ops/job-template.sh', 'ops/job.sh'), ('ops/cpu-job-template.sh', 'ops/cpu_job.sh')):
            text = (HERE / template).read_text().replace('@PYTHON@', shlex.quote(python)).replace('@CODE_DIR@', shlex.quote(str(HERE)))
            C.check('@' not in text.replace('$@', ''), 'unrendered placeholder in %s' % template)
            place(target, text.encode())
            os.chmod(str(campaign / target), 0o755)
        manifest = {'schema': C.MANIFEST_SCHEMA, 'campaign': str(campaign), 'code_dir': str(HERE), 'code_revision': args.code_revision, 'git': checkout,
                    'code_files': {name: C.sha(HERE / name) for name in package}, 'files': files, 'protocol_sha256': C.sha(protocol_file), 'worker': worker,
                    'planned_configs': configs, 'bank': bank,
                    'initializers': initializers, 'references_source': str(references), 'python': python, 'interpreter': versions,
                    'protected_trees': [str(root) for root in protected], 'inputs_sha256': files['config/inputs.json']}
        C.write_exclusive(campaign, campaign / 'frozen-manifest.json', manifest)
        print(json.dumps({'campaign': str(campaign), 'manifest_sha256': C.sha(campaign / 'frozen-manifest.json'), 'code_revision': args.code_revision,
                          'protocol_sha256': manifest['protocol_sha256'], 'worker_protocol_sha256': worker['protocol_sha256'],
                          'initializers': {name: record['sha256'] for name, record in initializers.items()}, 'planned_configs': len(configs), 'interpreter': versions}, indent=2))
        return 0
    except C.Refusal as error:
        print(json.dumps({'refused': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
