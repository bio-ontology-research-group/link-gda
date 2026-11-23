"""CPU preflight for the suite HPO campaign; writes one JSON receipt and touches nothing outside --scratch.

1. compiles every root orchestration file;
2. checks the campaign protocol against the worker protocol (the same function the stager uses);
3. builds the 3 qualification and 18 screen configs with the worker's own make_config.build_config and validates each
   with validate_config (worker_configs.py), then applies the stager's pin checks;
4. builds one hpo_continuation config from a mocked completed parent and checks it differs from its screen only in the
   continuation keys;
5. runs select_hpo.py and the worker's leaderboard.py on synthetic trajectories (screen, final, recipe) that include
   exact metric ties, and requires identical orders, promotions, winners and production recipe;
6. runs the engine-qualification admission check on synthetic receipt sets (complete, one condition missing, foreign
   worker sources);
7. checks the controller deadline arithmetic and the storage peaks.
Synthetic trajectories carry mocked metrics and file hashes; they test selection logic only and say nothing about
training. Run with the query interpreter (torch) from the repository root.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(HERE / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def compile_all():
    rows = {}
    for path in sorted(list(HERE.glob('*.py')) + list((HERE / 'ops').glob('*.py'))):
        compile(path.read_text(), str(path), 'exec')
        rows[str(path.relative_to(HERE))] = True
    return rows


def metrics(h10, h1, mrr, mr, cases=545, associations=586):
    micro = {'hits@10': h10, 'hits@1': h1, 'mrr': mrr, 'mr': mr}
    return {'queries': cases, 'associations': associations, 'answer_micro': micro, 'query_macro': micro}


def fake_trajectory(root, identifier, config, curve_values, worker_protocol_sha):
    """Completed trajectory directory with mocked metrics; files exist so both verifiers can hash them."""
    training = root / identifier / 'training'
    (training / 'fits').mkdir(parents=True)
    (training / 'models').mkdir()
    identity = {'config': config, 'config_sha256': 'mock', 'sources': {}, 'references': {}, 'protocol_sha256': worker_protocol_sha}
    dump(training / 'identity.json', identity)
    curve, steps = [], config['maximum_steps']
    for number, step in enumerate(range(0, steps + 1, config['evaluation_every_updates'])):
        raw, calibrated = curve_values(number)
        scores, fit = training / 'fits' / ('fit-step-%06d.npz' % step), training / 'fits' / ('fit-step-%06d.json' % step)
        scores.write_bytes(('%s-%d' % (identifier, step)).encode())
        fit.write_text('{}\n')
        curve.append({'step': step, 'validation': {'real': {'raw': metrics(*raw), 'calibrated': metrics(*calibrated)}}, 'scores': str(scores), 'scores_sha256': sha(scores),
                      'fit_metadata': str(fit), 'fit_metadata_sha256': sha(fit), 'checkpoint': {'path': 'mock', 'sha256': 'mock', 'step': step}})
    dump(training / 'curve.json', curve)
    sys.path.insert(0, str(HERE / 'query' / 'code'))
    from metrics import selector
    selection = {view: selector(curve, view) for view in ('raw', 'calibrated')}
    dump(training / 'selection.json', selection)
    pointers = {}
    for view in ('raw', 'calibrated'):
        model = training / 'models' / ('step-%06d.pt' % selection[view]['step'])
        model.write_bytes(('model-%s-%d' % (identifier, selection[view]['step'])).encode())
        pointers[view] = {'path': str(model.relative_to(training)), 'sha256': sha(model), 'step': selection[view]['step'], 'root': str(training), 'kind': 'model_only'}
    initial = training / 'models' / 'step-000000.pt'
    if not initial.exists():
        initial.write_bytes(('model-%s-0' % identifier).encode())
    checkpoint = training / 'checkpoints' / ('step-%06d-1.pt' % steps)
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(('state-%s' % identifier).encode())
    progress = {'step': steps, 'pending_evaluation': None, 'checkpoint': str(checkpoint.relative_to(training)), 'checkpoint_sha256': sha(checkpoint),
                'retained_best': pointers, 'artifacts': {str(checkpoint.relative_to(training)): sha(checkpoint)},
                'initial_checkpoint': {'path': 'models/step-000000.pt', 'sha256': sha(initial), 'step': 0, 'root': str(training), 'kind': 'model_only'}}
    dump(training / 'progress.json', progress)
    files = {name: sha(training / name) for name in ('curve.json', 'selection.json', 'identity.json', 'progress.json')}
    dump(training / 'COMPLETE.json', {'identity_sha256': canonical(identity), 'steps': steps, 'files': files})
    return {'id': identifier, 'trial_dir': str(training.parent), 'training_dir': str(training), 'complete_sha256': sha(training / 'COMPLETE.json')}


def selector_agreement(scratch, protocol_path, worker_protocol_path, decision_path, planned):
    protocol = json.loads(Path(protocol_path).read_text())
    worker_sha = sha(worker_protocol_path)
    root = scratch / 'synthetic'
    screens = sorted(identifier for identifier in planned if identifier.startswith('S-'))
    tie = (.30, .10, .15, 900.0)

    def screen_curve(index):
        def values(number):
            if number == 0:
                return (.20, .05, .10, 1200.0), (.21, .06, .11, 1150.0)
            if index in (3, 11, 7):
                return tie, tie
            return (.20 + .005 * index, .06, .12, 1000.0 - index), (.22 + .004 * ((index * 7) % 18), .07, .13, 990.0 - index)
        return values
    entries = []
    for index, identifier in enumerate(screens):
        config = dict(planned[identifier])
        config['protocol_sha256'] = worker_sha
        entries.append(fake_trajectory(root / 'S', identifier, config, screen_curve(index), worker_sha))
    index_path = scratch / 'screen-index.json'
    dump(index_path, {'stage': 'screen', 'protocol_sha256': sha(protocol_path), 'entries': entries})
    mine, theirs = scratch / 'screen-mine.json', scratch / 'screen-theirs.json'
    run_mine = subprocess.run([sys.executable, '-I', str(HERE / 'select_hpo.py'), '--stage', 'screen', '--protocol', str(protocol_path), '--worker-protocol', str(worker_protocol_path),
                               '--index', str(index_path), '--output', str(mine)], capture_output=True, text=True)
    run_theirs = subprocess.run([sys.executable, '-I', str(HERE / 'query' / 'code' / 'leaderboard.py'), 'promote', '--protocol', str(worker_protocol_path),
                                 '--index', str(index_path), '--output', str(theirs)], capture_output=True, text=True)
    result = {'screen_mine_rc': run_mine.returncode, 'screen_theirs_rc': run_theirs.returncode, 'screen_mine_stderr': run_mine.stderr[-1500:], 'screen_theirs_stderr': run_theirs.stderr[-1500:]}
    if run_mine.returncode or run_theirs.returncode:
        result['passed'] = False
        return result
    left, right = json.loads(mine.read_text()), json.loads(theirs.read_text())
    result['screen_orders_equal'] = all([row['id'] for row in left['leaderboards'][view]] == [row['id'] for row in right['leaderboards'][view]] for view in ('raw', 'calibrated'))
    result['screen_promoted_equal'] = sorted(left['promoted']) == sorted(right['promoted_ids'])
    result['promoted'] = left['promoted']
    tied = [screens[3], screens[7], screens[11]]
    raw_order = [row['id'] for row in left['leaderboards']['raw']]
    result['exact_tie_ordered_by_id_only'] = [identifier for identifier in raw_order if identifier in tied] == sorted(tied)
    continuations = []
    for number, identifier in enumerate(left['promoted']):
        config = dict(planned[identifier], purpose='hpo_continuation', maximum_steps=20000, experiment_id='C-' + identifier[2:], protocol_sha256=worker_sha)

        def values(step_number, number=number):
            if step_number == 0:
                return (.20, .05, .10, 1200.0), (.21, .06, .11, 1150.0)
            return (.25 + .01 * number, .08, .14, 950.0), (.27 + .01 * ((number + 1) % 4), .09, .15, 940.0)
        continuations.append(fake_trajectory(root / 'C', config['experiment_id'], config, values, worker_sha))
    final_index = scratch / 'final-index.json'
    dump(final_index, {'stage': 'final', 'protocol_sha256': sha(protocol_path), 'entries': continuations})
    final = scratch / 'final-mine.json'
    run_final = subprocess.run([sys.executable, '-I', str(HERE / 'select_hpo.py'), '--stage', 'final', '--protocol', str(protocol_path), '--worker-protocol', str(worker_protocol_path),
                                '--index', str(final_index), '--output', str(final)], capture_output=True, text=True)
    result.update({'final_rc': run_final.returncode, 'final_stderr': run_final.stderr[-1500:]})
    recipe_index = scratch / 'recipe-index.json'
    dump(recipe_index, {'stage': 'recipe', 'protocol_sha256': sha(protocol_path), 'entries': continuations, 'promotion_sha256': sha(mine),
                        'final_sha256': sha(final) if final.exists() else None, 'decision_sha256': sha(decision_path)})
    recipe, freeze = scratch / 'recipe-mine.json', scratch / 'recipe-theirs.json'
    run_recipe = subprocess.run([sys.executable, '-I', str(HERE / 'select_hpo.py'), '--stage', 'recipe', '--protocol', str(protocol_path), '--worker-protocol', str(worker_protocol_path),
                                 '--index', str(recipe_index), '--final', str(final), '--decision', str(decision_path), '--output', str(recipe)], capture_output=True, text=True)
    run_freeze = subprocess.run([sys.executable, '-I', str(HERE / 'query' / 'code' / 'leaderboard.py'), 'freeze', '--protocol', str(worker_protocol_path), '--index', str(recipe_index),
                                 '--decision', str(decision_path), '--output', str(freeze)], capture_output=True, text=True)
    result.update({'recipe_rc': run_recipe.returncode, 'recipe_stderr': run_recipe.stderr[-1500:], 'freeze_rc': run_freeze.returncode, 'freeze_stderr': run_freeze.stderr[-1500:]})
    if run_final.returncode or run_recipe.returncode or run_freeze.returncode:
        result['passed'] = False
        return result
    final_value, recipe_value, freeze_value = json.loads(final.read_text()), json.loads(recipe.read_text()), json.loads(freeze.read_text())
    result['final_orders_equal'] = all([row['id'] for row in final_value['leaderboards'][view]] == [row['id'] for row in freeze_value['leaderboards'][view]] for view in ('raw', 'calibrated'))
    result['winners_equal'] = all((final_value['winners'][view]['id'], final_value['winners'][view]['step'], final_value['winners'][view]['recipe'])
                                  == (freeze_value['separately_selected_winners'][view]['id'], freeze_value['separately_selected_winners'][view]['step'],
                                      freeze_value['separately_selected_winners'][view]['recipe']) for view in ('raw', 'calibrated'))
    result['production_recipe_equal'] = recipe_value['production_recipe'] == freeze_value['production_recipe']
    result['common_leaderboard'] = recipe_value['common_leaderboard']
    result['same_checkpoint_comparison_present'] = all('calibrated_minus_raw' in row['raw_selected'] for row in final_value['same_checkpoint_raw_calibrated'].values())
    result['calibration_benefit_presumed'] = recipe_value['calibration_benefit_presumed']
    result['passed'] = all(result[key] for key in ('screen_orders_equal', 'screen_promoted_equal', 'exact_tie_ordered_by_id_only', 'final_orders_equal', 'winners_equal',
                                                   'production_recipe_equal', 'same_checkpoint_comparison_present')) and result['calibration_benefit_presumed'] is False
    return result


def engine_gate(scratch, advance, protocol, worker, template):
    """advance.engine_receipts on synthetic receipts: a complete set admits; a missing condition or foreign sources refuse."""
    spec = protocol['engine_qualification']
    code = HERE / 'query' / 'code'

    def write_set(folder, drop=None, foreign=False):
        folder.mkdir(parents=True)
        dump(folder / 'construction.json', {'subcommand': 'construction', 'device': 'cuda', 'passed': True})
        dump(folder / 'scores.json', {'subcommand': 'scores', 'device': 'cuda', 'passed': True})
        for row in spec['query_resumes']:
            if drop == row:
                continue
            name = '%s-%s-s%d' % (row['encoder'], row['curriculum'], row['seed'])
            config = dict(template, purpose='qualification', encoder=row['encoder'], curriculum=row['curriculum'], seed=row['seed'], sampler_seed=row['seed'], experiment_id=name)
            if foreign:
                config['source_hashes'] = dict(config['source_hashes'], **{'train.py': '0' * 64})
            dump(folder / 'configs' / (name + '.json'), config)
            dump(folder / ('resume-%s.json' % name), {'subcommand': 'resume', 'engine': 'query', 'passed': True,
                                                      'calls': [{'command': [str(code / 'train.py'), '--config', str(folder / 'configs' / (name + '.json'))], 'returncode': 0}]})
        dump(folder / 'configs' / 'pretrain.json', {'seed': spec['pretrain_resume']['seed']})
        dump(folder / 'resume-pretrain.json', {'subcommand': 'resume', 'engine': 'pretrain', 'passed': True,
                                               'calls': [{'command': [str(HERE / 'pretrain' / 'train.py'), '--config', str(folder / 'configs' / 'pretrain.json')], 'returncode': 0}]})
        return folder
    qualify_sha = sha(code / 'qualify.py')
    complete = advance.engine_receipts(spec, worker, qualify_sha, write_set(scratch / 'engine-complete'), code)[2]
    missing = advance.engine_receipts(spec, worker, qualify_sha, write_set(scratch / 'engine-missing', drop=spec['query_resumes'][2]), code)[2]
    foreign = advance.engine_receipts(spec, worker, qualify_sha, write_set(scratch / 'engine-foreign', foreign=True), code)[2]
    return {'complete_set_problems': complete, 'missing_condition_problems': missing, 'foreign_source_problems': foreign,
            'passed': not complete and bool(missing) and bool(foreign), 'note': 'synthetic receipts; tests the gate logic only'}


def rehearse(campaign):
    """On a campaign staged with --local-check-uncommitted: create the trial directories exactly as tick would and classify them."""
    A = load('suite_advance', 'advance.py')
    context = A.load(campaign, require_release=False)
    assert context['manifest']['git'].get('unverified_local_rehearsal') is True, 'rehearsal only on a local-check campaign'
    qualification = A.qualification_trials(context)
    screens = A.screen_trials(context)
    folders = sorted({folder for _, _, folder in qualification}) + screens
    states = {folder.name: A.classify(context, folder, {}) for folder in folders}
    runtimes = {folder.name: json.loads((folder / 'runtime.json').read_text()) for folder in folders}
    configs_identical = all((folder / 'config.json').read_bytes() == (Path(campaign) / 'config' / 'planned' / (folder.name + '.json')).read_bytes() for folder in folders)
    commands = {name: context and A.C.sbatch_command(campaign, Path(campaign) / 'trials' / runtimes[name]['stage'] / name, 0, runtimes[name]) for name in ('Q-step-pause', screens[0].name)}
    value = {'trials': len(folders), 'states': sorted(set(states.values())), 'configs_identical_to_planned': configs_identical,
             'q_runtimes': {name: runtimes[name] for name in ('Q-ref', 'Q-step-pause', 'Q-time-pause')}, 'sbatch_examples': commands,
             'engine_gate_before_admission': A.engine_admitted(context)}
    value['passed'] = len(folders) == 21 and value['states'] == ['ready'] and configs_identical and value['engine_gate_before_admission'] is None
    return value


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--rehearse-campaign':
        value = rehearse(sys.argv[2])
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0 if value['passed'] else 1
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--inputs', required=True, help='inputs JSON (schema clmpt-suite-hpo-inputs/1); paths may be local copies')
    parser.add_argument('--initializer-sha256', action='append', default=[], help='NAME=SHA for an initializer not present here, from its delivery receipt')
    parser.add_argument('--decision', required=True, help='selection decision JSON')
    parser.add_argument('--scratch', required=True, help='new empty directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    C, S, K, controller = load('suite_common', 'suite.py'), load('suite_stage', 'stage_hpo.py'), load('suite_capacity', 'capacity.py'), load('suite_controller', 'controller.py')
    scratch = Path(args.scratch).resolve()
    assert not scratch.exists() or not any(scratch.iterdir()), 'scratch must be new or empty'
    scratch.mkdir(parents=True, exist_ok=True)
    receipt = {'compiled': compile_all()}
    protocol_path = HERE / 'hpo-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    worker_protocol_path = HERE / protocol['worker_protocol']['package_relpath']
    S.check_worker_protocol(protocol, json.loads(worker_protocol_path.read_text()))
    receipt['worker_protocol'] = {'path': str(worker_protocol_path), 'sha256': sha(worker_protocol_path), 'consistent_with_campaign_protocol': True}
    names, source_files, reference_files = S.worker_files()
    receipt['worker_source_files'] = source_files
    inputs = json.loads(Path(args.inputs).read_text())
    overrides = dict(item.split('=', 1) for item in args.initializer_sha256)
    initializers = {}
    for candidate in protocol['initializer_candidates']:
        path = Path(inputs['initializers'][candidate['name']])
        record = {key: candidate[key] for key in C.INITIALIZER_KEYS}
        if path.is_file():
            assert path.stat().st_size == candidate['expected_bytes']
            record.update(path=str(path), sha256=sha(path), sha_source='hashed here')
        else:
            record.update(path=str(path), sha256=overrides[candidate['name']], sha_source='delivery receipt (file not present here)')
        initializers[candidate['name']] = record
    receipt['initializers'] = {name: {key: record[key] for key in ('path', 'sha256', 'sha_source')} for name, record in initializers.items()}
    bank_dir = Path(inputs['bank_dir'])
    bank_complete = sha(bank_dir / 'COMPLETE.json')
    assert bank_complete == protocol['admitted_bank_complete_sha256'], 'bank COMPLETE differs: %s' % bank_complete
    planned_dir = scratch / 'planned'
    items = S.planned_items(protocol, str(worker_protocol_path), bank_dir, initializers, planned_dir)
    dump(scratch / 'planned-plan.json', {'configs': [items[identifier] for identifier in sorted(items)]})
    built = subprocess.run(C.worker_configs_command(sys.executable, HERE / 'query' / 'code', scratch / 'planned-plan.json', scratch / 'planned-receipt.json'), capture_output=True, text=True)
    receipt['planned_build'] = {'command': C.worker_configs_command(sys.executable, HERE / 'query' / 'code', scratch / 'planned-plan.json', scratch / 'planned-receipt.json'),
                                'returncode': built.returncode, 'stdout': built.stdout[-2000:], 'stderr': built.stderr[-3000:]}
    assert built.returncode == 0, built.stderr[-3000:]
    worker = {'protocol_path': str(worker_protocol_path), 'protocol_sha256': sha(worker_protocol_path),
              'source_hashes': {name: sha(HERE / 'query' / 'code' / name) for name in source_files},
              'reference_hashes': {name: sha(HERE / 'query' / 'references' / name) for name in reference_files}}
    planned = {}
    for identifier in sorted(items):
        config = json.loads((planned_dir / ('%s.json' % identifier)).read_text())
        S.check_planned(config, identifier, worker, bank_complete, protocol)
        planned[identifier] = config
    worker_receipt = json.loads((scratch / 'planned-receipt.json').read_text())
    receipt['planned_configs'] = {'count': len(planned), 'validated_by_worker': worker_receipt['passed'], 'stager_pin_checks': True,
                                  'checks_per_config': worker_receipt['configs'][0]['checks'], 'ids': sorted(planned)}
    parent_id = sorted(identifier for identifier in planned if identifier.startswith('S-'))[0]
    parent = scratch / 'mock-parent' / parent_id / 'training'
    (parent / 'checkpoints').mkdir(parents=True)
    (parent / 'checkpoints' / 'step-005000-1.pt').write_bytes(b'mock terminal state')
    dump(parent / 'progress.json', {'step': 5000, 'checkpoint': 'checkpoints/step-005000-1.pt', 'checkpoint_sha256': sha(parent / 'checkpoints' / 'step-005000-1.pt')})
    dump(parent / 'COMPLETE.json', {'steps': 5000, 'mock': True})
    continuation = C.continuation_item(scratch / 'continuation' / ('C-%s.json' % parent_id[2:]), planned[parent_id], parent, 'C-' + parent_id[2:], protocol['continuation_total_steps'])
    dump(scratch / 'continuation-plan.json', {'configs': [continuation]})
    built = subprocess.run(C.worker_configs_command(sys.executable, HERE / 'query' / 'code', scratch / 'continuation-plan.json', scratch / 'continuation-receipt.json'),
                           capture_output=True, text=True)
    receipt['continuation_build'] = {'returncode': built.returncode, 'stderr': built.stderr[-3000:]}
    assert built.returncode == 0, built.stderr[-3000:]
    built_config = json.loads(Path(continuation['output']).read_text())
    differences = C.continuation_differences(planned[parent_id], built_config)
    receipt['continuation_build'].update({'differs_from_parent_in': differences, 'only_continuation_keys': set(differences) <= set(C.CONTINUATION_KEYS),
                                          'note': 'parent COMPLETE/progress/checkpoint are mocks; validates config construction only'})
    receipt['selectors'] = selector_agreement(scratch, protocol_path, worker_protocol_path, Path(args.decision), planned)
    receipt['engine_gate'] = engine_gate(scratch, load('suite_advance', 'advance.py'), protocol, worker, planned['Q-ref'])
    policy = protocol['runtime']
    receipt['deadlines'] = {'grace_60min': controller.deadlines(policy, 10800, 3600), 'no_grace': controller.deadlines(policy, 10800, 0),
                            'grace_20min': controller.deadlines(policy, 10800, 1200), 'time_limit_parse': controller.seconds('03:00:00')}
    sizes = K.sizes(protocol, None)
    receipt['storage_peaks_bytes'] = {identifier[:1]: K.peak_bytes(protocol, sizes, planned[identifier]) for identifier in (sorted(planned)[0], parent_id)}
    receipt['storage_peaks_bytes']['C'] = K.peak_bytes(protocol, sizes, built_config)
    checks = {'compiled': all(receipt['compiled'].values()), 'planned_21': len(planned) == 21, 'continuation_only_budget_keys': receipt['continuation_build']['only_continuation_keys'],
              'selectors_agree': receipt['selectors'].get('passed') is True, 'engine_gate': receipt['engine_gate']['passed'],
              'deadlines': receipt['deadlines']['grace_60min'][:2] == (13200, 13500) and receipt['deadlines']['no_grace'][:2] == (10200, 10500)}
    receipt.update({'checks': checks, 'passed': all(checks.values()), 'python': sys.executable, 'scratch': str(scratch),
                    'scope': 'CPU only; no training, no GPU, no scheduler; synthetic trajectories test selection logic only'})
    dump(Path(args.output), receipt)
    print(json.dumps({'passed': receipt['passed'], 'checks': checks}, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
