"""Suite HPO selection from saved VALIDATION artifacts only (no TEST, no tensors replayed). Standard library only.

screen  verify all 18 screens at 5000 updates; per view rank each screen by its own selected checkpoint; continue the
        union of the top two raw and top two calibrated (deduplicated, 2-4, no fill) from the exact terminal state.
final   verify every continuation at 20000 updates; per view the first-ranked trajectory wins. Raw and calibrated
        winners are selected separately and reported separately.
recipe  freeze the common attribution recipe from a recorded explicit decision naming the leaderboard; the other view's
        winner is kept beside it, never discarded.

Per trajectory and view: own step-0 hits@1 and hits@10 floor; eligible checkpoints reach both; maximize hits@10, hits@1,
MRR, -MR; earliest step on exact ties (recomputed here and required to equal the worker's selection.json). Across
trajectories: hits@10 desc, hits@1 desc, MRR desc, MR asc, then trial id asc and nothing else (the order of the worker's
leaderboard.py, which advance.py runs beside this as a cross-check). Every selected, initial and terminal
checkpoint is reported in BOTH views at that same checkpoint; the calibrated-minus-raw difference is descriptive, and
no direction is presumed.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


VIEWS = ('raw', 'calibrated')
METRICS = ('hits@10', 'hits@1', 'mrr', 'mr')
DECISION_RULE = ['hits_at_10', 'hits_at_1', 'mrr', 'lower_mean_rank']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def micro(point, view):
    return point['validation']['real'][view]['answer_micro']


def both_views(point):
    values = {view: {name: micro(point, view)[name] for name in METRICS} for view in VIEWS}
    values['calibrated_minus_raw'] = {name: values['calibrated'][name] - values['raw'][name] for name in METRICS}
    values['step'] = point['step']
    return values


def own_selection(curve, view):
    floor = micro(curve[0], view)
    eligible = [point for point in curve if micro(point, view)['hits@1'] >= floor['hits@1'] and micro(point, view)['hits@10'] >= floor['hits@10']]
    assert eligible, 'no eligible checkpoint, not even step 0'

    def key(point):
        value = micro(point, view)
        return (value['hits@10'], value['hits@1'], value['mrr'], -value['mr'], -point['step'])
    return max(eligible, key=key), [point['step'] for point in eligible]


def verified_trajectory(entry, protocol, purpose, expected_steps, hash_scores=True):
    directory = Path(entry['training_dir'])
    assert sha(directory / 'COMPLETE.json') == entry['complete_sha256'], 'COMPLETE changed: %s' % directory
    complete = read(directory / 'COMPLETE.json')
    identity = read(directory / 'identity.json')
    config = identity['config']
    assert complete['identity_sha256'] == canonical(identity) and complete['steps'] == expected_steps, 'identity or steps differ: %s' % directory
    assert identity['protocol_sha256'] == protocol['worker_sha256'] == config['protocol_sha256'], 'trajectory under another worker protocol: %s' % directory
    assert config['purpose'] == purpose and config['phase'] == 'train' and config['experiment_id'] == entry['id'], 'purpose or id differs: %s' % directory
    assert all(config[key] == value for key, value in protocol['fixed_query_settings'].items()), 'fixed query setting differs: %s' % directory
    assert config.get('test_evaluation', False) is False, 'TEST flag set: %s' % directory
    for name in ('curve.json', 'selection.json', 'identity.json', 'progress.json'):
        assert sha(directory / name) == complete['files'][name], '%s differs from COMPLETE: %s' % (name, directory)
    curve = read(directory / 'curve.json')
    assert [point['step'] for point in curve] == list(range(0, expected_steps + 1, protocol['evaluation_every_updates'])), 'evaluation schedule differs: %s' % directory
    recorded = read(directory / 'selection.json')
    progress = read(directory / 'progress.json')
    assert progress['step'] == expected_steps and progress['pending_evaluation'] is None
    if hash_scores:
        for point in curve:
            assert sha(point['scores']) == point['scores_sha256'] and sha(point['fit_metadata']) == point['fit_metadata_sha256'], 'saved scores changed: %s' % point['scores']
    for point in curve:
        for view in VIEWS:
            value = point['validation']['real'][view]
            assert value['queries'] == protocol['clinical']['validation_cases'] and value['associations'] == protocol['clinical']['validation_associations']
    chosen = {}
    for view in VIEWS:
        best, eligible = own_selection(curve, view)
        assert recorded[view]['step'] == best['step'] and recorded[view]['eligible_steps'] == eligible, 'worker selection differs from the recomputed one: %s %s' % (directory, view)
        pointer = progress['retained_best'][view]
        assert pointer['step'] == best['step'] and sha(Path(pointer['root']) / pointer['path']) == pointer['sha256'], 'retained %s checkpoint missing or changed: %s' % (view, directory)
        chosen[view] = {'step': best['step'], 'metrics': {name: micro(best, view)[name] for name in METRICS}, 'same_checkpoint_both_views': both_views(best),
                        'pointer': pointer, 'scores': best['scores'], 'scores_sha256': best['scores_sha256']}
    initial = progress['initial_checkpoint']
    assert sha(Path(initial['root']) / initial['path']) == initial['sha256'], 'initial checkpoint missing or changed: %s' % directory
    return {'id': entry['id'], 'config': config, 'chosen': chosen, 'initial': {'pointer': initial, 'both_views': both_views(curve[0])},
            'terminal': {'step': curve[-1]['step'], 'both_views': both_views(curve[-1]), 'checkpoint': progress['checkpoint'], 'checkpoint_sha256': progress['checkpoint_sha256']},
            'training_dir': str(directory), 'complete_sha256': entry['complete_sha256']}


def metric_key(metrics):
    return (-metrics['hits@10'], -metrics['hits@1'], -metrics['mrr'], metrics['mr'])


def recipe(config):
    """Same shape as the worker's leaderboard.py recipe and the production protocol's production_recipe."""
    initializer = config['initializer']
    return {'initializer_recipe': {key: initializer[key] for key in ('pretrain_learning_rate', 'pretrain_regularization', 'initializer_update_budget')},
            'learning_rate': config['learning_rate'], 'temperature': config['temperature']}


def identities(config):
    return {key: config[key] for key in ('pretrained_sha256', 'data_complete_sha256', 'protocol_sha256', 'encoder', 'curriculum', 'mixture_id', 'seed', 'sampler_seed', 'weight_decay')}


def order(row, view):
    return metric_key(row['chosen'][view]['metrics']) + (row['id'],)


def board(rows, view):
    ordered = sorted(rows, key=lambda row: order(row, view))
    return [{'rank': number + 1, 'id': row['id'], 'step': row['chosen'][view]['step'], 'metrics': row['chosen'][view]['metrics'],
             'same_checkpoint_both_views': row['chosen'][view]['same_checkpoint_both_views'], 'recipe': recipe(row['config'])} for number, row in enumerate(ordered)]


def per_trajectory(rows):
    return {row['id']: {'raw_selected': row['chosen']['raw']['same_checkpoint_both_views'], 'calibrated_selected': row['chosen']['calibrated']['same_checkpoint_both_views'],
                        'initial': row['initial']['both_views'], 'terminal': row['terminal']['both_views'],
                        'raw_and_calibrated_select_same_step': row['chosen']['raw']['step'] == row['chosen']['calibrated']['step']} for row in rows}


def screen(protocol, index):
    stage = protocol['stages']['S_screen']
    rows = [verified_trajectory(entry, protocol, stage['purpose'], stage['steps']) for entry in index['entries']]
    assert len({row['id'] for row in rows}) == len(rows), 'trial ids must be unique'
    candidates = [(row['pretrain_learning_rate'], row['pretrain_regularization'], row['initializer_update_budget']) for row in protocol['initializer_candidates']]
    expected = set((a, b, c, lr, t) for (a, b, c), lr, t in itertools.product(candidates, protocol['query_learning_rates'], protocol['temperatures']))
    observed = [tuple(row['config']['initializer'][key] for key in ('pretrain_learning_rate', 'pretrain_regularization', 'initializer_update_budget'))
                + (row['config']['learning_rate'], row['config']['temperature']) for row in rows]
    assert len(observed) == len(set(observed)) and set(observed) == expected, 'screen grid incomplete or duplicated'
    assert len({row['config']['data_complete_sha256'] for row in rows}) == 1 and len({row['config']['pretrained_sha256'] for row in rows}) == len(candidates)
    boards = {view: board(rows, view) for view in VIEWS}
    promoted = sorted({row['id'] for view in VIEWS for row in boards[view][:2]})
    assert 2 <= len(promoted) <= 4
    by_id = {row['id']: row for row in rows}
    return {'leaderboards': boards, 'same_checkpoint_raw_calibrated': per_trajectory(rows), 'promoted': promoted, 'no_fill': True,
            'promoted_by_view': {view: [row['id'] for row in boards[view][:2]] for view in VIEWS},
            'continue': [{'id': identifier, 'continuation_parent': by_id[identifier]['training_dir'], 'parent_complete_sha256': by_id[identifier]['complete_sha256'],
                          'parent_checkpoint_sha256': by_id[identifier]['terminal']['checkpoint_sha256'], 'maximum_steps': protocol['continuation_total_steps']} for identifier in promoted],
            'not_promoted': sorted(set(by_id) - set(promoted)),
            'resume_source': 'exact terminal 5000-update full state; never a view-selected earlier checkpoint'}


def final(protocol, index):
    rows = [verified_trajectory(entry, protocol, protocol['stages']['C_continuation']['purpose'], protocol['continuation_total_steps']) for entry in index['entries']]
    assert 2 <= len(rows) <= 4
    boards = {view: board(rows, view) for view in VIEWS}
    winners = {}
    for view in VIEWS:
        top = boards[view][0]
        row = next(row for row in rows if row['id'] == top['id'])
        winners[view] = {'id': row['id'], 'recipe': recipe(row['config']), 'identities': identities(row['config']), 'step': row['chosen'][view]['step'], 'metrics': row['chosen'][view]['metrics'],
                         'same_checkpoint_both_views': row['chosen'][view]['same_checkpoint_both_views'], 'checkpoint': row['chosen'][view]['pointer'],
                         'scores': row['chosen'][view]['scores'], 'scores_sha256': row['chosen'][view]['scores_sha256'], 'training_dir': row['training_dir'],
                         'initial_checkpoint': row['initial']['pointer'], 'terminal_checkpoint_sha256': row['terminal']['checkpoint_sha256']}
    same_recipe = winners['raw']['recipe'] == winners['calibrated']['recipe']
    return {'leaderboards': boards, 'same_checkpoint_raw_calibrated': per_trajectory(rows), 'winners': winners,
            'separately_selected_comparison': {'same_trajectory': winners['raw']['id'] == winners['calibrated']['id'], 'same_recipe': same_recipe,
                                               'same_checkpoint': same_recipe and winners['raw']['step'] == winners['calibrated']['step'],
                                               'raw_winner_in_both_views': winners['raw']['same_checkpoint_both_views'],
                                               'calibrated_winner_in_both_views': winners['calibrated']['same_checkpoint_both_views'],
                                               'note': 'descriptive VALIDATION comparison of separately selected checkpoints; it does not show that calibration helps or hurts'}}


def common_recipe(protocol, final_decision, decision):
    assert decision['common_leaderboard'] in VIEWS, 'decision names no admitted leaderboard'
    assert decision['rule'] == DECISION_RULE == protocol['selection']['decision_rule_names'], 'decision rule differs from the frozen selection rule'
    assert decision.get('raw_calibration_comparison_required') is True, 'decision must require the raw/calibrated comparison'
    chosen = decision['common_leaderboard']
    other = 'raw' if chosen == 'calibrated' else 'calibrated'
    winners = final_decision['winners']
    production = dict({'common_leaderboard': chosen}, **winners[chosen]['recipe'])
    production.update({'source_trajectory': winners[chosen]['id'], 'source_selected_step': winners[chosen]['step']})
    return {'common_leaderboard': chosen, 'common_recipe': winners[chosen]['recipe'], 'production_recipe': production, 'common_winner': winners[chosen],
            'other_view_winner': winners[other], 'other_view_recipe_differs': winners[other]['recipe'] != winners[chosen]['recipe'],
            'same_checkpoint_raw_calibrated': final_decision['same_checkpoint_raw_calibrated'],
            'separately_selected_comparison': final_decision['separately_selected_comparison'],
            'production_checkpoint_policy': 'every production trajectory keeps its own raw-selected and calibrated-selected checkpoints and both views of each; the common recipe fixes only the training configuration',
            'second_recipe': 'if the other view recipe differs, retaining it for the 19 PFS fold/seed pairs is a separate user decision (PLAN.md raw/calibrated section); nothing is scheduled automatically',
            'calibration_benefit_presumed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['screen', 'final', 'recipe'], required=True)
    parser.add_argument('--protocol', required=True, help='campaign protocol')
    parser.add_argument('--worker-protocol', required=True, help='query worker protocol the trajectories were trained under')
    parser.add_argument('--index', required=True)
    parser.add_argument('--final', help='recipe: frozen final decision')
    parser.add_argument('--decision', help='recipe: recorded selection decision')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    assert protocol['status'] == 'FROZEN' and protocol['test_access'] is False
    protocol['sha256'] = sha(args.protocol)
    protocol['worker_sha256'] = sha(args.worker_protocol)
    output = Path(args.output)
    assert not output.exists()
    index = read(args.index)
    assert index['stage'] == args.stage and index['protocol_sha256'] == protocol['sha256']
    result = {'stage': args.stage, 'protocol_sha256': protocol['sha256'], 'worker_protocol_sha256': protocol['worker_sha256'], 'index_sha256': sha(args.index),
              'selector_sha256': sha(__file__), 'test_access': False}
    if args.stage in ('screen', 'final'):
        result.update(screen(protocol, index) if args.stage == 'screen' else final(protocol, index))
    else:
        assert index['final_sha256'] == sha(args.final) and index['decision_sha256'] == sha(args.decision)
        result.update({'final_sha256': sha(args.final), 'decision_sha256': sha(args.decision)})
        result.update(common_recipe(protocol, read(args.final), read(args.decision)))
    temporary = output.with_name('.' + output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.rename(output)
    print(json.dumps({key: result[key] for key in ('stage', 'promoted', 'common_leaderboard') if key in result}, indent=2))


if __name__ == '__main__':
    main()
