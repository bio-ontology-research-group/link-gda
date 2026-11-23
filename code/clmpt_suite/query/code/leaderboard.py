"""Validation-only HPO promotion and common-recipe freeze for the suite (no TEST, no tensors replayed).

promote  all complete 5000-update screens of the HPO grid -> separate raw and calibrated leaderboards ->
         deduplicated union of each top two (2-4 configurations, no fill) to continue to 20000 updates.
freeze   all promoted 20000-update continuations -> separately selected raw and calibrated winners; the
         common attribution recipe is the winner of the leaderboard named by an explicit decision file
         (common_leaderboard 'raw' or 'calibrated', rule H10, H1, MRR, lower MR). Without that file it refuses.

Both commands recompute every per-trajectory selection from curve.json (own step-0 H1/H10 floor; H10, H1,
MRR, -MR; earliest step) and require it to equal selection.json and the retained model-only pointers. Both
also report, for every trajectory, raw and calibrated metrics of the SAME checkpoint at its raw-selected,
calibrated-selected and terminal steps, and signed calibrated-minus-raw differences. The differences are
descriptive; nothing here assumes that calibration improves ranking.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


VIEWS = ('raw', 'calibrated')
METRICS = ('hits@10', 'hits@1', 'mrr', 'mr')
RULE = ['hits_at_10', 'hits_at_1', 'mrr', 'lower_mean_rank']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def metric_key(metrics):
    return (-metrics['hits@10'], -metrics['hits@1'], -metrics['mrr'], metrics['mr'])


def recipe(config):
    initializer = config['initializer']
    return {'initializer_recipe': {key: initializer[key] for key in ('pretrain_learning_rate', 'pretrain_regularization', 'initializer_update_budget')},
            'learning_rate': config['learning_rate'], 'temperature': config['temperature']}


def same_checkpoint(point):
    values = {view: point['validation']['real'][view]['answer_micro'] for view in VIEWS}
    return {'step': point['step'], 'raw': {key: values['raw'][key] for key in METRICS}, 'calibrated': {key: values['calibrated'][key] for key in METRICS},
            'calibrated_minus_raw': {key: values['calibrated'][key] - values['raw'][key] for key in METRICS}}


def verified_trajectory(entry, protocol, protocol_sha, purpose, expected_steps):
    directory = Path(entry['training_dir'])
    assert sha(directory / 'COMPLETE.json') == entry['complete_sha256']
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    identity = json.loads((directory / 'identity.json').read_text())
    config = identity['config']
    assert complete['identity_sha256'] == canonical(identity) and complete['steps'] == expected_steps
    assert identity['protocol_sha256'] == protocol_sha and config['purpose'] == purpose and config['phase'] == 'train'
    for name in ('curve.json', 'selection.json', 'identity.json', 'progress.json'):
        assert sha(directory / name) == complete['files'][name]
    spec = protocol['admitted_banks'][config['data_complete_sha256']]
    curve = json.loads((directory / 'curve.json').read_text())
    assert [point['step'] for point in curve] == list(range(0, expected_steps + 1, config['evaluation_every_updates']))
    for point in curve:
        assert sha(point['scores']) == point['scores_sha256'] and sha(point['fit_metadata']) == point['fit_metadata_sha256']
        for view in VIEWS:
            value = point['validation']['real'][view]
            assert value['queries'] == spec['validation_cases'] and value['associations'] == spec['validation_associations']
    recorded = json.loads((directory / 'selection.json').read_text())
    progress = json.loads((directory / 'progress.json').read_text())
    assert progress['step'] == expected_steps and progress['pending_evaluation'] is None
    by_step = {point['step']: point for point in curve}
    chosen = {}
    for view in VIEWS:
        floor = curve[0]['validation']['real'][view]['answer_micro']
        eligible = [point for point in curve if point['validation']['real'][view]['answer_micro']['hits@1'] >= floor['hits@1']
                    and point['validation']['real'][view]['answer_micro']['hits@10'] >= floor['hits@10']]
        best = max(eligible, key=lambda point: tuple(point['validation']['real'][view]['answer_micro'][key] * sign for key, sign in zip(METRICS, (1, 1, 1, -1))) + (-point['step'],))
        assert recorded[view]['step'] == best['step'] and recorded[view]['eligible_steps'] == [point['step'] for point in eligible]
        pointer = progress['retained_best'][view]
        assert pointer['step'] == best['step'] and sha(Path(pointer['root']) / pointer['path']) == pointer['sha256']
        chosen[view] = {'step': best['step'], 'metrics': {key: best['validation']['real'][view]['answer_micro'][key] for key in METRICS}, 'pointer': pointer,
                        'scores': best['scores'], 'scores_sha256': best['scores_sha256']}
    comparison = {'at_raw_selected': same_checkpoint(by_step[chosen['raw']['step']]), 'at_calibrated_selected': same_checkpoint(by_step[chosen['calibrated']['step']]),
                  'at_terminal': same_checkpoint(by_step[expected_steps]), 'at_initial': same_checkpoint(by_step[0])}
    return {'id': entry['id'], 'config': config, 'recipe': recipe(config), 'chosen': chosen, 'same_checkpoint': comparison,
            'separately_selected_calibrated_minus_raw': {key: chosen['calibrated']['metrics'][key] - chosen['raw']['metrics'][key] for key in METRICS},
            'terminal_checkpoint': progress['checkpoint'], 'terminal_checkpoint_sha256': progress['checkpoint_sha256'],
            'training_dir': str(directory), 'complete_sha256': entry['complete_sha256']}


def leaderboards(rows):
    """Per view: H10 desc, H1 desc, MRR desc, MR asc, then the stable trial id ascending (agreed root rule)."""
    assert len({row['id'] for row in rows}) == len(rows), 'trial ids must be unique'
    order = lambda view: lambda row: metric_key(row['chosen'][view]['metrics']) + (row['id'],)
    return {view: sorted(rows, key=order(view)) for view in VIEWS}


def board_view(boards):
    return {view: [{'rank': number + 1, 'id': row['id'], 'step': row['chosen'][view]['step'], 'metrics': row['chosen'][view]['metrics'], 'recipe': row['recipe']}
                   for number, row in enumerate(boards[view])] for view in VIEWS}


def direction_counts(rows):
    """Descriptive tally of same-checkpoint calibrated-minus-raw signs; lower MR is better."""
    result = {}
    for where in ('at_raw_selected', 'at_calibrated_selected', 'at_terminal', 'at_initial'):
        tally = {}
        for key in METRICS:
            better = Counter()
            for row in rows:
                delta = row['same_checkpoint'][where]['calibrated_minus_raw'][key]
                delta = -delta if key == 'mr' else delta
                better['calibrated_better' if delta > 0 else 'raw_better' if delta < 0 else 'equal'] += 1
            tally[key] = dict(better)
        result[where] = tally
    return result


def comparison_table(rows):
    return [{'id': row['id'], 'recipe': row['recipe'], 'raw_selected_step': row['chosen']['raw']['step'], 'calibrated_selected_step': row['chosen']['calibrated']['step'],
             'same_checkpoint': row['same_checkpoint'], 'separately_selected_calibrated_minus_raw': row['separately_selected_calibrated_minus_raw']} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['promote', 'freeze'])
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--decision')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    assert protocol['status'] == 'FROZEN' and protocol['test_access'] is False
    protocol_sha = sha(args.protocol)
    index = json.loads(Path(args.index).read_text())
    output = Path(args.output)
    assert not output.exists()
    result = {'command': args.command, 'protocol_sha256': protocol_sha, 'index_sha256': sha(args.index), 'selector_sha256': sha(__file__), 'test_access': False}
    if args.command == 'promote':
        rules = protocol['purposes']['hpo_screen']
        rows = [verified_trajectory(entry, protocol, protocol_sha, 'hpo_screen', rules['maximum_steps'][0]) for entry in index['entries']]
        expected = len(protocol['hpo_initializer_recipes']) * len(protocol['query_learning_rates']) * len(protocol['temperatures'])
        keys = sorted(json.dumps(row['recipe'], sort_keys=True) for row in rows)
        assert len(rows) == expected == len(set(keys)), 'every HPO screen must be complete and distinct before promotion'
        assert len({row['config']['data_complete_sha256'] for row in rows}) == 1
        boards = leaderboards(rows)
        chosen = sorted({row['id'] for view in VIEWS for row in boards[view][:2]})
        assert 2 <= len(chosen) <= 4
        result.update({'leaderboards': board_view(boards), 'promoted_ids': chosen, 'no_fill': True,
                       'continue': [{'id': row['id'], 'continuation_parent': row['training_dir'], 'parent_complete_sha256': row['complete_sha256'],
                                     'parent_checkpoint_sha256': row['terminal_checkpoint_sha256'], 'maximum_steps': protocol['purposes']['hpo_continuation']['maximum_steps'][0]}
                                    for row in rows if row['id'] in chosen],
                       'raw_calibrated_comparison': comparison_table(rows), 'same_checkpoint_direction_counts': direction_counts(rows),
                       'prunable_after_continuations_verified': [{'id': row['id'], 'full_state': row['terminal_checkpoint'], 'sha256': row['terminal_checkpoint_sha256']}
                                                                 for row in rows if row['id'] not in chosen],
                       'resume_source': 'exact terminal 5000 full state; never a view-selected earlier checkpoint'})
    else:
        assert args.decision, 'freeze requires an explicit decision file'
        decision = json.loads(Path(args.decision).read_text())
        assert decision['common_leaderboard'] in VIEWS and decision['rule'] == RULE and decision['raw_calibration_comparison_required'] is True
        rules = protocol['purposes']['hpo_continuation']
        rows = [verified_trajectory(entry, protocol, protocol_sha, 'hpo_continuation', rules['maximum_steps'][0]) for entry in index['entries']]
        assert 2 <= len(rows) <= 4 and index.get('promotion_sha256')
        boards = leaderboards(rows)
        winners = {view: {'id': boards[view][0]['id'], 'recipe': boards[view][0]['recipe'], 'step': boards[view][0]['chosen'][view]['step'],
                          'metrics': boards[view][0]['chosen'][view]['metrics'], 'same_checkpoint': boards[view][0]['same_checkpoint'],
                          'pointer': boards[view][0]['chosen'][view]['pointer']} for view in VIEWS}
        common = winners[decision['common_leaderboard']]
        result.update({'decision': decision, 'decision_sha256': sha(args.decision), 'promotion_sha256': index['promotion_sha256'],
                       'leaderboards': board_view(boards), 'separately_selected_winners': winners,
                       'winners_agree': winners['raw']['recipe'] == winners['calibrated']['recipe'],
                       'raw_calibrated_comparison': comparison_table(rows), 'same_checkpoint_direction_counts': direction_counts(rows),
                       'production_recipe': {'common_leaderboard': decision['common_leaderboard'], **common['recipe'],
                                             'source_trajectory': common['id'], 'source_selected_step': common['step']},
                       'note': 'The production recipe is the common attribution recipe for controls and biological arms. The other view\'s winner is retained as a '
                               'separately selected result; when winners differ, running it across the suite needs separate approval. Copy production_recipe '
                               'into a NEW frozen query protocol (never edit v1) together with the admitted production banks.'})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result.get(key) for key in ('command', 'promoted_ids', 'winners_agree', 'production_recipe')}, indent=2))


if __name__ == '__main__':
    main()
