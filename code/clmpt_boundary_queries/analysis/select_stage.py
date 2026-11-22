"""Frozen-rule promotion between stages from saved validation artifacts only (no TEST, no tensors replayed).

S3: initializer shortlist from the 54 screen trajectories of one mode.
S5: configurations to continue to the longer TOTAL budget from the refinement pool of one mode.
S6: distinct final winning recipes of both modes and the matched 2x2 cell plan with exact-reuse marks.
Per trajectory, each view keeps its own step-0 H1/H10 floor and selects the lexicographic best eligible
scheduled checkpoint (H10, H1, MRR, -MR, earliest step), recomputed here and required to equal selection.json.
"""
import argparse
import hashlib
import json
from pathlib import Path


VIEWS = ('raw', 'calibrated')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def verified_trajectory(entry, protocol, expected_steps):
    directory = Path(entry['training_dir'])
    assert sha(directory / 'COMPLETE.json') == entry['complete_sha256']
    complete = json.loads((directory / 'COMPLETE.json').read_text())
    identity = json.loads((directory / 'identity.json').read_text())
    config = identity['config']
    identity_sha = hashlib.sha256((json.dumps(identity, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
    assert complete['identity_sha256'] == identity_sha and complete['steps'] == expected_steps
    assert identity['protocol_sha256'] == protocol['sha256'] and config['seed'] == 0
    for name in ('curve.json', 'selection.json', 'identity.json', 'progress.json'):
        assert sha(directory / name) == complete['files'][name]
    curve = json.loads((directory / 'curve.json').read_text())
    assert [point['step'] for point in curve] == list(range(0, expected_steps + 1, protocol['evaluation_every_updates']))
    recorded = json.loads((directory / 'selection.json').read_text())
    progress = json.loads((directory / 'progress.json').read_text())
    assert progress['step'] == expected_steps and progress['pending_evaluation'] is None
    chosen = {}
    for view in VIEWS:
        floor = curve[0]['validation']['real'][view]['answer_micro']
        for point in curve:
            assert sha(point['scores']) == point['scores_sha256'] and sha(point['fit_metadata']) == point['fit_metadata_sha256']
        eligible = [point for point in curve if point['validation']['real'][view]['answer_micro']['hits@1'] >= floor['hits@1'] and point['validation']['real'][view]['answer_micro']['hits@10'] >= floor['hits@10']]

        def key(point):
            value = point['validation']['real'][view]['answer_micro']
            return (value['hits@10'], value['hits@1'], value['mrr'], -value['mr'], -point['step'])
        best = max(eligible, key=key)
        assert recorded[view]['step'] == best['step'] and recorded[view]['eligible_steps'] == [point['step'] for point in eligible]
        pointer = progress['retained_best'][view]
        assert pointer['step'] == best['step'] and sha(Path(pointer['root']) / pointer['path']) == pointer['sha256']
        value = best['validation']['real'][view]
        assert value['queries'] == 545 and value['associations'] == 586
        chosen[view] = {'step': best['step'], 'metrics': value['answer_micro'], 'pointer': pointer}
    return {'id': entry['id'], 'config': config, 'chosen': chosen, 'terminal_checkpoint': progress['checkpoint'], 'terminal_checkpoint_sha256': progress['checkpoint_sha256'],
            'training_dir': str(directory), 'complete_sha256': entry['complete_sha256']}


def metric_key(metrics):
    return (-metrics['hits@10'], -metrics['hits@1'], -metrics['mrr'], metrics['mr'])


def recipe(config):
    return {'taxonomy_mode': config['taxonomy_mode'], 'pretrain_learning_rate': config['pretrain_learning_rate'], 'pretrain_regularization': config['pretrain_regularization'],
            'initializer_update_budget': config['initializer_update_budget'], 'learning_rate': config['learning_rate'], 'temperature': config['temperature']}


def union_top2(boards):
    chosen = sorted({row['id'] for view in VIEWS for row in boards[view][:2]})
    assert 2 <= len(chosen) <= 4 or (len(chosen) == 1 and all(len(boards[view]) == 1 for view in VIEWS))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['S3', 'S5', 'S6'], required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    assert protocol['status'] == 'FROZEN'
    protocol['sha256'] = sha(args.protocol)
    index = json.loads(Path(args.index).read_text())
    output = Path(args.output)
    assert not output.exists()
    result = {'stage': args.stage, 'protocol_sha256': protocol['sha256'], 'index_sha256': sha(args.index), 'selector_sha256': sha(__file__), 'test_access': False}
    if args.stage == 'S3':
        rows = [verified_trajectory(entry, protocol, 2500) for entry in index['entries']]
        assert len(rows) == 54 and len({row['config']['taxonomy_mode'] for row in rows}) == 1
        aggregation = protocol['stages']['S3_initializer_shortlist']['aggregation']
        assert aggregation in ('best_of', 'mean')
        boards = {}
        for view in VIEWS:
            grouped = {}
            for row in rows:
                grouped.setdefault(row['config']['pretrained_sha256'], []).append(row)
            board = []
            for initializer, members in grouped.items():
                assert len(members) == 2
                if aggregation == 'best_of':
                    metrics = min((member['chosen'][view]['metrics'] for member in members), key=metric_key)
                else:
                    metrics = {name: sum(member['chosen'][view]['metrics'][name] for member in members) / 2 for name in ('hits@10', 'hits@1', 'mrr', 'mr')}
                config = members[0]['config']
                board.append({'id': initializer, 'metrics': metrics, 'initializer_update_budget': config['initializer_update_budget'],
                              'pretrain_learning_rate': config['pretrain_learning_rate'], 'pretrain_regularization': config['pretrain_regularization'], 'members': [member['id'] for member in members]})
            board.sort(key=lambda row: metric_key(row['metrics']) + (row['initializer_update_budget'], row['pretrain_learning_rate'], row['pretrain_regularization']))
            boards[view] = board
        result.update({'aggregation': aggregation, 'leaderboards': boards, 'shortlisted_initializer_sha256': union_top2(boards), 'no_fill': True})
    elif args.stage == 'S5':
        rows = [verified_trajectory(entry, protocol, 2500) for entry in index['entries']]
        assert len({row['config']['taxonomy_mode'] for row in rows}) == 1
        boards = {view: sorted(rows, key=lambda row: metric_key(row['chosen'][view]['metrics']) + (row['config']['initializer_update_budget'], row['id'])) for view in VIEWS}
        chosen = union_top2(boards)
        result.update({'leaderboards': {view: [{'id': row['id'], 'step': row['chosen'][view]['step'], 'metrics': row['chosen'][view]['metrics'], 'recipe': recipe(row['config'])} for row in boards[view]] for view in VIEWS},
                       'continue': [{'id': row['id'], 'continuation_parent': row['training_dir'], 'parent_complete_sha256': row['complete_sha256'], 'parent_checkpoint_sha256': row['terminal_checkpoint_sha256'],
                                     'maximum_steps': protocol['continuation_total_steps']} for row in rows if row['id'] in chosen],
                       'resume_source': 'exact terminal 2500 full state; never a mode-selected earlier checkpoint'})
    else:
        rows = [verified_trajectory(entry, protocol, protocol['continuation_total_steps']) for entry in index['entries']]
        winners = {}
        for mode in protocol['taxonomy_modes']:
            members = [row for row in rows if row['config']['taxonomy_mode'] == mode]
            assert members
            for view in VIEWS:
                best = min(members, key=lambda row: metric_key(row['chosen'][view]['metrics']) + (row['config']['initializer_update_budget'], row['id']))
                winners['%s/%s' % (mode, view)] = {'id': best['id'], 'recipe': recipe(best['config']), 'step': best['chosen'][view]['step'], 'metrics': best['chosen'][view]['metrics']}
        distinct = []
        for winner in winners.values():
            shared = {key: value for key, value in winner['recipe'].items() if key != 'taxonomy_mode'}
            if shared not in distinct:
                distinct.append(shared)
        plan = []
        for number, shared in enumerate(distinct):
            for mode in protocol['taxonomy_modes']:
                for curriculum, mixture in (('three_pattern_subset', protocol['subset_mixture_id']), ('full13', protocol['expanded_mixture_id'])):
                    reusable = [row['id'] for row in rows if row['config']['taxonomy_mode'] == mode and row['config']['mixture_id'] == mixture
                                and {key: value for key, value in recipe(row['config']).items() if key != 'taxonomy_mode'} == shared]
                    plan.append({'recipe_index': number, 'recipe': shared, 'mode': mode, 'curriculum': curriculum, 'mixture_id': mixture,
                                 'reuse_existing_trajectory': reusable[0] if reusable else None, 'new_run_steps': None if reusable else protocol['continuation_total_steps']})
        result.update({'winners': winners, 'distinct_recipes': distinct, 'matched_cells': plan, 'new_cells': sum(cell['reuse_existing_trajectory'] is None for cell in plan)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key in ('stage', 'shortlisted_initializer_sha256', 'continue', 'new_cells')}, indent=2))


if __name__ == '__main__':
    main()
