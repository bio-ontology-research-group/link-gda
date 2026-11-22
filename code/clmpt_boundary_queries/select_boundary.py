"""Boundary-sweep promotion and final winners from saved validation artifacts only (no TEST, no tensors replayed).

Every trajectory is verified by the frozen selector's own verified_trajectory (analysis/select_stage.py, unchanged),
which recomputes each view's own step-0 H1/H10 floor and lexicographic best checkpoint (H10, H1, MRR, -MR, earliest)
and requires it to equal the trajectory's selection.json. Ranking, ties and the union rule are the frozen
selector's metric_key and union_top2; this file only admits entries trained under two protocols: the boundary
protocol for new trajectories and the executed clmpt-axiom13-hpo protocol for the reused control.

screen: the 8 new 2500-update screens plus the control screen; frozen S5 rule (top2 raw union top2 calibrated,
        2-4, no fill); new winners continue from their exact terminal 2500 full state to 20000 updates, a promoted
        control reuses its completed 20000-update continuation.
final:  every promoted 20000-update trajectory; per view the frozen S6 winner key.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location('frozen_select_stage', str(HERE / 'analysis' / 'select_stage.py'))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
VIEWS = F.VIEWS


def loaded_protocols(protocol_path, control_protocol_path):
    protocol = json.loads(Path(protocol_path).read_text())
    assert protocol['status'] == 'FROZEN' and protocol['test_access'] is False
    protocol['sha256'] = F.sha(protocol_path)
    control = json.loads(Path(control_protocol_path).read_text())
    assert control['status'] == 'FROZEN'
    control['sha256'] = F.sha(control_protocol_path)
    assert control['sha256'] == protocol['control']['protocol_sha256']
    assert control['evaluation_every_updates'] == protocol['evaluation_every_updates']
    return protocol, {protocol['sha256']: protocol, control['sha256']: control}


def verified_rows(index, protocol, admitted, expected_steps):
    rows = []
    for entry in index['entries']:
        assert entry['role'] in ('new', 'control') and entry['protocol_sha256'] in admitted
        assert (entry['protocol_sha256'] == protocol['control']['protocol_sha256']) == (entry['role'] == 'control')
        row = F.verified_trajectory(entry, admitted[entry['protocol_sha256']], expected_steps)
        row['role'] = entry['role']
        rows.append(row)
    assert len({row['id'] for row in rows}) == len(rows)
    return rows


def check_screen_pool(rows, protocol):
    control = [row for row in rows if row['role'] == 'control']
    new = [row for row in rows if row['role'] == 'new']
    pins = protocol['control']
    assert len(control) == 1 and control[0]['id'] == pins['screen']['trial_id'] and control[0]['complete_sha256'] == pins['screen']['complete_sha256']
    assert control[0]['config']['pretrained_sha256'] == pins['pretrained_sha256']
    expected = sorted((row['pretrain_learning_rate'], row['initializer_update_budget']) for row in protocol['expected_initializers'])
    assert sorted((row['config']['pretrain_learning_rate'], row['config']['initializer_update_budget']) for row in new) == expected
    assert len({row['config']['pretrained_sha256'] for row in rows}) == len(rows)
    screen = protocol['stages']['S2_screen']
    for row in rows:
        config = row['config']
        assert config['taxonomy_mode'] == 'original' and config['maximum_steps'] == screen['steps'] and not config.get('continuation_parent')
        for key in pins['recipe_keys_equal_to_new_screens']:
            assert config[key] == control[0]['config'][key], key
        assert (config['learning_rate'], config['temperature'], config['mixture_id']) == (screen['learning_rate'], screen['temperature'], screen['mixture_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['screen', 'final'], required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--control-protocol', required=True, help='frozen protocol under which the control trajectories were trained')
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    protocol, admitted = loaded_protocols(args.protocol, args.control_protocol)
    index = json.loads(Path(args.index).read_text())
    assert index['stage'] == args.stage and index['protocol_sha256'] == protocol['sha256'] and index['complete_planned_grid'] is True
    output = Path(args.output)
    assert not output.exists()
    result = {'stage': args.stage, 'protocol_sha256': protocol['sha256'], 'admitted_protocol_sha256': sorted(admitted), 'index_sha256': F.sha(args.index),
              'selector_sha256': F.sha(_spec.origin), 'wrapper_sha256': F.sha(__file__), 'test_access': False}
    if args.stage == 'screen':
        rows = verified_rows(index, protocol, admitted, protocol['continuation_parent_steps'])
        check_screen_pool(rows, protocol)
        boards = {view: sorted(rows, key=lambda row: F.metric_key(row['chosen'][view]['metrics']) + (row['config']['initializer_update_budget'], row['id'])) for view in VIEWS}
        chosen = F.union_top2(boards)
        assert 2 <= len(chosen) <= 4
        result.update({'leaderboards': {view: [{'id': row['id'], 'role': row['role'], 'step': row['chosen'][view]['step'], 'metrics': row['chosen'][view]['metrics'],
                                                'recipe': F.recipe(row['config']), 'pretrained_sha256': row['config']['pretrained_sha256']} for row in boards[view]] for view in VIEWS},
                       'promoted': chosen, 'no_fill': True,
                       'continue': [{'id': row['id'], 'continuation_parent': row['training_dir'], 'parent_complete_sha256': row['complete_sha256'],
                                     'parent_checkpoint_sha256': row['terminal_checkpoint_sha256'], 'maximum_steps': protocol['continuation_total_steps']}
                                    for row in rows if row['id'] in chosen and row['role'] == 'new'],
                       'reuse_control_long': protocol['control']['long'] if protocol['control']['screen']['trial_id'] in chosen else None,
                       'resume_source': 'exact terminal 2500 full state; never a view-selected earlier checkpoint'})
    else:
        rows = verified_rows(index, protocol, admitted, protocol['continuation_total_steps'])
        for row in rows:
            if row['role'] == 'control':
                assert row['id'] == protocol['control']['long']['trial_id'] and row['complete_sha256'] == protocol['control']['long']['complete_sha256']
        winners, boards = {}, {}
        for view in VIEWS:
            ordered = sorted(rows, key=lambda row: F.metric_key(row['chosen'][view]['metrics']) + (row['config']['initializer_update_budget'], row['id']))
            boards[view] = [{'id': row['id'], 'role': row['role'], 'step': row['chosen'][view]['step'], 'metrics': row['chosen'][view]['metrics'],
                             'recipe': F.recipe(row['config']), 'pointer': row['chosen'][view]['pointer']} for row in ordered]
            winners[view] = boards[view][0]
        result.update({'leaderboards': boards, 'winners': winners})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key in ('stage', 'promoted', 'continue', 'reuse_control_long', 'winners')}, indent=2))


if __name__ == '__main__':
    main()
