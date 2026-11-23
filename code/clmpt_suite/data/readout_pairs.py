"""Same-checkpoint raw versus calibrated validation readouts and the two leaderboards.

For every trajectory and every evaluated step, the raw and calibrated metrics of the SAME saved scores are paired.
Each trajectory's raw-selected and calibrated-selected checkpoints are recomputed with the frozen selector, and a
two-by-two table crosses the selection view with the scoring view. Two leaderboards are built from the selected
checkpoints: the calibrated one, which the recorded user decision uses to freeze the common attribution recipe, and
the raw one, kept separately. Differences are descriptive; nothing here presumes that calibration helps.

With --recompute, every step's metrics are recomputed from fits/fit-step-*.npz with the pinned metrics module and
must equal curve.json to 1e-12.
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import Refusal, check, read, sha, write_json

METRICS_SHA256 = 'be7c1dead8c5890c49cf8b9c95bed6cf0e87a69f2cf7e10718f4247343aaa164'
KEYS = ('hits@1', 'hits@10', 'mrr', 'mr')
RULE_KEYS = {'hits_at_10': ('hits@10', 1), 'hits_at_1': ('hits@1', 1), 'mrr': ('mrr', 1), 'lower_mean_rank': ('mr', -1)}
VIEWS = ('raw', 'calibrated')


def load_metrics(path):
    check(sha(path) == METRICS_SHA256, 'metrics module differs from the executed be7c1dea')
    spec = importlib.util.spec_from_file_location('frozen_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def micro(row, view):
    value = row['validation']['real'][view]['answer_micro']
    return {key: float(value[key]) for key in KEYS}


def rank_key(metrics, rule):
    out = []
    for name in rule:
        check(name in RULE_KEYS, 'unknown leaderboard rule term %s' % name)
        key, sign = RULE_KEYS[name]
        out.append(sign * metrics[key])
    return tuple(out)


def recompute(directory, curve, bank, metrics_module, tolerance):
    answers = [row['gene_ids'] for row in bank['validation']]
    worst = 0.
    for row in curve:
        archive = directory / 'fits' / ('fit-step-%06d.npz' % row['step'])
        check(archive.is_file(), 'missing saved scores %s' % archive)
        data = np.load(archive)
        check(data['candidate_ids'].tolist() == bank['candidate_ids'], 'candidate order differs at step %d' % row['step'])
        real = data['validation_real']
        for view, scores in (('raw', real), ('calibrated', metrics_module.calibrate(real))):
            value = metrics_module.measure(scores, answers, bank['candidate_ids'])['answer_micro']
            for key in KEYS:
                difference = abs(value[key] - row['validation']['real'][view]['answer_micro'][key])
                worst = max(worst, difference)
                check(difference <= tolerance, 'step %d %s %s differs by %g' % (row['step'], view, key, difference))
    return worst


def trajectory(directory, metrics_module, bank, tolerance):
    directory = pathlib.Path(directory)
    curve = sorted(read(directory / 'curve.json'), key=lambda row: row['step'])
    check(curve and curve[0]['step'] == 0, '%s curve lacks step 0' % directory)
    names = {row['disease_name'] for row in bank['validation']} if bank is not None else None
    if names is not None:
        check(len(names) == len(bank['validation']), 'validation cases are not distinct diseases; LOO calibration identity fails')
    selections = {view: metrics_module.selector(curve, view) for view in VIEWS}
    saved = directory / 'selection.json'
    if saved.is_file():
        recorded = read(saved)
        for view in VIEWS:
            check(recorded[view]['step'] == selections[view]['step'], '%s %s selection differs from selection.json' % (directory, view))
    worst = recompute(directory, curve, bank, metrics_module, tolerance) if bank is not None else None
    by_step = {row['step']: row for row in curve}
    paired = []
    for row in curve:
        raw, cal = micro(row, 'raw'), micro(row, 'calibrated')
        paired.append({'step': row['step'], 'raw': raw, 'calibrated': cal, 'calibrated_minus_raw': {k: cal[k] - raw[k] for k in KEYS}})
    better = {k: sum((p['calibrated_minus_raw'][k] < 0) if k == 'mr' else (p['calibrated_minus_raw'][k] > 0) for p in paired) for k in KEYS}
    worse = {k: sum((p['calibrated_minus_raw'][k] > 0) if k == 'mr' else (p['calibrated_minus_raw'][k] < 0) for p in paired) for k in KEYS}
    table = {'selected_by_%s' % sel: {'step': selections[sel]['step'], **{'scored_%s' % view: micro(by_step[selections[sel]['step']], view) for view in VIEWS}} for sel in VIEWS}
    label = directory.parent.name if directory.name == 'training' else directory.name
    return {'trajectory': label, 'path': str(directory), 'steps': len(curve), 'curve_sha256': sha(directory / 'curve.json'),
            'selections': {view: {'step': selections[view]['step'], 'eligible_steps': selections[view]['eligible_steps']} for view in VIEWS},
            'two_by_two': table, 'same_checkpoint': paired,
            'same_checkpoint_counts': {'steps': len(paired), 'calibrated_better': better, 'calibrated_worse': worse, 'ties': {k: len(paired) - better[k] - worse[k] for k in KEYS}},
            'recompute_max_abs_difference': worst}


def leaderboard(results, view, rule):
    rows = []
    for result in results:
        metrics = result['two_by_two']['selected_by_%s' % view]['scored_%s' % view]
        rows.append({'trajectory': result['trajectory'], 'step': result['selections'][view]['step'], 'metrics': metrics, 'key': rank_key(metrics, rule)})
    rows.sort(key=lambda row: (tuple(-v for v in row['key']), row['trajectory']))
    for position, row in enumerate(rows):
        row['position'] = position + 1
        row['tied_with_next'] = position + 1 < len(rows) and rows[position + 1]['key'] == row['key']
        row['key'] = list(row['key'])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trajectory', action='append', required=True, help='training directory with curve.json (repeatable)')
    parser.add_argument('--decision', required=True, help='SELECTION-DECISION.json')
    parser.add_argument('--metrics', required=True, help='frozen query/code/metrics.py')
    parser.add_argument('--bank', help='bank.json for --recompute')
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--tolerance', type=float, default=1e-12)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        started = time.time()
        decision = read(args.decision)
        check(decision['common_leaderboard'] in VIEWS, 'decision names no leaderboard')
        check(decision.get('raw_calibration_comparison_required') is True, 'decision does not require the paired comparison')
        metrics_module = load_metrics(args.metrics)
        bank = read(args.bank) if args.recompute else None
        check(not args.recompute or args.bank, '--recompute needs --bank')
        results = [trajectory(path, metrics_module, bank, args.tolerance) for path in args.trajectory]
        check(len({r['trajectory'] for r in results}) == len(results), 'duplicate trajectory names')
        boards = {view: leaderboard(results, view, decision['rule']) for view in VIEWS}
        common = decision['common_leaderboard']
        other = 'raw' if common == 'calibrated' else 'calibrated'
        winners = {view: boards[view][0]['trajectory'] if boards[view] else None for view in VIEWS}
        value = {'decision': {'path': str(pathlib.Path(args.decision).resolve()), 'sha256': sha(args.decision), **decision},
                 'metrics_sha256': METRICS_SHA256, 'trajectories': results, 'leaderboards': boards,
                 'common_recipe_candidate': {'leaderboard': common, 'trajectory': winners[common],
                                             'step': boards[common][0]['step'] if boards[common] else None,
                                             'tied_at_top': bool(boards[common]) and boards[common][0]['tied_with_next']},
                 'separate_winner': {'leaderboard': other, 'trajectory': winners[other]},
                 'winners_differ': winners['raw'] != winners['calibrated'],
                 'interpretation': 'Descriptive validation comparison. Calibrated-minus-raw differences at identical checkpoints and separately selected checkpoints are reported without presuming that calibration helps. No TEST data are read.',
                 'recomputed_from_saved_scores': bool(args.recompute), 'elapsed_seconds': time.time() - started}
        write_json(pathlib.Path(args.output), value)
        print(json.dumps({'common': value['common_recipe_candidate'], 'separate': value['separate_winner'], 'winners_differ': value['winners_differ'],
                          'trajectories': {r['trajectory']: {'two_by_two': r['two_by_two'], 'same_checkpoint_counts': r['same_checkpoint_counts']} for r in results},
                          'leaderboards': {view: [[row['position'], row['trajectory'], row['step']] for row in board] for view, board in boards.items()}}, indent=1))
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}), flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
