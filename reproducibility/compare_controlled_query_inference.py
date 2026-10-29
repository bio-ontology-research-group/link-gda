"""Compare all 480 regenerated query bundles with the frozen production collection."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from analysis.summarize_controlled_query import (
    ARMS, INITS, SEEDS, checked_file, load_bundle, verify_plan,
)
from analysis.summarize_query_tables import metadata
from phenoquery.common import metrics, sha, write

RTOL = 1e-5
ATOL = 1e-6
DATA_FILES = ('training.tsv', 'validation.tsv', 'test.tsv', 'profiles.json', 'candidates.txt')
TRAINING_EVIDENCE = ('training_completion_sha256', 'training_identity_sha256',
                     'validation_trajectory_sha256', 'validation_scores_sha256')


def compare(reference_root, replay_root, expected_candidates=13625):
    reference_root, replay_root = Path(reference_root).resolve(), Path(replay_root).resolve()
    if reference_root == replay_root:
        raise ValueError('Reference and replay must be separate collections')
    reference_plan, reference_global = verify_plan(reference_root)
    replay_plan, replay_global = verify_plan(replay_root)
    if reference_plan != replay_plan or reference_global != replay_global:
        raise ValueError('Require the same frozen global selection and controlled design')
    records, inputs, previous_diseases = [], {}, set()
    for fold in range(10):
        canonical = reference_root/reference_plan['campaigns'][0]/'data'/f'fold{fold}'
        genes, rows, seen, _, _ = metadata(canonical)
        diseases = sorted({d for _, d in rows})
        if len(genes) != expected_candidates or previous_diseases.intersection(diseases):
            raise ValueError('Candidate count or outer test-fold identity mismatch')
        previous_diseases.update(diseases)
        data_hashes = {name: sha(canonical/name) for name in DATA_FILES}
        inputs[str(fold)] = data_hashes
        for init in INITS:
            for seed in SEEDS:
                campaign = f'{init}-seed{seed}'
                children = [root/campaign for root in (reference_root, replay_root)]
                for child in children:
                    for name, digest in data_hashes.items():
                        checked_file(child/'data'/f'fold{fold}'/name, digest)
                for arm in ARMS:
                    initial_and_selected = {}
                    for kind in ('selected', 'initial'):
                        loaded = [load_bundle(child, fold, arm, kind, init, seed,
                            genes, diseases, rows, seen, data_hashes) for child in children]
                        initial_and_selected[kind] = loaded
                        if loaded[0][2] != loaded[1][2] or any(
                                loaded[0][1][name] != loaded[1][1][name] for name in TRAINING_EVIDENCE):
                            raise ValueError('Training or validation evidence differs between collections')
                        name = arm if kind == 'selected' else arm+'_frozen'
                        relative = f'{campaign}/test/fold{fold}/{name}'
                        paths = [root/relative/'scores.npz' for root in (reference_root, replay_root)]
                        with np.load(paths[0], allow_pickle=False) as left, np.load(paths[1], allow_pickle=False) as right:
                            original, regenerated = left['scores'], right['scores']
                            if original.dtype != regenerated.dtype:
                                raise ValueError('Prediction dtype differs between collections')
                            # Promote before subtraction so a floating-point overflow
                            # cannot hide a failed comparison. Tolerance is relative
                            # to the production score, not whichever value is larger.
                            original = original.astype(np.float64)
                            regenerated = regenerated.astype(np.float64)
                            difference = np.abs(regenerated-original)
                            outside = int(np.count_nonzero(difference > ATOL+RTOL*np.abs(original)))
                            maximum = float(difference.max())
                            cells = int(original.size)
                        for path, evidence in zip(paths, loaded):
                            checked_file(path, evidence[1]['scores.npz'])
                        rank_comparisons = {}
                        for mode in ('raw', 'calibrated'):
                            left, right = loaded[0][0][mode], loaded[1][0][mode]
                            rank_comparisons[mode] = dict(changed_ranks=int(np.count_nonzero(left != right)),
                                max_absolute_rank_difference=float(np.max(np.abs(left-right))),
                                reference_metrics=metrics(left), replay_metrics=metrics(right))
                        records.append(dict(bundle=relative, score_cells=cells, association_pairs=len(rows),
                            score_values_outside_tolerance=outside, maximum_absolute_score_difference=maximum,
                            ranks=rank_comparisons, reference_evidence=loaded[0][1], replay_evidence=loaded[1][1]))
                    if initial_and_selected['selected'][0][2] == 0:
                        for side in (0, 1):
                            for mode in ('raw', 'calibrated'):
                                np.testing.assert_array_equal(initial_and_selected['selected'][side][0][mode],
                                    initial_and_selected['initial'][side][0][mode],
                                    err_msg='Step-zero selection differs from initial checkpoint ranks')
        print('COMPARED_CONTROLLED_FOLD', fold, 48, flush=True)
    if len(records) != 480:
        raise ValueError('Incomplete prediction comparison')
    # Recheck the frozen selections after the full read, just as the scorer does.
    for root in (reference_root, replay_root):
        if verify_plan(root) != (reference_plan, reference_global):
            raise ValueError('Global selection changed during comparison')
    outside = sum(r['score_values_outside_tolerance'] for r in records)
    changed = {mode: sum(r['ranks'][mode]['changed_ranks'] for r in records) for mode in ('raw', 'calibrated')}
    return dict(status='passed' if outside == 0 and not any(changed.values()) else 'requires_review',
        score_bundles=480, training_runs=240, score_cells=sum(r['score_cells'] for r in records),
        association_rank_pairs_per_mode=sum(r['association_pairs'] for r in records),
        score_tolerance=dict(rtol=RTOL, atol=ATOL, reference='production'),
        score_values_outside_tolerance=outside, changed_ranks=changed,
        maximum_absolute_score_difference=max(r['maximum_absolute_score_difference'] for r in records),
        global_selection_sha256=reference_global, protocol_sha256=reference_plan['protocol_sha256'],
        input_sha256=inputs, comparator_sha256=sha(__file__),
        bundle_verifier_sha256=sha(Path(sys.modules[load_bundle.__module__].__file__)),
        scope='Verified collection comparison: prediction tolerance and exact raw/per-gene-calibrated association ranks. Execution provenance, training reproduction and downstream bias-condition comparisons are separate evidence.',
        records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--replay-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new comparison output path')
    result = compare(args.reference_root, args.replay_root)
    write(args.output, result)
    print(result['status'], '480 bundles; changed ranks:', result['changed_ranks'])
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
