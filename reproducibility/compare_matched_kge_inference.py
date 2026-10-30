"""Compare all 400 regenerated matched KGE predictions with production scores."""
import argparse
import copy
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'code')]
from build_matched_kge_manifest import build, INPUTS
from analysis.evaluate_stepwise_bias import std_loo_cal
from analysis.summarize_query_tables import metadata
from phenoquery.common import metrics, pairs, ranks, sha, write

RTOL, ATOL = 1e-5, 1e-6


def training_evidence(manifest):
    result = copy.deepcopy(manifest)
    for record in result['evidence'].values():
        del record['scored_sha256']
    for model in result['models']:
        # Paths and prediction bytes may change after independent inference.
        model['folds'] = {}
    return result


def load(path, digest, genes, diseases, training_diseases):
    if sha(path) != digest:
        raise ValueError('Prediction checksum mismatch')
    with np.load(path, allow_pickle=False) as bundle:
        if sorted(bundle.files) != ['diseases', 'genes', 'scores', 'train_diseases', 'train_scores']:
            raise ValueError('Unexpected or duplicate prediction arrays')
        if (bundle['genes'].astype(str).tolist() != genes
                or bundle['diseases'].astype(str).tolist() != diseases
                or bundle['train_diseases'].astype(str).tolist() != training_diseases):
            raise ValueError('Prediction identities or ordering differ from the benchmark')
        arrays = {name:bundle[name] for name in ('scores', 'train_scores')}
        for name, identities in [('scores', diseases), ('train_scores', training_diseases)]:
            values = arrays[name]
            if (values.dtype.kind != 'f' or values.shape != (len(identities), len(genes))
                    or not np.isfinite(values).all()):
                raise ValueError('Invalid prediction dtype, shape or values')
    if sha(path) != digest:
        raise ValueError('Prediction changed during comparison')
    return arrays


def compare(reference_root, replay_root, reference_data, replay_data,
            reference_annotations, replay_annotations, expected_candidates=13625):
    roots = [Path(p).resolve() for p in (reference_root, replay_root)]
    data_roots = [Path(p).resolve() for p in (reference_data, replay_data)]
    annotations = [Path(p).resolve() for p in (reference_annotations, replay_annotations)]
    if roots[0] == roots[1]:
        raise ValueError('Reference and replay must be separate collections')
    manifests = [build(root, data, root/'score-manifest.json', annotation,
                       expected_candidates=expected_candidates)
                 for root, data, annotation in zip(roots, data_roots, annotations)]
    if training_evidence(manifests[0]) != training_evidence(manifests[1]):
        raise ValueError('Frozen training, benchmark or selection evidence differs')
    records, seen_diseases = [], set()
    for fold in range(10):
        genes, rows, _, _, _ = metadata(data_roots[0]/f'fold{fold}')
        diseases = sorted({d for _, d in rows})
        training_diseases = sorted({d for _, d in pairs(data_roots[0]/f'fold{fold}'/'training.tsv')})
        if len(genes) != expected_candidates or seen_diseases.intersection(diseases):
            raise ValueError('Candidate count or outer test-fold identity mismatch')
        seen_diseases.update(diseases)
        for models in zip(*(manifest['models'] for manifest in manifests)):
            entries = [model['folds'][str(fold)] for model in models]
            paths = [root/entry['path'] for root, entry in zip(roots, entries)]
            if paths[0].resolve() == paths[1].resolve():
                raise ValueError('Reference and replay scores resolve to the same file')
            values = [load(path, entry['sha256'], genes, diseases, training_diseases)
                      for path, entry in zip(paths, entries)]
            predictions = {}
            for name in ('scores', 'train_scores'):
                if values[0][name].dtype != values[1][name].dtype:
                    raise ValueError('Prediction dtype differs between collections')
                original, regenerated = [v[name].astype(np.float64) for v in values]
                difference = np.abs(regenerated-original)
                predictions[name] = dict(score_cells=int(original.size),
                    score_values_outside_tolerance=int(np.count_nonzero(difference > ATOL+RTOL*np.abs(original))),
                    maximum_absolute_score_difference=float(difference.max()))
            original, regenerated = [v['scores'].astype(np.float64) for v in values]
            comparison = {}
            for mode in ('raw', 'calibrated'):
                matrices = [original, regenerated]
                if mode == 'calibrated':
                    matrices = [std_loo_cal(matrix) for matrix in matrices]
                left, right = [ranks(matrix, diseases, genes, rows) for matrix in matrices]
                comparison[mode] = dict(changed_ranks=int(np.count_nonzero(left != right)),
                    max_absolute_rank_difference=float(np.max(np.abs(left-right))),
                    reference_metrics=metrics(left), replay_metrics=metrics(right))
            records.append(dict(fold=fold, model=models[0]['name'],
                score_cells=sum(p['score_cells'] for p in predictions.values()), association_pairs=len(rows),
                score_values_outside_tolerance=sum(p['score_values_outside_tolerance'] for p in predictions.values()),
                maximum_absolute_score_difference=max(p['maximum_absolute_score_difference'] for p in predictions.values()),
                predictions=predictions, ranks=comparison,
                reference_score_sha256=entries[0]['sha256'], replay_score_sha256=entries[1]['sha256']))
        print('COMPARED_MATCHED_FOLD', fold, 40, flush=True)
    if len(records) != 400:
        raise ValueError('Incomplete matched prediction comparison')
    # Recheck the small binding records and inputs after the complete read. The
    # builder already checked each checkpoint, selection and score before it.
    for root, data, annotation, manifest in zip(roots, data_roots, annotations, manifests):
        if (sha(root/'FROZEN.json') != manifest['freeze_sha256']
                or sha(annotation) != manifest['gene_phenotypes_sha256']):
            raise ValueError('Frozen campaign inputs changed during comparison')
        for run, evidence in manifest['evidence'].items():
            for filename, key in [('TRAINED.json', 'training_sha256'),
                                  ('identity.json', 'identity_sha256'),
                                  ('SCORED.json', 'scored_sha256'),
                                  ('validation.json', 'validation_sha256')]:
                if sha(root/run/filename) != evidence[key]:
                    raise ValueError('Campaign evidence changed during comparison')
        for fold in range(10):
            for name in INPUTS:
                if sha(data/f'fold{fold}'/name) != manifest['benchmark'][str(fold)][name]:
                    raise ValueError('Benchmark changed during comparison')
    outside = sum(record['score_values_outside_tolerance'] for record in records)
    changed = {mode:sum(record['ranks'][mode]['changed_ranks'] for record in records)
               for mode in ('raw', 'calibrated')}
    return dict(status='passed' if outside == 0 and not any(changed.values()) else 'requires_review',
        training_runs=100, score_files=400, variants=40,
        score_cells=sum(record['score_cells'] for record in records),
        association_rank_pairs_per_mode=sum(record['association_pairs'] for record in records),
        score_tolerance=dict(rtol=RTOL, atol=ATOL, reference='production'),
        score_values_outside_tolerance=outside, changed_ranks=changed,
        maximum_absolute_score_difference=max(record['maximum_absolute_score_difference'] for record in records),
        freeze_sha256=manifests[0]['freeze_sha256'], protocol_sha256=manifests[0]['protocol_sha256'],
        benchmark_sha256=manifests[0]['benchmark'], comparator_sha256=sha(__file__),
        verifier_sha256=sha(HERE/'build_matched_kge_manifest.py'),
        rank_source_sha256=sha(Path(sys.modules[ranks.__module__].__file__)),
        calibration_source_sha256=sha(Path(sys.modules[std_loo_cal.__module__].__file__)),
        scope='All frozen choices: test and auxiliary training prediction tolerance, plus exact raw/per-gene-calibrated test-association ranks. Independent execution provenance, training reproduction and all 42 bias conditions require separate evidence.',
        records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root', 'replay-root', 'reference-data', 'replay-data',
                 'reference-annotations', 'replay-annotations', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop('output')
    if output.exists():
        raise FileExistsError('Choose a new comparison output path')
    result = compare(**args)
    write(output, result)
    print(result['status'], '400 score files; changed ranks:', result['changed_ranks'])
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
