"""Compare all 42 bias conditions after independently regenerating model scores."""
import argparse
import copy
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'code')]
from generate_production_bias_tables import validate, METRICS
from phenoquery.common import read, sha, write

RANK_KEYS = (
    'r_s1_reference_dedup', 'r_s2_reference_dedup', 'r_s1_positive_filter',
    'r_s2_positive_filter', 'r_s0', 'r_s1', 'r_s2', 'r_s2_raw',
    'r_s4_filt', 'r_s4_raw_filt', 'r_s1_filt', 'r_s0_filt', 'r_ws_s1', 'r_ws_s0',
)


def compare(reference_manifest, replay_manifest, reference_report, replay_report,
            reference_ranks, replay_ranks):
    manifests, reports, directories = [[Path(p).resolve() for p in pair] for pair in
        [(reference_manifest,replay_manifest), (reference_report,replay_report),
         (reference_ranks,replay_ranks)]]
    if any(left == right for left,right in (manifests,reports,directories)):
        raise ValueError('Require separate reference and replay analysis artifacts')
    collections = []
    for manifest_path,report_path,ranks_dir in zip(manifests,reports,directories):
        manifest,report = read(manifest_path),read(report_path)
        metadata_path = report_path.with_suffix('.metadata.json')
        metadata = read(metadata_path)
        kind,identities,_ = validate(report,metadata,manifest,sha(manifest_path))
        if metadata.get('schema_version') != 1:
            raise ValueError('Unsupported evaluation metadata schema')
        expected = {f'fold{fold}-{suffix}' for fold in range(10) for suffix in ('ranks.npz','rows.json')}
        if {p.name for p in ranks_dir.iterdir()} != expected:
            raise ValueError('Require exactly ten complete rank/row file pairs')
        files = [manifest_path,report_path,metadata_path,*sorted(ranks_dir.iterdir())]
        if any(not p.is_file() or p.is_symlink() for p in files):
            raise ValueError('Require regular analysis artifacts')
        collections.append(dict(kind=kind,identities=identities,report=report,metadata=metadata,
            hashes={str(p):sha(p) for p in files}))
    left,right = collections
    if left['kind'] != right['kind'] or left['identities'] != right['identities']:
        raise ValueError('Reference and replay experimental designs differ')
    metadata = [copy.deepcopy(c['metadata']) for c in collections]
    for item in metadata:
        # Each distinct manifest hash was checked against its own report above.
        del item['score_manifest_sha256']
    if metadata[0] != metadata[1]:
        raise ValueError('Evaluation implementation, cohorts or metadata differ')
    differences = []
    for model,conditions in left['report'].items():
        for condition,values in conditions.items():
            for metric in METRICS:
                actual = right['report'][model][condition][metric]
                if values[metric] != actual:
                    differences.append(dict(model=model,condition=condition,metric=metric,
                        reference=values[metric],replay=actual))
    names = {model+'::'+key for model in left['identities'] for key in RANK_KEYS}
    rank_records,associations,total_values,changed = [],0,0,0
    for fold in range(10):
        row_paths = [d/f'fold{fold}-rows.json' for d in directories]
        rank_paths = [d/f'fold{fold}-ranks.npz' for d in directories]
        if row_paths[0].samefile(row_paths[1]) or rank_paths[0].samefile(rank_paths[1]):
            raise ValueError('Reference and replay share rank artifacts')
        if sha(row_paths[0]) != sha(row_paths[1]):
            raise ValueError(f'Paired row identities or ordering differ: fold {fold}')
        rows = read(row_paths[0])
        if not isinstance(rows,list) or not rows:
            raise ValueError('Missing evaluation rows')
        associations += len(rows)
        fold_changed,maximum = 0,0.0
        with np.load(rank_paths[0],allow_pickle=False) as original, np.load(rank_paths[1],allow_pickle=False) as actual:
            if (len(original.files) != len(names) or len(actual.files) != len(names)
                    or set(original.files) != names or set(actual.files) != names):
                raise ValueError(f'Missing, duplicate or unexpected rank arrays: fold {fold}')
            changed_arrays = {}
            for name in sorted(names):
                a,b = original[name],actual[name]
                if a.dtype != b.dtype or a.shape != b.shape or a.shape != (len(rows),):
                    raise ValueError(f'Rank dtype/shape or row count differs: {fold}/{name}')
                if any(v.dtype.kind not in 'fi' or not np.isfinite(v).all()
                       or (v < 1).any() or (v > 13625).any() for v in (a,b)):
                    raise ValueError(f'Invalid rank values: {fold}/{name}')
                count = int(np.count_nonzero(a != b))
                error = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
                total_values += a.size
                fold_changed += count
                maximum = max(maximum,error)
                if count: changed_arrays[name] = dict(changed_ranks=count,maximum_absolute_difference=error)
        changed += fold_changed
        rank_records.append(dict(fold=fold,arrays=len(names),association_pairs=len(rows),
            changed_ranks=fold_changed,maximum_absolute_difference=maximum,changed_arrays=changed_arrays,
            reference_sha256=sha(rank_paths[0]),replay_sha256=sha(rank_paths[1]),rows_sha256=sha(row_paths[0])))
    if associations != left['metadata']['cohorts']['pairs']:
        raise ValueError('Saved rank rows do not cover the full evaluation cohort')
    for collection in collections:
        for name,digest in collection['hashes'].items():
            if sha(Path(name)) != digest:
                raise ValueError('Analysis artifact changed during comparison')
    return dict(status='passed' if not differences and changed == 0 else 'requires_review',
        kind=left['kind'],variants=len(left['identities']),conditions_per_variant=42,folds=10,
        association_pairs=associations,rank_arrays=len(names)*10,rank_values=int(total_values),
        changed_ranks=changed,metric_differences=differences,rank_comparisons=rank_records,
        reference_artifacts=left['hashes'],replay_artifacts=right['hashes'],
        comparator_sha256=sha(__file__),report_validator_sha256=sha(HERE/'generate_production_bias_tables.py'),
        evaluator_sha256=left['metadata']['evaluation_sha256'],
        scope='Complete reported bias-condition and saved-rank agreement against a verified reference. Each report is bound to its own score manifest. Independent execution, score/weight verification and primary bootstrap replay remain separate requirements.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-manifest','replay-manifest','reference-report','replay-report',
                 'reference-ranks','replay-ranks','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = vars(parser.parse_args()); output = args.pop('output')
    if output.exists(): raise FileExistsError('Choose a new comparison output path')
    proof = compare(**args)
    write(output,proof)
    print(proof['status'],proof['rank_arrays'],'arrays;',len(proof['metric_differences']),
          'metric differences;',proof['changed_ranks'],'changed ranks')
    if proof['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
