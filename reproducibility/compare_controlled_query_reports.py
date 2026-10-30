"""Compare complete primary reports after verified independent query inference."""
import argparse
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'code')]
from compare_controlled_query_inference import TRAINING_EVIDENCE, RTOL, ATOL
from analysis.summarize_controlled_query import ARMS, INITS, SEEDS
from phenoquery.common import read, sha, write

OUTPUT_EVIDENCE = {'completion_sha256', 'scores.npz', 'ranks.npz'}


def differences(left, right, path=''):
    """Keep every changed field, including unrounded bootstrap endpoints."""
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            name = path+'/'+key
            if key not in left or key not in right:
                result.append(dict(field=name, reference=left.get(key), replay=right.get(key),
                                   missing_from='reference' if key not in left else 'replay'))
            else:
                result.extend(differences(left[key], right[key], name))
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [d for i,(a,b) in enumerate(zip(left,right)) for d in differences(a,b,path+'/'+str(i))]
    if left != right:
        return [dict(field=path,reference=left,replay=right)]
    return []


def require_finite(value):
    if isinstance(value, dict):
        for item in value.values(): require_finite(item)
    elif isinstance(value, list):
        for item in value: require_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite report or inference-comparison value')


def compare(reference_report, replay_report, inference_comparison, reference_sha256):
    paths = [Path(p) for p in (reference_report,replay_report,inference_comparison)]
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError('Require regular report and comparison files')
    if paths[0].samefile(paths[1]):
        raise ValueError('Require separate reference and replay reports')
    hashes = [sha(p) for p in paths]
    if hashes[0] != reference_sha256:
        raise ValueError('Reference report differs from the verified analysis digest')
    reference,replay,comparison = [read(p) for p in paths]
    for item in (reference,replay,comparison): require_finite(item)
    if (comparison['status'] != 'passed' or comparison['score_bundles'] != 480
            or comparison['training_runs'] != 240
            or comparison['score_values_outside_tolerance'] != 0
            or comparison['changed_ranks'] != dict(raw=0,calibrated=0)
            or comparison['score_tolerance'] != dict(rtol=RTOL,atol=ATOL,reference='production')
            or comparison['comparator_sha256'] != sha(HERE/'compare_controlled_query_inference.py')
            or comparison['bundle_verifier_sha256'] != sha(HERE.parent/'code/analysis/summarize_controlled_query.py')):
        raise ValueError('Require successful complete inference comparison with the pinned implementation')
    expected = {}
    for init in INITS:
        for seed in SEEDS:
            campaign = f'{init}-seed{seed}'
            for fold in range(10):
                for arm in ARMS:
                    for kind in ('initial','selected'):
                        name = arm+'_frozen' if kind == 'initial' else arm
                        expected[f'{campaign}/test/fold{fold}/{name}'] = f'{campaign}/fold{fold}/{arm}/{kind}'
    records = {r['bundle']:r for r in comparison['records']}
    if len(comparison['records']) != 480 or set(records) != set(expected):
        raise ValueError('Incomplete or duplicated inference bundle coverage')
    input_keys = set(expected.values()) | {f'data/fold{fold}' for fold in range(10)}
    conditions = {f'{i}/{a}/{k}/{m}' for i in INITS for a in ARMS
                  for k in ('initial','selected') for m in ('raw','calibrated')}
    contrast_keys = [(i,c) for i in INITS for c in ('disease_bridges','conjunction','typed_EQ','adaptation')]
    for report in (reference,replay):
        if (report['schema'] != 1 or report['training_runs'] != 240 or report['score_bundles'] != 480
                or set(report['conditions']) != conditions
                or [(c['initialization'],c['contrast']) for c in report['primary_contrasts']] != contrast_keys
                or set(report['input_sha256']) != input_keys
                or set(report['selected_steps']) != {k for k in expected.values() if k.endswith('/selected')}
                or [f['fold'] for f in report['folds']] != list(range(10))):
            raise ValueError('Require complete controlled reports and frozen selections')
        for key in ('global_selection_sha256','protocol_sha256'):
            if report[key] != comparison[key]:
                raise ValueError('Report and inference comparison use different frozen experiments')
        if report['analysis_source_sha256']['summarize_controlled_query.py'] != comparison['bundle_verifier_sha256']:
            raise ValueError('Report and inference comparison use different report implementations')
        for fold in range(10):
            if report['input_sha256'][f'data/fold{fold}'] != comparison['input_sha256'][str(fold)]:
                raise ValueError('Report and inference comparison use different benchmark inputs')
    changes = []
    for bundle,key in expected.items():
        record = records[bundle]
        if (record['score_values_outside_tolerance'] != 0
                or set(record['ranks']) != {'raw','calibrated'}
                or any(record['ranks'][m]['changed_ranks'] != 0 for m in record['ranks'])):
            raise ValueError('A bundle failed inference comparison')
        a,b = (report['input_sha256'][key] for report in (reference,replay))
        if (set(a) != OUTPUT_EVIDENCE | set(TRAINING_EVIDENCE) or set(b) != set(a)
                or a != record['reference_evidence'] or b != record['replay_evidence']
                or any(a[k] != b[k] for k in TRAINING_EVIDENCE)):
            raise ValueError('Report inputs are not bound to the compared prediction bundles')
        for field in sorted(OUTPUT_EVIDENCE):
            if a[field] != b[field]:
                changes.append(dict(bundle=bundle,field=field,reference=a[field],replay=b[field]))
    # Only the three explicitly checked inference-output digests may differ.
    # Every other report field is compared, including all conditions, strata,
    # seed dispersions, selection records, and bootstrap settings/endpoints.
    scientific = [{k:v for k,v in report.items() if k != 'input_sha256'} for report in (reference,replay)]
    changed = differences(*scientific)
    if [sha(p) for p in paths] != hashes:
        raise ValueError('Evidence changed during comparison')
    return dict(status='requires_review' if changed else 'passed',score_bundles=480,
        conditions=32,primary_contrasts=8,report_differences=changed,
        verified_output_digest_changes=changes,reference_report_sha256=hashes[0],
        replay_report_sha256=hashes[1],inference_comparison_sha256=hashes[2],comparator_sha256=sha(__file__),
        scope='Complete primary report agreement, including all eight bootstrap contrasts, bound to a successful 480-bundle score/rank comparison and a previously verified reference digest. Reports must be regenerated from their own scores. Execution provenance and all bias sensitivities remain separate checks.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-report','replay-report','inference-comparison','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-sha256',required=True)
    args = vars(parser.parse_args()); output = args.pop('output')
    if output.exists(): raise FileExistsError('Choose a new comparison output path')
    result = compare(**args)
    write(output,result)
    print(result['status'],len(result['report_differences']),'changed report fields')
    if result['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
