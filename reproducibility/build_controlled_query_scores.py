"""Package the complete controlled-query score-to-report replay without model weights."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'code')]
from analysis.summarize_controlled_query import summarize
from phenoquery.common import read, sha, write
from checkpoint_archives import CheckpointInventory, MAX_PART_BYTES, verify_packed


def build(root, report_json, max_part_bytes=MAX_PART_BYTES, expected_candidates=13625):
    root, report_json = Path(root).resolve(), Path(report_json).resolve()
    report_sha = sha(report_json)
    report = read(report_json)
    # Recompute all ranks, strata, seeds and the prespecified bootstrap. Comparing
    # through JSON also normalizes integer dictionary keys in the serialized report.
    replayed = json.loads(json.dumps(summarize(root, expected_candidates=expected_candidates)))
    if report != replayed:
        raise ValueError('Reference report differs from complete controlled score replay')
    plan = read(root / 'CONTROLLED.json')
    inventory = CheckpointInventory(root)
    inventory.include('CONTROLLED.json', report['plan_sha256'])
    inventory.include('GLOBAL-SELECTION.json', report['global_selection_sha256'])
    inputs = []
    for campaign in plan['campaigns']:
        child = root / campaign
        inventory.include(f'{campaign}/source.sha256', plan['source_sha256'])
        inventory.include(f'{campaign}/PROTOCOL.md', plan['protocol_sha256'])
        frozen = read(root / 'GLOBAL-SELECTION.json')
        inventory.include(f'{campaign}/SELECTION.json', frozen['selections'][campaign])
        benchmark = CheckpointInventory(child / 'data')
        for fold in range(10):
            for name, digest in report['input_sha256'][f'data/fold{fold}'].items():
                benchmark.include(f'fold{fold}/{name}', digest)
            for arm in plan['arms']:
                parent = f'{campaign}/training/{arm}-fold{fold}'
                for kind in ('initial', 'selected'):
                    evidence = report['input_sha256'][f'{campaign}/fold{fold}/{arm}/{kind}']
                    inventory.include(f'{parent}/COMPLETE.json', evidence['training_completion_sha256'])
                    inventory.include(f'{parent}/identity.json', evidence['training_identity_sha256'])
                    inventory.include(f'{parent}/progress.json', evidence['validation_trajectory_sha256'])
                    validation = 'validation.npz' if kind == 'selected' else 'initial-validation.npz'
                    inventory.include(f'{parent}/{validation}', evidence['validation_scores_sha256'])
                    score_arm = arm if kind == 'selected' else arm + '_frozen'
                    output = f'{campaign}/test/fold{fold}/{score_arm}'
                    inventory.include(f'{output}/COMPLETE.json', evidence['completion_sha256'])
                    for name in ('scores.npz', 'ranks.npz'):
                        inventory.include(f'{output}/{name}', evidence[name])
        # Resolve the original data link once, but install ordinary files at each
        # campaign's expected paths. Replay requires no machine-specific symlinks.
        inputs.append((benchmark, f'controlled-data-{campaign}', f'controlled-query/{campaign}/data'))
    reference = CheckpointInventory(report_json.parent)
    reference.include(report_json.name, report_sha)
    inputs.extend([(inventory, 'controlled-scores', 'controlled-query'),
                   (reference, 'controlled-reference', 'controlled-query-reference')])
    components, expected = [], {}
    for collection, archive_prefix, destination_prefix in inputs:
        members = collection.expected_files(destination_prefix)
        if expected.keys() & members.keys():
            raise ValueError('Duplicate score archive destination')
        expected.update(members)
        parts = collection.components(archive_prefix, destination_prefix, max_part_bytes)
        for part in parts:
            part['description'] = 'Controlled-query saved scores and analysis evidence'
        components.extend(parts)
    if sha(report_json) != report_sha or sha(root/'GLOBAL-SELECTION.json') != report['global_selection_sha256']:
        raise ValueError('Report or global selection changed during export')
    return dict(description='Complete controlled-query saved-score replay: 240 selections and 480 score bundles',
        provenance=dict(builder_sha256=sha(__file__), report_sha256=report_sha,
            reference_report=f'controlled-query-reference/{report_json.name}',
            global_selection_sha256=report['global_selection_sha256'],
            protocol_sha256=report['protocol_sha256'], analysis_source_sha256=report['analysis_source_sha256'],
            training_runs=240, score_bundles=480, expected_candidates=expected_candidates,
            scope='Saved-score statistics and table replay; model inference and training are separate verification steps',
            expected_files=expected), components=components)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--report-json', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('verify-packed')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'plan':
        result = build(args.root, args.report_json)
        if args.output.exists() and read(args.output) != result:
            raise ValueError('Existing export plan differs; choose a new output path')
        write(args.output, result)
        print('Verified all 480 score bundles and the complete controlled report')
    else:
        result = verify_packed(args.plan, args.manifest)
        result['scope'] = 'Packed score/evidence inventory matches the verified export plan; extraction is separate'
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
