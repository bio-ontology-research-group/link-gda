"""Export and replay all matched KGE score conditions and prespecified comparisons."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'code')]
from analysis.summarize_matched_kge import summarize
from build_matched_kge_manifest import build as build_manifest
from checkpoint_archives import CheckpointInventory, MAX_PART_BYTES, verify_packed
from phenoquery.common import read, sha, write


def replay(score_manifest, data_root, report_json, summary_json, ranks_dir, output,
           workers=4, expected_candidates=13625):
    score_manifest, data_root, report_json, summary_json, ranks_dir, output = map(
        lambda p: Path(p).resolve(), (score_manifest, data_root, report_json, summary_json, ranks_dir, output))
    if data_root.name != 'data':
        raise ValueError('Shared evaluator requires benchmark files under WORKSPACE/data')
    if workers < 1:
        raise ValueError('At least one evaluator worker is required')
    output.mkdir(parents=True, exist_ok=False)
    evaluator = HERE.parent/'code/analysis/evaluate_stepwise_bias.py'
    command = [sys.executable, str(evaluator), '--base-dir', str(data_root.parent),
        '--score-manifest', str(score_manifest), '--out-json', str(output/'evaluation.json'),
        '--ranks-dir', str(output/'ranks'), '--workers', str(workers)]
    with (output/'evaluation.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    comparisons = [(report_json, output/'evaluation.json'),
                   (report_json.with_suffix('.metadata.json'), output/'evaluation.metadata.json')]
    comparisons.extend((ranks_dir/f'fold{fold}-{kind}', output/'ranks'/f'fold{fold}-{kind}')
                       for fold in range(10) for kind in ('rows.json', 'ranks.npz'))
    for reference, actual in comparisons:
        if sha(reference) != sha(actual):
            raise ValueError(f'Full matched evaluation replay differs: {reference.name}')
    result = summarize(score_manifest, output/'evaluation.json', output/'ranks', data_root,
                       expected_candidates=expected_candidates)
    write(output/'summary.json', result)
    if sha(summary_json) != sha(output/'summary.json'):
        raise ValueError('Matched primary summary differs from the complete score replay')
    proof = dict(status='passed', scope='Saved-score replay only; model inference and training are separate',
        models=40, conditions_per_model=42, primary_comparisons=10, rank_files=20,
        score_manifest_sha256=sha(score_manifest), report_sha256=sha(report_json),
        metadata_sha256=sha(report_json.with_suffix('.metadata.json')),
        summary_sha256=sha(summary_json), evaluator_sha256=sha(evaluator),
        replay_source_sha256=sha(__file__), log_sha256=sha(output/'evaluation.log'))
    write(output/'verification.json', proof)
    return proof


def build(root, data_root, score_manifest, gene_phenotypes, report_json, summary_json,
          ranks_dir, verification_dir, workers=4, max_part_bytes=MAX_PART_BYTES,
          expected_candidates=13625):
    root, data_root, score_manifest, report_json, summary_json, ranks_dir = map(
        lambda p: Path(p).resolve(), (root, data_root, score_manifest, report_json, summary_json, ranks_dir))
    if score_manifest.parent != root:
        raise ValueError('Create the score manifest directly inside the matched run root before evaluation')
    verified = build_manifest(root, data_root, score_manifest, gene_phenotypes,
                              expected_candidates=expected_candidates)
    if read(score_manifest) != verified:
        raise ValueError('Score manifest differs from the complete verified campaign')
    # Preserve manifest bytes and therefore every downstream provenance hash.
    for model in verified['models']:
        for fold, entry in model['folds'].items():
            expected = f"fold{fold}/{model['graph']}/{model['family']}/scores-{model['validation_choice']}.npz"
            if entry['path'] != expected:
                raise ValueError('Score manifest does not use the portable campaign-relative layout')
    proof = replay(score_manifest, data_root, report_json, summary_json, ranks_dir,
                   verification_dir, workers, expected_candidates)
    inventory = CheckpointInventory(root)
    inventory.include(score_manifest.name, proof['score_manifest_sha256'])
    inventory.include('FROZEN.json', verified['freeze_sha256'])
    for run, evidence in verified['evidence'].items():
        inventory.include(f'{run}/TRAINED.json', evidence['training_sha256'])
        inventory.include(f'{run}/identity.json', evidence['identity_sha256'])
        inventory.include(f'{run}/validation.json', evidence['validation_sha256'])
        inventory.include(f'{run}/SCORED.json', evidence['scored_sha256'])
    for model in verified['models']:
        for entry in model['folds'].values():
            inventory.include(entry['path'], entry['sha256'])
    data = CheckpointInventory(data_root)
    for fold, inputs in verified['benchmark'].items():
        for name, digest in inputs.items():
            data.include(f'fold{fold}/{name}', digest)
    report = CheckpointInventory(report_json.parent)
    report.include(report_json.name, proof['report_sha256'])
    report.include(report_json.with_suffix('.metadata.json').name, proof['metadata_sha256'])
    summary = CheckpointInventory(summary_json.parent)
    summary.include(summary_json.name, proof['summary_sha256'])
    ranks = CheckpointInventory(ranks_dir)
    for fold in range(10):
        for kind in ('rows.json', 'ranks.npz'):
            name = f'fold{fold}-{kind}'
            ranks.include(name, sha(Path(verification_dir)/'ranks'/name))
    verification = CheckpointInventory(verification_dir)
    verification.include('verification.json')
    collections = [(inventory, 'matched-scores', 'matched-kge'),
        (data, 'matched-benchmark', 'matched-benchmark/data'),
        (report, 'matched-evaluation', 'matched-kge-reference/evaluation'),
        (summary, 'matched-summary', 'matched-kge-reference/summary'),
        (ranks, 'matched-ranks', 'matched-kge-reference/ranks'),
        (verification, 'matched-replay-proof', 'matched-kge-reference/verification')]
    components, expected_files = [], {}
    for collection, archive_prefix, destination_prefix in collections:
        members = collection.expected_files(destination_prefix)
        if expected_files.keys() & members.keys():
            raise ValueError('Duplicate matched archive member')
        expected_files.update(members)
        parts = collection.components(archive_prefix, destination_prefix, max_part_bytes)
        for part in parts:
            part['description'] = 'Matched KGE saved scores, benchmark and analysis evidence'
        components.extend(parts)
    return dict(description='Complete matched KGE score replay: 400 scores, 40 models and 42 conditions',
        provenance=dict(builder_sha256=sha(__file__), score_manifest_sha256=proof['score_manifest_sha256'],
            score_manifest=f'matched-kge/{score_manifest.name}',
            reference_report=f'matched-kge-reference/evaluation/{report_json.name}',
            reference_summary=f'matched-kge-reference/summary/{summary_json.name}',
            reference_ranks='matched-kge-reference/ranks', data_root='matched-benchmark/data',
            freeze_sha256=verified['freeze_sha256'], protocol_sha256=verified['protocol_sha256'],
            training_runs=100, score_files=400, models=40, conditions_per_model=42,
            expected_candidates=expected_candidates, replay=proof, expected_files=expected_files),
        components=components)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('plan', 'replay'):
        p = sub.add_parser(action)
        for arg in ('data-root', 'score-manifest', 'report-json', 'summary-json', 'ranks-dir', 'output'):
            p.add_argument('--'+arg, type=Path, required=True)
        p.add_argument('--workers', type=int, default=4)
        if action == 'plan':
            for arg in ('root', 'gene-phenotypes', 'verification-dir'):
                p.add_argument('--'+arg, type=Path, required=True)
    p = sub.add_parser('verify-packed')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'plan':
        if args.output.exists():
            parser.error('Choose a new export plan path')
        result = build(args.root, args.data_root, args.score_manifest, args.gene_phenotypes,
            args.report_json, args.summary_json, args.ranks_dir, args.verification_dir, args.workers)
        write(args.output, result)
        print('Verified the full matched campaign and all 40-by-42 evaluation conditions')
    elif args.action == 'replay':
        print(json.dumps(replay(args.score_manifest, args.data_root, args.report_json,
            args.summary_json, args.ranks_dir, args.output, args.workers), indent=2))
    else:
        result = verify_packed(args.plan, args.manifest)
        result['scope'] = 'Packed score/evidence inventory matches the verified plan; extraction is separate'
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
