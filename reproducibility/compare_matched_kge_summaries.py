"""Compare matched primary statistics after verified full bias replay."""
import argparse
import copy
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),str(HERE.parent/'code')]
from compare_controlled_query_reports import differences, require_finite
from analysis.summarize_matched_kge import primary_spec, PROTOCOL
from phenoquery.common import read, sha, write


def compare(reference_summary,replay_summary,reference_manifest,replay_manifest,
            reference_report,replay_report,bias_comparison,reference_sha256):
    summaries,manifests,reports = [[Path(p).resolve() for p in pair] for pair in
        [(reference_summary,replay_summary),(reference_manifest,replay_manifest),(reference_report,replay_report)]]
    comparison_path = Path(bias_comparison).resolve()
    if any(a.samefile(b) for a,b in (summaries,manifests,reports)):
        raise ValueError('Require separate reference and replay artifacts')
    files = [*summaries,*manifests,*reports,comparison_path]
    hashes = {str(p):sha(p) for p in files}
    if hashes[str(summaries[0])] != reference_sha256:
        raise ValueError('Reference summary differs from the verified analysis digest')
    comparison = read(comparison_path)
    require_finite(comparison)
    if (comparison['status'] != 'passed' or comparison['kind'] != 'matched'
            or comparison['variants'] != 40 or comparison['conditions_per_variant'] != 42
            or comparison['folds'] != 10 or comparison['rank_arrays'] != 5600
            or comparison['changed_ranks'] != 0 or comparison['metric_differences']
            or comparison['comparator_sha256'] != sha(HERE/'compare_production_bias_replays.py')
            or comparison['report_validator_sha256'] != sha(HERE/'generate_production_bias_tables.py')
            or comparison['evaluator_sha256'] != sha(HERE.parent/'code/analysis/evaluate_stepwise_bias.py')):
        raise ValueError('Require a successful complete matched bias comparison')
    rank_records = {r['fold']:r for r in comparison['rank_comparisons']}
    if len(comparison['rank_comparisons']) != 10 or set(rank_records) != set(range(10)):
        raise ValueError('Incomplete or duplicated rank comparison folds')
    expected_primary = primary_spec()
    primary_names = {spec[graph] for spec in expected_primary.values() for graph in ('original','nobridge')}
    normalized,manifest_values,artifact_evidence = [],[],[]
    for side,summary_path,manifest_path,report_path in zip(('reference','replay'),summaries,manifests,reports):
        artifacts = comparison[side+'_artifacts']
        for name,digest in artifacts.items():
            if sha(Path(name)) != digest: raise ValueError('Compared bias artifact changed')
        metadata_path = report_path.with_suffix('.metadata.json')
        for path in (manifest_path,report_path,metadata_path):
            if artifacts.get(str(path)) != sha(path):
                raise ValueError('Summary inputs are not the compared bias artifacts')
        artifact_evidence.append(artifacts)
        manifest,summary = read(manifest_path),read(summary_path)
        require_finite(summary)
        if (summary['schema'] != 1 or summary['training_runs'] != 100 or summary['score_files'] != 400
                or summary['verified_primary_score_files'] != 100 or set(summary['primary']) != set(expected_primary)
                or [(c['family'],c['endpoint']) for c in summary['contrasts']] != [
                    (family,endpoint) for family in expected_primary for endpoint in ('hits@10','mr')]
                or set(summary['checksums']) != {f'fold{f}' for f in range(10)}
                or summary['associations'] != comparison['association_pairs']
                or manifest['primary_comparisons'] != expected_primary):
            raise ValueError('Require all matched primary choices, folds and ten contrasts')
        provenance = summary['provenance']
        expected_provenance = dict(manifest_sha256=sha(manifest_path),report_sha256=sha(report_path),
            metadata_sha256=sha(metadata_path),freeze_sha256=manifest['freeze_sha256'],
            training_protocol_sha256=manifest['protocol_sha256'],analysis_protocol_sha256=sha(PROTOCOL),
            summary_source_sha256=sha(HERE.parent/'code/analysis/summarize_matched_kge.py'),
            evaluation_source_sha256=comparison['evaluator_sha256'],
            bootstrap_source_sha256=sha(HERE.parent/'code/analysis/summarize_controlled_query.py'))
        if provenance != expected_provenance:
            raise ValueError('Primary summary provenance differs from its verified inputs or implementation')
        models = {m['name']:m for m in manifest['models']}
        for fold in range(10):
            record,checked = summary['checksums'][f'fold{fold}'],rank_records[fold]
            if (checked['changed_ranks'] != 0 or checked['arrays'] != 560
                    or set(record) != {'rows_sha256','ranks_sha256','scores'}
                    or record['rows_sha256'] != checked['rows_sha256']
                    or record['ranks_sha256'] != checked[side+'_sha256']
                    or record['scores'] != {name:models[name]['folds'][str(fold)]['sha256'] for name in primary_names}):
                raise ValueError('Primary summary does not bind the compared ranks and primary scores')
        scientific = copy.deepcopy(summary)
        del scientific['checksums']
        for key in ('manifest_sha256','report_sha256','metadata_sha256'):
            del scientific['provenance'][key]
        normalized.append(scientific)
        manifest_values.append(manifest)
    if any(manifest_values[0][key] != manifest_values[1][key] for key in
           ('freeze_sha256','protocol_sha256','benchmark','primary_comparisons')):
        raise ValueError('Frozen experiment or benchmark differs')
    changed = differences(*normalized)
    for collection in [hashes,*artifact_evidence]:
        for path,digest in collection.items():
            if sha(Path(path)) != digest: raise ValueError('Comparison input changed during verification')
    return dict(status='requires_review' if changed else 'passed',training_runs=100,score_files=400,
        primary_score_files=100,primary_contrasts=10,summary_differences=changed,
        reference_summary_sha256=hashes[str(summaries[0])],replay_summary_sha256=hashes[str(summaries[1])],
        bias_comparison_sha256=hashes[str(comparison_path)],comparator_sha256=sha(__file__),
        scope='Complete matched primary summary agreement, including all ten unrounded bootstrap contrasts, selection epochs and graph/training settings. Each summary is bound to its own manifest/report and all-bias rank comparison. Summaries must be regenerated from their own scores; independent inference execution and score comparison remain separate requirements.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-summary','replay-summary','reference-manifest','replay-manifest',
                 'reference-report','replay-report','bias-comparison','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-sha256',required=True)
    args = vars(parser.parse_args()); output = args.pop('output')
    if output.exists(): raise FileExistsError('Choose a new comparison output path')
    result = compare(**args)
    write(output,result)
    print(result['status'],len(result['summary_differences']),'changed primary summary fields')
    if result['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
