"""Summarize fixed query variants across seeds for the main bias comparison."""
import argparse
import json
from pathlib import Path
import statistics

from generate_controlled_query_tables import INITS, ARMS
from generate_production_bias_tables import validate, sha

CONDITIONS = (
    ('S1_Cal_All', 'Full, calibrated'),
    ('S3_ZeroDirectOverlap_Cal', 'Zero overlap, calibrated'),
    ('S3_ZeroDirectOverlap_StatusZ', 'Zero overlap, status adjusted'),
    ('S4_Cumulative_All4', 'Combined S4'),
)


def render(report, metadata, manifest, manifest_digest, report_digest):
    kind, _, _ = validate(report, metadata, manifest, manifest_digest)
    if kind != 'controlled':
        raise ValueError('This summary requires the complete controlled query campaign')
    summaries = {}
    lines = [f'% Report SHA256: {report_digest}', f'% Score manifest SHA256: {manifest_digest}',
        r'\begin{table*}[tb]', r'\centering\small',
        r'\caption{Controlled-query sensitivity to phenotype overlap and score adjustment. '
        r'Entries give Hits@10 (\%), mean $\pm$ sample standard deviation across three seeds, '
        r'for the validation-selected checkpoint of each graph and readout. The full calibrated '
        r'column uses 6,571 associations; both zero-overlap columns use the same 4,431 associations. '
        r'Status adjustment standardizes calibrated scores within the seen and unseen candidate groups. '
        r'The combined S4 protocol additionally deduplicates calibration-reference profiles, excludes '
        r'training-profile matches, filters co-causal competitors and weights 4,011 profiles equally. '
        r'Seed dispersion summarizes the fitted runs; the columns describe their performance under '
        r'the specified scoring and evaluation populations.}',
        r'\label{tab:controlled_bias_summary}', r'\begin{adjustbox}{max width=\linewidth}',
        r'\begin{tabular}{llrrrr}', r'\toprule',
        'Initialization & Graph and readout & '+' & '.join(label for _,label in CONDITIONS)+r' \\',
        r'\midrule']
    for initialization, label in INITS.items():
        for arm, arm_label in ARMS.items():
            key = f'{initialization}/{arm}'
            summaries[key] = {}
            cells = []
            for condition, _ in CONDITIONS:
                records = [report[f'ControlledQuery ({initialization}, seed{seed}, {arm}, selected)'][condition]
                           for seed in range(3)]
                values = [record['h10'] for record in records]
                mean, sd = statistics.mean(values), statistics.stdev(values)
                summaries[key][condition] = dict(per_seed_h10=values, mean_h10=mean, sd_h10=sd,
                    population=records[0]['n'], unit='profiles' if condition == 'S4_Cumulative_All4' else 'association_pairs')
                cells.append(f'${mean:.2f}\\pm {sd:.2f}$')
            lines.append(' & '.join([label,arm_label,*cells])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table*}'])
    return '\n'.join(lines)+'\n', summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-json', type=Path, required=True)
    parser.add_argument('--score-manifest', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    report_bytes, manifest_bytes = args.report_json.read_bytes(), args.score_manifest.read_bytes()
    metadata_bytes = args.report_json.with_suffix('.metadata.json').read_bytes()
    report, manifest, metadata = map(json.loads, (report_bytes,manifest_bytes,metadata_bytes))
    table, summary = render(report,metadata,manifest,sha(manifest_bytes),sha(report_bytes))
    proof = dict(scope='Descriptive seed summary of the complete verified bias report; no new inferential contrasts.',
        report_sha256=sha(report_bytes), manifest_sha256=sha(manifest_bytes), metadata_sha256=sha(metadata_bytes),
        generator_sha256=sha(Path(__file__).read_bytes()), table_sha256=sha(table.encode()), conditions=summary)
    outputs = {'controlled-query-bias-summary.tex':table,
               'controlled-query-bias-summary.json':json.dumps(proof,indent=2)+'\n'}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for name,value in outputs.items():
        path = args.out_dir/name
        if path.exists() and path.read_text() != value:
            raise ValueError('Existing output differs; choose a new output directory')
    for name,value in outputs.items():
        (args.out_dir/name).write_text(value)


if __name__ == '__main__':
    main()
