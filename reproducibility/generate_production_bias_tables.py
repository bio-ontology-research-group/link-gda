"""Render complete controlled-query or matched-KGE bias reports without pooling seeds."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys

from generate_baseline_tables import COHORTS, COLUMNS, METRICS, sha
from generate_controlled_query_tables import INITS, ARMS
from generate_factor_table import OPERATIONS
from generate_matched_kge_tables import FAMILIES, LABELS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from training.run_matched_kge import PRIMARY

# Fixed benchmark populations; n counts profiles only for PROFILE_CONDITIONS.
POPULATIONS = {
    'S1_ReferenceDedup_All': 6571, 'S2_ReferenceDedup_All': 6571,
    'S1_PositiveFilter_All': 6571, 'S2_PositiveFilter_All': 6571,
    'S1_EqualProfile_All': 5928, 'S2_EqualProfile_All': 5928,
    'S1_NoTrainingProfileMatch': 6361, 'S2_NoTrainingProfileMatch': 6361,
    'S3_ReferenceDedup': 4431, 'S3_PositiveFilter': 4431,
    'S3_EqualProfile': 4061, 'S3_NoTrainingProfileMatch': 4265,
    'S0_Raw_All': 6571, 'S0_Raw_Seen': 3085, 'S0_Raw_Unseen': 3486,
    'S0_Raw_Seen_Jgt0': 2140, 'S0_Raw_Seen_J0': 945,
    'S1_Cal_All': 6571, 'S1_Cal_Seen': 3085, 'S1_Cal_Unseen': 3486,
    'S1_Cal_Seen_Jgt0': 2140, 'S1_Cal_Seen_J0': 945,
    'S2_StatusZ_All': 6571, 'S2_StatusZ_Seen': 3085, 'S2_StatusZ_Unseen': 3486,
    'S2_RawStatusZ_All': 6571, 'S2_RawStatusZ_Seen': 3085, 'S2_RawStatusZ_Unseen': 3486,
    'S2_WithinStatus_Seen': 3085, 'S2_WithinStatus_Unseen': 3486,
    'S2_RawWithinStatus_Seen': 3085, 'S2_RawWithinStatus_Unseen': 3486,
    'S3_ZeroDirectOverlap_StatusZ': 4431, 'S3_ZeroDirectOverlap_RawStatusZ': 4431,
    'S3_ZeroDirectOverlap_Cal': 4431, 'S3_ZeroDirectOverlap_Raw': 4431,
    'S3_ZeroDirectOverlap_Seen_J0': 945, 'S3_ZeroDirectOverlap_Unseen': 3486,
    'S4_Cumulative_All4': 4011, 'S4_RawCumulative_NoS1': 4011,
    'S4_Isolated_Dedup_Filt_EqD_on_S1': 5878, 'S4_Isolated_Dedup_Filt_EqD_on_S0': 5878,
}
PROFILE_CONDITIONS = {'S1_EqualProfile_All', 'S2_EqualProfile_All', 'S3_EqualProfile',
    'S4_Cumulative_All4', 'S4_RawCumulative_NoS1',
    'S4_Isolated_Dedup_Filt_EqD_on_S1', 'S4_Isolated_Dedup_Filt_EqD_on_S0'}
CHOICES = ('raw_mr', 'raw_head', 'calibrated_mr', 'calibrated_head')
CHOICE_LABELS = dict(raw_mr='raw MR', raw_head='raw head',
                     calibrated_mr='calibrated MR', calibrated_head='calibrated head')


def design(manifest):
    """Return the full fixed design and display panels, independent of performance."""
    panels, expected = [], {}
    if manifest.get('training_runs') == 240 and manifest.get('score_files') == 480:
        kind, protocol = 'controlled', 'controlled-query-20261028.md'
        for initialization, label in INITS.items():
            for seed in range(3):
                rows = []
                for arm, arm_label in ARMS.items():
                    for checkpoint in ('initial', 'selected'):
                        name = f'ControlledQuery ({initialization}, seed{seed}, {arm}, {checkpoint})'
                        expected[name] = dict(initialization=initialization, seed=seed,
                                              arm=arm, checkpoint_kind=checkpoint)
                        rows.append((name, arm_label + ', ' + checkpoint))
                panels.append((f'{initialization}-seed{seed}', f'{label} initialization, seed {seed}', rows))
    elif manifest.get('training_runs') == 100 and manifest.get('score_files') == 400:
        kind, protocol = 'matched', 'matched-kge-20261029.md'
        for family in FAMILIES:
            rows = []
            for graph in ('original', 'nobridge'):
                for choice in CHOICES:
                    name = f'MatchedKGE ({family}, {graph}, {choice})'
                    expected[name] = dict(family=family, graph=graph,
                        validation_choice=choice, primary=choice == PRIMARY[family])
                    label = ('Original' if graph == 'original' else 'No bridge')
                    label += ', ' + CHOICE_LABELS[choice]
                    if choice == PRIMARY[family]: label += r' ($\ast$)'
                    rows.append((name, label))
            panels.append((family, LABELS[family], rows))
    else:
        raise ValueError('Require the complete 240-run query or 100-run matched campaign')
    if manifest.get('protocol_sha256') != sha((ROOT/'reproducibility/protocols'/protocol).read_bytes()):
        raise ValueError('Campaign protocol differs')
    return kind, expected, panels


def validate(report, metadata, manifest, manifest_digest):
    kind, expected, panels = design(manifest)
    models = manifest['models']
    if (manifest.get('schema') != 1 or len(models) != len(expected)
            or {m['name'] for m in models} != set(expected)
            or set(report) != set(expected) or set(metadata['models']) != set(expected)
            or len(metadata['models']) != len(expected)):
        raise ValueError('Missing, duplicate or unexpected campaign variants')
    if (manifest['folds'] != list(range(10)) or metadata['folds'] != list(range(10))
            or any(set(m['folds']) != {str(f) for f in range(10)} for m in models)):
        raise ValueError('Require all ten folds for every variant')
    if (metadata['score_manifest_sha256'] != manifest_digest
            or metadata['evaluation_sha256'] != sha((ROOT/'code/analysis/evaluate_stepwise_bias.py').read_bytes())):
        raise ValueError('Report provenance differs from the manifest or evaluator')
    if metadata['cohorts'] != COHORTS:
        raise ValueError('Report cohorts differ from the paper benchmark')
    for model in models:
        if any(type(model.get(k)) is not type(v) or model.get(k) != v
               for k, v in expected[model['name']].items()):
            raise ValueError('Model label and experimental identity disagree')
        values = report[model['name']]
        if set(values) != set(POPULATIONS):
            raise ValueError('Require the exact 42 scoring/cohort conditions')
        for condition, stats in values.items():
            if (set(stats) != set(METRICS)
                    or any(isinstance(stats[m], bool) or not isinstance(stats[m], (int,float))
                           or not math.isfinite(stats[m]) for m in METRICS)):
                raise ValueError('Missing, unexpected or non-finite metrics')
            if (not 0 <= stats['h1'] <= stats['h10'] <= 100 or not 0 < stats['mrr'] <= 1
                    or not 1 <= stats['mr'] <= 13625 or not 1 <= stats['med'] <= 13625
                    or not isinstance(stats['n'], int) or stats['n'] != POPULATIONS[condition]):
                raise ValueError('Invalid metric range or evaluation population')
    return kind, expected, panels


def tabular(report, rows, columns):
    lines = [r'\begin{adjustbox}{max width=\linewidth}',
        r'\begin{tabular}{l' + 'r'*len(columns) + '}', r'\toprule',
        'Configuration & ' + ' & '.join(label for _,label in columns) + r' \\', r'\midrule']
    for name, label in rows:
        cells = [f"{report[name][condition]['h10']:.2f} / {report[name][condition]['mr']:.1f}"
                 for condition, _ in columns]
        lines.append(' & '.join([label, *cells]) + r' \\')
    return lines + [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}']


def render(report, metadata, manifest, manifest_digest, report_digest):
    kind, identities, panels = validate(report, metadata, manifest, manifest_digest)
    outputs = {}
    for slug, title, rows in panels:
        note = (r'Each seed is reported separately, with both initial and validation-selected checkpoints. '
                if kind == 'controlled' else
                r'Rows retain all four validation choices for each graph; $\ast$ marks the prespecified primary. '
                r'Every scoring column evaluates that row\textquotesingle s fixed checkpoint. ')
        caption = (title + r': bias sensitivities over ten folds and 13,625 candidates. '
            r'Cells give H@10 (\%)/MR. ' + note +
            r'Panel A: S0 uses raw scores; S1 applies per-gene calibration; S2 adds within-query '
            r'training-status standardization. S0--S2 use 6,571 association pairs. Zero-overlap '
            r'columns use 4,431 pairs; S4 combines reference deduplication, training-profile exclusion, '
            r'co-causal filtering and equal-profile weighting (4,011 profiles). '
            r'Panel B changes one operation relative to S1: reference deduplication and co-causal '
            r'filtering retain 6,571 pairs, equal weighting uses their 5,928 profiles, and training-profile '
            r'exclusion retains 6,361 pairs. These scoring, competition, weighting and population changes '
            r'are descriptive sensitivities under the specified protocols. The accompanying CSV '
            r'retains every variant, all 42 conditions, exact metric values and population units.')
        lines = [f'% Report SHA256: {report_digest}', f'% Score manifest SHA256: {manifest_digest}',
            r'\begin{landscape}', r'\begin{table*}[p]', r'\centering\footnotesize',
            r'\setlength{\tabcolsep}{3pt}', r'\caption{' + caption + '}',
            r'\label{tab:' + kind + '_bias_' + slug.replace('_','-') + '}',
            r'\textbf{Panel A: scoring adjustments and evaluation cohorts}\par\smallskip']
        lines += tabular(report, rows, [(key,label) for key,label,_ in COLUMNS])
        lines += [r'\par\medskip', r'\textbf{Panel B: individual changes relative to S1}\par\smallskip']
        factor_headers = [r'S1 reference', r'\shortstack{Deduplicate\\reference profiles}',
            r'\shortstack{Filter other\\causal genes}', r'\shortstack{Equal profile\\weights}',
            r'\shortstack{Exclude training-\\profile matches}']
        factors = [(operation[0],label) for operation,label in zip(OPERATIONS,factor_headers)]
        lines += tabular(report, rows, factors)
        lines += [r'\end{table*}', r'\clearpage', r'\end{landscape}', '']
        outputs[f'{kind}-bias-{slug}.tex'] = '\n'.join(lines)
    stream = io.StringIO(newline=''); writer = csv.writer(stream)
    identity_columns = ('initialization','seed','arm','checkpoint_kind') if kind == 'controlled' else (
        'family','graph','validation_choice','primary')
    writer.writerow(['model', *identity_columns, 'condition','population_unit',*METRICS])
    for _,_,rows in panels:
        for name,_ in rows:
            for condition, stats in sorted(report[name].items()):
                writer.writerow([name, *(identities[name][k] for k in identity_columns), condition,
                    'profiles' if condition in PROFILE_CONDITIONS else 'association_pairs',
                    *(stats[m] for m in METRICS)])
    outputs[f'{kind}-bias-metrics.csv'] = stream.getvalue()
    return kind, outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-json', type=Path, required=True)
    p.add_argument('--score-manifest', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    report_bytes, manifest_bytes = args.report_json.read_bytes(), args.score_manifest.read_bytes()
    metadata_bytes = args.report_json.with_suffix('.metadata.json').read_bytes()
    report, manifest, metadata = map(json.loads, (report_bytes,manifest_bytes,metadata_bytes))
    kind, outputs = render(report, metadata, manifest, sha(manifest_bytes), sha(report_bytes))
    proof = dict(scope='Report rendering and provenance checks; numerical score/inference verification is separate.',
        variants=len(report), conditions=42, report_sha256=sha(report_bytes),
        metadata_sha256=sha(metadata_bytes), manifest_sha256=sha(manifest_bytes),
        generator_sha256=sha(Path(__file__).read_bytes()),
        outputs={name:sha(value.encode()) for name,value in outputs.items()})
    outputs[f'{kind}-bias-export.json'] = json.dumps(proof,indent=2)+'\n'
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for name,value in outputs.items():
        path = args.out_dir/name
        if path.exists() and path.read_bytes() != value.encode():
            raise ValueError(f'Existing output differs; choose a new directory: {path}')
    for name,value in outputs.items(): (args.out_dir/name).write_bytes(value.encode())
    print(f'Rendered {len(report)} variants and all 42 conditions; seeds and choices remain separate.')


if __name__ == '__main__': main()
