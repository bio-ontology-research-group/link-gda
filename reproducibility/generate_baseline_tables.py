"""Export all 26 completed baseline variants; historical partial reports are rejected."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


PHRANK = [
    ('Phrank (Native HPO Direct Raw)', 'Native HPO, direct'),
    ('Phrank (Native HPO Disease-Max Raw)', 'Native HPO, disease-max'),
    *[(f'Phrank-HPO ({mode}_{kind})', f'Human-only UPheno, {label}, {suffix}')
      for mode, label in [('direct', 'direct'), ('disease', 'disease-max')]
      for kind, suffix in [('raw', 'raw'), ('normalized', 'normalized')]],
    *[(f'Phrank-UPheno (Human+MGI {mode} {kind})', f'Human+MGI UPheno, {label}, {suffix}')
      for mode, label in [('Direct', 'direct'), ('Disease', 'disease-max')]
      for kind, suffix in [('Raw', 'raw'), ('Norm', 'normalized')]],
    ('Phrank-UPheno (MGI-Only Raw)', 'MGI-only UPheno, raw'),
    ('Phrank-UPheno (MGI-Only Norm)', 'MGI-only UPheno, normalized'),
]
PHENORANK = [
    ('PhenoRank (Native 2016 1000-Sim)', 'Native 2016, simulation'),
    ('PhenoRank (Native 2016 NoSim)', 'Native 2016, continuous'),
    *[(f'PhenoRank-UPheno ({source} {kind})', f'UPheno {label}, {suffix}')
      for source, label in [('HM-Add+PPI', 'H+MGI additive, PPI'),
                            ('HM-Bal+PPI', 'H+MGI balanced, PPI'),
                            ('Mouse+PPI', 'MGI-only, PPI'),
                            ('Mouse-NoPPI', 'MGI-only, no PPI')]
      for kind, suffix in [('NoSim', 'continuous'), ('1000-Sim', 'simulation'),
                           ('1000-Sim TB', 'simulation + TB')]],
]
PANELS = [('phrank', 'Phrank', PHRANK), ('phenorank', 'PhenoRank', PHENORANK)]
COHORTS = dict(pairs=6571, diseases=6059, seen_pairs=3085, zero_overlap_pairs=4431,
               training_profile_matches=210, training_profile_matches_in_zero_overlap=166,
               combined_pairs=4265, combined_diseases=4017, combined_profiles=4011)
COLUMNS = [('S0_Raw_All', 'S0: raw', 6571), ('S1_Cal_All', 'S1: calibrated', 6571),
           ('S2_StatusZ_All', 'S2: status-Z', 6571),
           ('S3_ZeroDirectOverlap_Raw', 'Zero overlap: raw', 4431),
           ('S3_ZeroDirectOverlap_StatusZ', 'S3: status-Z', 4431),
           ('S4_Cumulative_All4', 'S4: combined', 4011)]
METRICS = ('h1', 'h10', 'mrr', 'mr', 'med', 'n')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate(report, metadata, manifest, manifest_digest):
    expected = {key for _, _, panel in PANELS for key, _ in panel}
    models = manifest['models']
    if (manifest.get('all_baseline_components') is not True
            or set(manifest['components']) != {'native-phrank', 'native-phrank-disease-max',
                'native-phenorank', 'upheno-phrank', 'upheno-phenorank'}
            or len(models) != 26 or {m['name'] for m in models} != expected
            or set(report) != expected or set(metadata['models']) != expected):
        raise ValueError('Require all 26 final baseline variants and all five components')
    folds = list(range(10))
    if (manifest['folds'] != folds or metadata['folds'] != folds
            or any(set(m['folds']) != {str(f) for f in folds} for m in models)):
        raise ValueError('Require all ten folds')
    if metadata['score_manifest_sha256'] != manifest_digest:
        raise ValueError('Evaluation used a different score manifest')
    if metadata['cohorts'] != COHORTS:
        raise ValueError('Evaluation cohorts differ from the paper benchmark')
    conditions = set(next(iter(report.values())))
    if len(conditions) != 42:
        raise ValueError('Require all 42 scoring/cohort conditions')
    populations = {condition: next(iter(report.values()))[condition]['n'] for condition in conditions}
    for model, values in report.items():
        if set(values) != conditions:
            raise ValueError(f'Conditions differ for {model}')
        for condition, stats in values.items():
            if any(not isinstance(stats[m], (int, float)) or isinstance(stats[m], bool)
                   or not math.isfinite(stats[m]) for m in METRICS):
                raise ValueError(f'Non-finite or missing metrics: {model}/{condition}')
            if (not 0 <= stats['h1'] <= stats['h10'] <= 100 or not 0 < stats['mrr'] <= 1
                    or not 1 <= stats['mr'] <= 13625 or not 1 <= stats['med'] <= 13625
                    or not isinstance(stats['n'], int) or stats['n'] < 1
                    or stats['n'] != populations[condition]):
                raise ValueError(f'Invalid metrics or population: {model}/{condition}')
        for condition, _, n in COLUMNS:
            if values[condition]['n'] != n:
                raise ValueError(f'Unexpected population: {model}/{condition}')


def render(report, metadata, manifest, manifest_digest, report_digest):
    validate(report, metadata, manifest, manifest_digest)
    outputs = {}
    for slug, title, panel in PANELS:
        note = (r'Native HPO uses the author implementation with direct-gene and disease-max readouts. '
                r'UPheno variants use the author-compatible parentless-term policy; normalized scores '
                r'are additional adaptations.' if slug == 'phrank' else
                r'Native PhenoRank uses the retained 2016 resources. UPheno variants use refitted '
                r'annotation sources and the published tier sampler. Simulation variants use 1,000 '
                r'uncapped simulations per query; continuous variants omit simulation correction. '
                r'TB denotes the explicit continuous-score tie-break adaptation.')
        caption = (title + r' baseline results across ten folds and 13,625 candidate genes. '
                   r'Cells give H@10 (\%)/MR, pooling associations across folds. S0--S2 use 6,571 '
                   r'pairs; the zero-overlap columns use 4,431 pairs. S1 applies per-gene calibration; '
                   r'S2 additionally standardizes training-status groups. S3 restricts S2 to zero '
                   r'direct same-gene phenotype overlap. S4 additionally deduplicates calibration '
                   r'profiles, excludes training-profile matches, filters co-causal genes and weights '
                   r'the remaining 4,011 test profiles equally. These changes concern different '
                   r'scoring rules and populations; they are not independent causal bias effects. ' + note)
        lines = [f'% Report SHA256: {report_digest}', f'% Score manifest SHA256: {manifest_digest}',
                 r'\begin{landscape}', r'\begin{table*}[p]', r'\centering\footnotesize',
                 r'\setlength{\tabcolsep}{3pt}', r'\caption{' + caption + '}',
                 r'\label{tab:final_baseline_' + slug + '}',
                 r'\begin{adjustbox}{max width=\linewidth}', r'\begin{tabular}{lrrrrrr}',
                 r'\toprule', 'Variant & ' + ' & '.join(label for _, label, _ in COLUMNS) + r' \\',
                 r'\midrule']
        for key, label in panel:
            cells = [f"{report[key][condition]['h10']:.2f} / {report[key][condition]['mr']:.1f}"
                     for condition, _, _ in COLUMNS]
            lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table*}',
                      r'\clearpage', r'\end{landscape}', ''])
        outputs[f'final-baseline-{slug}.tex'] = '\n'.join(lines)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['model', 'condition', *METRICS])
    for _, _, panel in PANELS:
        for key, _ in panel:
            for condition, stats in sorted(report[key].items()):
                writer.writerow([key, condition, *(stats[m] for m in METRICS)])
    outputs['final-baseline-metrics.csv'] = stream.getvalue()
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-json', type=Path, required=True)
    p.add_argument('--score-manifest', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    report_bytes, manifest_bytes = a.report_json.read_bytes(), a.score_manifest.read_bytes()
    metadata_path = a.report_json.with_suffix('.metadata.json')
    metadata_bytes = metadata_path.read_bytes()
    outputs = render(json.loads(report_bytes), json.loads(metadata_bytes), json.loads(manifest_bytes),
                     sha(manifest_bytes), sha(report_bytes))
    proof = dict(report_sha256=sha(report_bytes), metadata_sha256=sha(metadata_bytes),
                 manifest_sha256=sha(manifest_bytes), generator_sha256=sha(Path(__file__).read_bytes()),
                 variants=26, conditions=42,
                 outputs={name: sha(value.encode()) for name, value in outputs.items()})
    outputs['final-baseline-export.json'] = json.dumps(proof, indent=2) + '\n'
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        path = a.out_dir / name
        if path.exists() and path.read_bytes() != value.encode():
            raise ValueError(f'Existing output differs; choose a new directory: {path}')
    for name, value in outputs.items():
        (a.out_dir / name).write_bytes(value.encode())
    print('Exported 26 baseline variants, all 42 conditions, and two LaTeX tables')


if __name__ == '__main__':
    main()
