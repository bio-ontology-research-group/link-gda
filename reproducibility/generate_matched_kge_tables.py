"""Render the fixed primary matched KGE comparisons and actual training budgets."""
import argparse
import hashlib
import json
from pathlib import Path

FAMILIES = ['U1', 'T1', 'INDIGENA', 'CONVKB', 'PPOOL']
LABELS = dict(U1='U1', T1='T1', INDIGENA='INDIGENA', CONVKB='ConvKB-D', PPOOL='P-pool')


def interval(record, digits):
    return f"{record['difference']:+.{digits}f} [{record['lower']:+.{digits}f}, {record['upper']:+.{digits}f}]"


def span(values):
    lo, hi = min(values), max(values)
    return f'{lo:,}' if lo == hi else f'{lo:,}--{hi:,}'


def render(report, digest):
    expected = {(family, metric) for family in FAMILIES for metric in ['hits@10', 'mr']}
    contrasts = {(r['family'], r['endpoint']): r for r in report['contrasts']}
    if (report['training_runs'] != 100 or report['score_files'] != 400
            or report['verified_primary_score_files'] != 100 or set(report['primary']) != set(FAMILIES)
            or len(report['contrasts']) != 10 or set(contrasts) != expected
            or report['inference']['confidence_level'] != .995 or report['inference']['comparisons'] != 10
            or report['inference']['replicates'] != 10000 or report['inference']['seed'] != 20261029):
        raise ValueError('Require the complete fixed matched KGE analysis')
    caption = (r'Matched graph-construction sensitivity across ten folds, one seed and '
        + f"{report['associations']:,}" + r' associations. Each family uses its prespecified primary '
        r'validation choice and scoring mode. H@10 is in percent and its differences are in '
        r'percentage points. Differences are no-bridge minus original: positive '
        r'H@10 and negative MR indicate improvement. Brackets give 99.5\% paired phenotype-profile-cluster '
        r'bootstrap intervals (10,000 replicates; seed 20261029), with nominal Bonferroni 95\% '
        r'family coverage across five families and two endpoints. These intervals describe benchmark '
        r'sampling variation conditional on the fitted models, not training-seed uncertainty. '
        r'The benchmark was reused during method development.')
    main = [f'% Summary SHA256: {digest}', r'\begin{table*}[tb]', r'\centering\small',
        r'\caption{' + caption + '}', r'\label{tab:matched_kge_primary}',
        r'\begin{adjustbox}{max width=\linewidth}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
        r'Family (scoring) & Original H@10 & No-bridge H@10 & $\Delta$H@10 [interval] & Original MR & No-bridge MR & $\Delta$MR [interval] \\',
        r'\midrule']
    for family in FAMILIES:
        row = report['primary'][family]
        left, right = [row['graphs'][g]['metrics'] for g in ['original', 'nobridge']]
        main.append(' & '.join([LABELS[family] + ' (' + row['scoring'] + ')', f"{left['h10']:.2f}",
            f"{right['h10']:.2f}", interval(contrasts[(family, 'hits@10')], 2), f"{left['mr']:.1f}",
            f"{right['mr']:.1f}", interval(contrasts[(family, 'mr')], 1)]) + r' \\')
    main.extend([r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table*}', ''])
    design = [f'% Summary SHA256: {digest}', r'\begin{table*}[tb]', r'\centering\small',
        r'\caption{Matched KGE graph sizes and training budgets. Ranges span the ten folds. '
        r'Forward triples exclude the inverses added by the trainer. The no-bridge graph removes '
        r'training-disease edges and rebuilds the embedding inventory; it is not a fixed-parameter '
        r'inference-time mask. Equal epoch budgets do not imply equal gradient-update counts or FLOPs. '
        r'P-pool epochs follow a warm start from the same graph\textquotesingle s U1 raw-head checkpoint; '
        r'ConvKB-D starts from U1\textquotesingle s calibrated-MR checkpoint.}',
        r'\label{tab:matched_kge_design}', r'\begin{adjustbox}{max width=\linewidth}',
        r'\begin{tabular}{llrrrrr}', r'\toprule',
        r'Family & Graph & Dimension & Nodes & Forward triples & Training epochs & Selected epoch \\', r'\midrule']
    for family in FAMILIES:
        for graph in ['original', 'nobridge']:
            row = report['primary'][family]['graphs'][graph]
            if any(len(row[k]) != 10 for k in ['nodes', 'forward_triples', 'epochs', 'selected_epochs']):
                raise ValueError('Missing fold-level design records')
            design.append(' & '.join([LABELS[family], 'Original' if graph=='original' else 'No bridge',
                str(row['dimension']), *[span(row[k]) for k in ['nodes', 'forward_triples', 'epochs', 'selected_epochs']]]) + r' \\')
    design.extend([r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table*}', ''])
    return {'matched-kge-primary.tex': '\n'.join(main), 'matched-kge-design.tex': '\n'.join(design)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-json', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args(); data = a.report_json.read_bytes()
    outputs = render(json.loads(data), hashlib.sha256(data).hexdigest())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        path = a.out_dir / name
        if path.exists() and path.read_text() != value:
            raise ValueError(f'Existing table differs; choose a new directory: {path}')
    for name, value in outputs.items(): (a.out_dir / name).write_text(value)


if __name__ == '__main__': main()
