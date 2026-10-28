"""Render the individually applied S1 sensitivities from the score-based report."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ('PhenoQuery (typed_and)', r'PhenoQuery, typed conjunction'),
    ('ULTRA (No Disease Bridges)', r'ULTRA, fixed disease-edge ablation'),
    ('LinkGDA-T1 (Retrained No-Bridge, Raw-Head)', r'\method{}-T1, no-bridge, Raw-Head'),
    ('Exomiser-Phive', r'Exomiser-Phive'),
]
OPERATIONS = [
    ('S1_Cal_All', 'S1 reference', 'pairs'),
    ('S1_ReferenceDedup_All', 'Deduplicate reference profiles', 'pairs'),
    ('S1_PositiveFilter_All', 'Filter other causal genes', 'pairs'),
    ('S1_EqualProfile_All', 'Weight test profiles equally', 'profiles'),
    ('S1_NoTrainingProfileMatch', 'Exclude training-profile matches', 'pairs'),
]


def render(report, digest):
    lines = [f'% Input report SHA256: {digest}', r'\begin{table*}[tb]',
             r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
             r'\caption{Individual evaluation changes applied to the full S1 calibrated panel. Each row changes one operation relative to S1 for the same model. Reference deduplication changes the calibration panel; co-causal filtering changes candidate competition; equal-profile weighting changes the contribution of repeated test profiles. Training-profile exclusion changes the evaluated population from 6,571 to 6,361 association pairs. All other rows retain those 6,571 pairs; the equal-profile row groups them into 5,928 profiles. Positive $\Delta$H@10 and negative $\Delta$MR indicate improvement relative to S1.}',
             r'\label{tab:bias_individual_factors}',
             r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Operation & H@10 (\%) & MR & $\Delta$H@10 (pp) & $\Delta$MR \\',
             r'\midrule']
    for i, (key, label) in enumerate(MODELS):
        model = report[key]
        baseline = model['S1_Cal_All']
        lines.append(r'\multicolumn{5}{l}{\emph{' + label + r'}} \\')
        for j, (condition, operation, unit) in enumerate(OPERATIONS):
            value = model[condition]
            expected_n = 5928 if unit == 'profiles' else 6361 if condition == 'S1_NoTrainingProfileMatch' else 6571
            if value['n'] != expected_n:
                raise ValueError(f'Unexpected population for {key}: {condition}')
            lines.append(f"{operation} & {value['h10']:.2f} & "
                         f"{value['mr']:.1f} & {value['h10']-baseline['h10']:+.2f} & "
                         f"{value['mr']-baseline['mr']:+.1f} " + r'\\')
        if i < len(MODELS)-1: lines.append(r'\midrule')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-json', type=Path, default=ROOT/'reproducibility/results/stepwise-bias-report-v4.json')
    p.add_argument('--output', type=Path)
    p.add_argument('--out-dir', type=Path, default=ROOT/'reproducibility/tables')
    args = p.parse_args()
    args.output = args.output or args.out_dir/'revised-bias-factors.tex'
    data = args.report_json.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(json.loads(data), hashlib.sha256(data).hexdigest()))


if __name__ == '__main__': main()
