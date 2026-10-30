#!/usr/bin/env python3
"""Render controlled-query tables from the fully verified report; no fitted results bundled."""
import argparse
import json
from pathlib import Path

INITS={'ultra_4g':'ULTRA','ultraquery':'UltraQuery'}
ARMS={'bridge_pool':'Disease bridges, Pool','plain_pool':'No disease bridges, Pool',
      'plain_and':'No disease bridges, AND','typed_and':'No disease bridges + typed EQ, AND'}
CONTRASTS={'disease_bridges':'Remove disease bridges (Pool)','conjunction':'AND versus Pool (no bridges)',
           'typed_EQ':'Add typed EQ (AND)','adaptation':'Adapt typed EQ AND versus initial'}


def value(record,metric):
    factor=100 if metric.startswith('hits@') else 1
    digits=1 if metric=='mr' else 3 if metric=='mrr' else 2
    mean=record['mean_over_seeds'][metric]*factor
    sd=record['sd_across_seeds'][metric]*factor
    return f'${mean:.{digits}f}\\pm {sd:.{digits}f}$'


def generate(report,out):
    if report['training_runs']!=240 or report['score_bundles']!=480 or len(report['primary_contrasts'])!=8:
        raise ValueError('Tables require the complete controlled comparison')
    out.mkdir(parents=True,exist_ok=True)
    n=report['associations'];clusters=report['profile_clusters']
    main=[r'\begin{table*}[tb]',r'\centering\small',
          r'\caption{Controlled phenotype-query comparison across ten folds and three seeds. All metrics use per-gene leave-one-test-disease-out calibration and association-pooled evaluation ('+f'{n:,}'+r' associations). Entries are means $\pm$ sample standard deviations across seeds. Initial and validation-selected checkpoints use the same graph and readout. Seed dispersion is distinct from the paired confidence intervals in Table~\ref{tab:controlled_query_contrasts}.}',
          r'\label{tab:controlled_query}',r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{llcccc}',r'\toprule',
          r'Initialization & Graph and readout & Initial H@10 (\%) & Selected H@10 (\%) & Selected H@1 (\%) & Selected MR \\',r'\midrule']
    for init,label in INITS.items():
        for arm,arm_label in ARMS.items():
            initial=report['conditions'][f'{init}/{arm}/initial/calibrated']
            selected=report['conditions'][f'{init}/{arm}/selected/calibrated']
            main.append(' & '.join([label,arm_label,value(initial,'hits@10'),value(selected,'hits@10'),value(selected,'hits@1'),value(selected,'mr')])+r' \\')
    main.extend([r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table*}'])
    (out/'controlled-query-main.tex').write_text('\n'.join(main)+'\n')
    contrasts=[r'\begin{table*}[tb]',r'\centering\small',
       r'\caption{Prespecified calibrated Hits@10 contrasts (percentage points). Each association\textquotesingle s hit contribution is averaged across seeds before 10,000 paired phenotype-profile-cluster bootstrap replicates ('+f'{clusters:,}'+r' profiles; seed 20261028). The eight comparisons form one family; intervals have nominal 99.375\% confidence (Bonferroni family $\alpha=0.05$). Association weights are retained within resampled profiles. Intervals describe benchmark sampling variation conditional on the fitted models; seed dispersion is reported separately.}',
       r'\label{tab:controlled_query_contrasts}',r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{llrrrr}',r'\toprule',
       r'Initialization & Contrast & Seed 0 & Seed 1 & Seed 2 & Mean [99.375\% interval] \\',r'\midrule']
    for c in report['primary_contrasts']:
        estimates=[f'{x:+.2f}' for x in c['per_seed_percentage_points']]
        estimates.append(f"{c['difference_percentage_points']:+.2f} [{c['lower_percentage_points']:+.2f}, {c['upper_percentage_points']:+.2f}]")
        contrasts.append(' & '.join([INITS[c['initialization']],CONTRASTS[c['contrast']],*estimates])+r' \\')
    contrasts.extend([r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table*}'])
    (out/'controlled-query-contrasts.tex').write_text('\n'.join(contrasts)+'\n')
    full=[r'\begin{landscape}']
    for init,label in INITS.items():
        full.extend([r'\begin{table*}[p]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{3pt}',
           r'\caption{Complete '+label+r'-initialized controlled-query results: native raw scores and per-gene calibrated scores, with both initial and validation-selected checkpoints. H@10 is given separately for all three seeds. Remaining columns give means $\pm$ sample standard deviations across seeds. All metrics pool the same '+f'{n:,}'+r' associations over ten folds.}',
           r'\label{tab:controlled_query_full_'+init+'}',r'\begin{adjustbox}{max width=\linewidth}',r'\begin{tabular}{lllrrrrrrr}',r'\toprule',
           r'Graph/readout & Checkpoint & Scoring & H@10 Seed 0 & H@10 Seed 1 & H@10 Seed 2 & H@10 (\%) & H@1 (\%) & MRR & MR \\',r'\midrule'])
        for arm,arm_label in ARMS.items():
            for kind in ['initial','selected']:
                for mode in ['raw','calibrated']:
                    r=report['conditions'][f'{init}/{arm}/{kind}/{mode}']
                    seed_values=[f"{r['per_seed'][str(s)]['hits@10']*100:.2f}" for s in range(3)]
                    full.append(' & '.join([arm_label,kind,mode,*seed_values,*[value(r,m) for m in ['hits@10','hits@1','mrr','mr']]])+r' \\')
        full.extend([r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',r'\end{table*}',r'\clearpage'])
    full.append(r'\end{landscape}')
    (out/'controlled-query-full.tex').write_text('\n'.join(full)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-json',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();generate(json.loads(a.report_json.read_text()),a.out_dir)


if __name__=='__main__':main()
