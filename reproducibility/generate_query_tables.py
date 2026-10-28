#!/usr/bin/env python3
"""Render the three historical query tables from verified score-derived statistics."""
import argparse
import json
from pathlib import Path

MAIN_ROWS = [
    ('general/original_pool_frozen','calibrated'),
    ('general/original_pool_adapted','calibrated'),('general/original_pool_adapted','raw'),
    ('general/original_and_adapted','calibrated'),
    ('general/augmented_pool_adapted','calibrated'),('general/augmented_and_adapted','calibrated'),
    ('query/original_pool_frozen','raw'),('query/original_pool_frozen','calibrated'),
    ('query/original_pool_adapted','raw'),('query/original_pool_adapted','calibrated'),
    ('query/original_and_adapted','raw'),
    ('query/augmented_pool_adapted','raw'),('query/augmented_pool_adapted','calibrated'),
    ('query/augmented_and_frozen','calibrated'),
    ('query/augmented_and_adapted','calibrated'),('query/augmented_and_adapted','raw')]
EQ_MAIN = ['bridge_pool','plain_pool','plain_and','entity_and','typed_and']
DIAGNOSTIC = [co+'/'+arm for co in ['general','query'] for arm in
               ['original_pool_frozen','original_pool_adapted','augmented_pool_adapted','augmented_and_adapted']]
STRATA = [('seen_high',r'Seen, $J_{\mathrm{same}}\geq0.5$'),
          ('seen_medium',r'Seen, $0.2\leq J_{\mathrm{same}}<0.5$'),
          ('seen_low',r'Seen, $0<J_{\mathrm{same}}<0.2$'),
          ('seen_zero',r'Seen, $J_{\mathrm{same}}=0$'),
          ('unseen','Unseen genes'),('multi_gene','Multi-gene diseases'),
          ('training_profile_match','Exact training-profile match')]


def label(key):
    cohort,arm = key.split('/')
    if cohort=='phenoquery':return r'PhenoQuery, \texttt{'+arm.replace('_',r'\_')+'}'
    graph,readout,stage=arm.split('_')
    return f"{'ULTRA' if cohort=='general' else 'UltraQuery'}, {'original' if graph=='original' else 'EQ-aug.'}, {readout.upper() if readout=='and' else 'Pool'}, {stage}"


def mode_label(record,mode,cohort):
    return ('Raw' if mode=='raw' else 'Cal.\\') + (' (Primary)' if cohort!='phenoquery' and record['primary_mode']==mode else '')


def row(key,mode,report,full=False):
    record=report['results'][key]; m=record['modes'][mode]; f=m['equal_fold']
    values=[label(key),mode_label(record,mode,key.split('/')[0]),f"{f['mr']:.2f}"+r'\std{'+f"{m['fold_mr_sd']:.2f}"+'}']
    if full:values[0]=values[0].split(', ',1)[1]
    if full:values.append(f"{m['pooled']['mr']:.2f}")
    values += [f"{m['median']:g}",f"{f['mrr']:.3f}",f"{100*f['hits@1']:.2f}",f"{100*f['hits@10']:.2f}"]
    if full:values.append(f"{100*f['hits@100']:.2f}")
    values += [f"{m['seen']['mr']:.2f}",f"{m['unseen']['mr']:.2f}"]
    return ' & '.join(values)+r' \\'


def table_start(caption,reference,columns,size='small',adjust=False):
    lines=[r'\begin{table*}[tb]','\\centering\\'+size,r'\caption{'+caption+'}',r'\label{'+reference+'}']
    if adjust:lines.append(r'\begin{adjustbox}{max width=\linewidth}')
    return lines+[r'\begin{tabular}{'+columns+'}',r'\toprule']


def table_end(adjust=False):
    lines=[r'\bottomrule',r'\end{tabular}']
    if adjust:lines.append(r'\end{adjustbox}')
    return lines+[r'\end{table*}']


def generate(report,out):
    if len(report['folds'])!=10:raise ValueError('Paper tables require all ten folds')
    counts=report['counts']
    if counts['pairs']!=6571 or counts['diseases']!=6059:
        raise ValueError('Unexpected benchmark population')
    aggregation=(r'MR is the equal-fold mean $\pm$ sample standard deviation; Median pools associations; '
                 r'MRR and Hits@k are equal-fold means. Seen and Unseen MR pool associations within each status '
                 r'($n=3{,}085$ and $n=3{,}486$). Primary denotes validation-selected scoring in the matched campaign. ')
    lines=table_start('Historical matched ULTRA and UltraQuery comparison and ULTRA-initialized PhenoQuery component study '
        r'(ten disease folds, 6,571 test associations, 13,625 candidates). '+aggregation+
        r'Panel B initializes every arm from \texttt{ultra\_4g}; it does not compare initialization effects.',
        'tab:ultra_matched','llrrrrrrr')
    lines += [r'Variant & Scoring & MR & Median & MRR & H@1 (\%) & H@10 (\%) & Seen MR & Unseen MR \\',
              r'\midrule',r'\multicolumn{9}{l}{\emph{Panel A: Matched graphs, readouts and adaptation stages}} \\']
    lines += [row(key,mode,report) for key,mode in MAIN_ROWS]
    lines += [r'\midrule',r'\multicolumn{9}{l}{\emph{Panel B: ULTRA-initialized PhenoQuery component campaign}} \\']
    for arm in EQ_MAIN:
        lines.append(row('phenoquery/'+arm,'calibrated',report))
        if arm=='bridge_pool':lines.append(row('phenoquery/'+arm,'raw',report))
    lines += table_end()
    out.mkdir(parents=True,exist_ok=True)
    (out/'revised-ultra-matched.tex').write_text('\n'.join(lines)+'\n')

    lines=[r'\begin{landscape}']
    for cohort in ['general','query','phenoquery']:
        title={'general':'ULTRA matched campaign','query':'UltraQuery matched campaign','phenoquery':'ULTRA-initialized PhenoQuery component campaign'}[cohort]
        lines+=table_start(title+': complete historical results, with both raw and calibrated scores for every arm. '+aggregation+
            'Pooled MR gives equal weight to every association.',
            'tab:ultra_full'+('' if cohort=='general' else '_'+cohort),'llrrrrrrrrr','footnotesize',True)
        lines += [r'Variant & Mode & Fold MR & Pooled MR & Median & MRR & H@1 (\%) & H@10 (\%) & H@100 (\%) & Seen MR & Unseen MR \\',r'\midrule']
        for key in report['results']:
            if key.split('/')[0]!=cohort:continue
            record=report['results'][key];primary=record['primary_mode']
            for mode in [primary,'raw' if primary=='calibrated' else 'calibrated']:
                lines.append(row(key,mode,report,True))
        lines+=table_end(True)+[r'\clearpage']
    lines.append(r'\end{landscape}')
    (out/'revised-ultra-full.tex').write_text('\n'.join(lines)+'\n')

    expected_seen=sum((f['seen_candidates']+1)/2 for f in report['folds'])/10
    expected_unseen=sum((f['unseen_candidates']+1)/2 for f in report['folds'])/10
    caption=(r'Training-status and phenotype-overlap diagnostics. Panel A gives equal-fold within-status mean ranks '
        f'(random expectations {expected_seen:,.1f} for seen and {expected_unseen:,.1f} for unseen genes) '
        'and exact wrong-query controls: each true gene is scored against every other test disease in its fold. '
        r'Panel B pools full-candidate ranks within each stratum; $J_{\mathrm{same}}$ is the maximum phenotype-set '
        r'Jaccard similarity to a training disease of the same gene. Exact profile matches may involve any training gene.')
    lines=[r'\begin{landscape}']+table_start(caption,'tab:ultra_diagnostics','lrrrrrrrr','footnotesize',True)
    lines += [r'\multicolumn{9}{l}{\emph{Panel A: Within-status ranks and wrong-query controls}} \\',r'\midrule',
        r'Variant & \multicolumn{2}{c}{Within-status Raw MR} & \multicolumn{2}{c}{Within-status Cal.\ MR} & \multicolumn{2}{c}{Raw Query Control} & \multicolumn{2}{c}{Cal.\ Query Control} \\',
        r' & Seen & Unseen & Seen & Unseen & Actual & Other & Actual & Other \\',r'\midrule']
    for key in DIAGNOSTIC:
        r=report['results'][key]['modes'];raw=r['raw']['diagnostics_equal_fold'];cal=r['calibrated']['diagnostics_equal_fold']
        values=[raw['within_seen_mr'],raw['within_unseen_mr'],cal['within_seen_mr'],cal['within_unseen_mr'],
                raw['actual_mr'],raw['other_query_mr'],cal['actual_mr'],cal['other_query_mr']]
        lines.append(label(key)+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
    lines += [r'\midrule',r'\multicolumn{9}{l}{\emph{Panel B: Association-pooled performance within phenotype-overlap strata}} \\',r'\midrule',
        r'Stratum & \multicolumn{2}{c}{ULTRA, original Pool} & \multicolumn{2}{c}{UltraQuery, EQ Pool} & \multicolumn{2}{c}{UltraQuery, EQ AND} & \multicolumn{2}{c}{PhenoQuery, typed AND} \\',
        r' & \multicolumn{2}{c}{Calibrated} & \multicolumn{2}{c}{Raw} & \multicolumn{2}{c}{Calibrated} & \multicolumn{2}{c}{Calibrated} \\',
        r' & H@10 (\%) & MR & H@10 (\%) & MR & H@10 (\%) & MR & H@10 (\%) & MR \\',r'\midrule']
    chosen=[('general/original_pool_adapted','calibrated'),('query/augmented_pool_adapted','raw'),
            ('query/augmented_and_adapted','calibrated'),('phenoquery/typed_and','calibrated')]
    for name,text in STRATA:
        values=[]
        for key,mode in chosen:
            m=report['results'][key]['modes'][mode]['strata'][name]
            values += [f"{100*m['hits@10']:.2f}",f"{m['mr']:.2f}"]
        count=f"{counts['strata'][name]:,}".replace(',',r'{,}')
        lines.append(text+f" ($n={count}$) & "+' & '.join(values)+r' \\')
    lines += table_end(True)+[r'\end{landscape}']
    (out/'revised-ultra-diagnostics.tex').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-json',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args()
    generate(json.loads(a.report_json.read_text()),a.out_dir)
    print('Generated historical query comparison, complete results and diagnostics tables')


if __name__=='__main__':main()
