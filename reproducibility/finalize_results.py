"""Build manuscript tables and predeclared comparisons from audited metrics."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_folds import aggregate
from paired_comparisons import compare,aligned_ranks
from plot_rank_cdfs import plot
from revision_metrics import sha,summarize

ROOT=None
METRICS=None
PAPER=None
REPLACE=False


def configure(workspace, out_dir, replace=False):
    global ROOT, METRICS, PAPER, REPLACE
    ROOT=Path(workspace).resolve()
    METRICS=ROOT/'revised-metrics'
    if not METRICS.is_dir():
        raise FileNotFoundError(f'Missing per-association metrics: {METRICS}')
    PAPER=Path(out_dir).resolve()
    (PAPER/'fig').mkdir(parents=True,exist_ok=True)
    REPLACE=replace


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def files(group):
    paths=[METRICS/group/f'fold_{i}.json' for i in range(10)]
    for p in paths:
        if not p.exists():raise FileNotFoundError(p)
    return paths


def table(path,caption,label,header,rows):
    n=len(header)
    text=[r'\begin{table*}[tb]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',
          r'\begin{tabular}{l'+'r'*(n-1)+'}',r'\toprule',' & '.join(header)+r' \\',r'\midrule']
    text += [' & '.join(row)+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    content='\n'.join(text)+'\n'
    if path.exists() and path.read_text()!=content and not REPLACE:
        raise FileExistsError(f'{path} differs; pass --replace for deliberate regeneration')
    path.write_text(content)


def main():
    tables=PAPER/'tables';tables.mkdir(exist_ok=True)
    groups=[('SimGIC','resnik_simgic'),('Resnik-BMA','resnik_resnik_bma'),('Lin-BMA','resnik_lin_bma'),
            ('Exomiser-Phive','phive'),('INDIGENA','main-indigena-pfs-calsel'),
            ('ULTRA, zero-shot','main-ultra-pfs'),('ConvKB-D','main-convkb-pfs-calsel'),
            ('LinkGDA','main-linkgda-pfs-calsel')]
    summaries={}
    for _,group in groups:
        summaries[group]=aggregate(files(group))
        dump(METRICS/(group+'-summary.json'),summaries[group])
    # Include all checkpoint/scoring combinations and modality variants.
    for p in sorted(METRICS.glob('main-*')):
        if p.is_dir() and len(list(p.glob('fold_*.json')))==10:
            summaries[p.name]=aggregate(files(p.name))
            dump(METRICS/(p.name+'-summary.json'),summaries[p.name])
    rows=[]
    for name,group in groups:
        for setting in ['raw','disease_loo']:
            m=summaries[group]['settings'][setting]['equal_fold_weight']['pair_average']
            rows.append([name,'Raw' if setting=='raw' else 'Calibrated',
                         f"{m['mr']['mean']:.2f}\\std{{{m['mr']['sd_across_folds']:.2f}}}",
                         f"{m['mrr']['mean']:.3f}",f"{100*m['hits@10']['mean']:.2f}"])
    table(tables/'revised-raw.tex','Raw and calibrated scores on the main pool, using the calibrated-selected checkpoint for trainable models. MR is fold mean $\\pm$ sample standard deviation.','tab:raw_scores',
          ['Method','Scoring','MR','MRR','H@10 (\\%)'],rows)
    primary=summaries['main-linkgda-pfs-calsel']['settings']['disease_loo']['pooled_associations']
    mainrows=[]
    for name,group in groups:
        item=summaries[group]['settings']['disease_loo']
        m=item['equal_fold_weight']['pair_average']
        mainrows.append([name,f"{m['mr']['mean']:.2f}\\std{{{m['mr']['sd_across_folds']:.2f}}}",
                         f"{item['pooled_associations']['median_pair_rank']:.0f}",f"{m['mrr']['mean']:.3f}",
                         f"{100*m['hits@10']['mean']:.2f}",f"{100*m['hits@100']['mean']:.2f}"])
    table(tables/'revised-main.tex',f"Calibrated ranking of {primary['candidates']:,} phenotype-annotated genes across ten disease folds. MR is fold mean $\\pm$ sample standard deviation; MRR and Hits are fold means. Median pools all associations. Trainable models use calibrated-selected checkpoints. Phive uses its native phenotype release and inputs; raw scores are in Suppl. Table~S3.",'tab:rq1',
          ['Method','MR','Median','MRR','H@10 (\\%)','H@100 (\\%)'],mainrows)
    rows=[]
    for modality in ['p','f','s','pf','ps','fs','pfs']:
        group='main-linkgda-'+modality+'-calsel'
        m=summaries[group]['settings']['disease_loo']['equal_fold_weight']['pair_average']
        rows.append(['LinkGDA-'+modality,f"{m['mr']['mean']:.2f}\\std{{{m['mr']['sd_across_folds']:.2f}}}",f"{m['mrr']['mean']:.3f}",f"{100*m['hits@10']['mean']:.2f}"])
    table(tables/'revised-modalities.tex',f"Annotation-source sensitivity on the {primary['candidates']:,}-gene pool. Each variant uses its fixed configuration and calibrated-selected checkpoint; scoring uses disease-level leave-one-out calibration. SD is across ten folds.",'tab:rq1_ablation',['Variant','MR','MRR','H@10 (\\%)'],rows)
    spec={'mode':'main','setting':'disease_loo','comparisons':[]}
    for group in ['main-indigena-pfs-calsel','main-convkb-pfs-calsel','main-ultra-pfs']:
        spec['comparisons'].append({'label':'LinkGDA vs '+group,'paired_files':[[str(a),str(b)] for a,b in zip(files('main-linkgda-pfs-calsel'),files(group))]})
    dump(METRICS/'main-comparisons.json',compare(spec))
    excluded={};rows=[]
    for pool in ['primary','full-go']:
        for modality in ['f','fs']:
            for model in ['linkgda','ultra']:
                group=f'{pool}-{model}-{modality}'+('-calsel' if model=='linkgda' else '')
                paths=files(group) if model=='linkgda' else [METRICS/group/'fold_0.json']
                objects=[json.loads(p.read_text()) for p in paths]
                for obj in objects[1:]:aligned_ranks(objects[0],obj)
                result={}
                for setting in objects[0]['settings']:
                    metrics=[x['settings'][setting]['unfiltered']['pair_average'] for x in objects]
                    result[setting]={metric:{'mean':float(np.mean([m[metric] for m in metrics])),
                                            'sd_across_seeds':float(np.std([m[metric] for m in metrics],ddof=1)) if len(metrics)>1 else 0.}
                                     for metric in metrics[0]}
                excluded[group]={'settings':result,'source_files':[{'path':str(p),'sha256':sha(p)} for p in paths]}
                m=result['disease_loo']
                rows.append(['No-MGI' if pool=='primary' else 'Full GO',('LinkGDA' if model=='linkgda' else 'ULTRA')+'-'+modality,
                             f"{m['mr']['mean']:.2f}\\std{{{m['mr']['sd_across_seeds']:.2f}}}",f"{m['mrr']['mean']:.3f}",f"{100*m['hits@10']['mean']:.2f}"])
        control=json.loads((ROOT/('revised-excluded-'+('primary' if pool=='primary' else 'secondary')+'-controls.json')).read_text())
        m=control['folds'][0]['controls']['function_count_high_first']['pair_average']
        rows.append(['No-MGI' if pool=='primary' else 'Full GO','GO-count control',f"{m['mr']:.2f}",f"{m['mrr']:.3f}",f"{100*m['hits@10']:.2f}"])
    dump(METRICS/'excluded-summary.json',excluded)
    primary_counts=json.loads((METRICS/'primary-ultra-f/fold_0.json').read_text())['settings']['disease_loo']['unfiltered']
    secondary_counts=json.loads((METRICS/'full-go-ultra-f/fold_0.json').read_text())['settings']['disease_loo']['unfiltered']
    table(tables/'revised-excluded.tex',f"Ranking {primary_counts['pairs']:,} associations from {primary_counts['diseases']:,} diseases whose genes lack MGI phenotypes. Primary evaluation ranks {primary_counts['candidates']:,} GO-annotated genes without MGI phenotypes; the secondary full-GO pool has {secondary_counts['candidates']:,} genes. Model scores use disease-level calibration; controls order uncalibrated annotation counts. MR is the mean over ten LinkGDA seeds and SD describes optimization variability; ULTRA is deterministic. Bootstrap intervals use seed-averaged pair ranks.",'tab:excluded',['Pool','Method','MR','MRR','H@10 (\\%)'],rows)
    spec={'mode':'excluded','setting':'disease_loo','comparisons':[]}
    for modality in ['f','fs']:
        baseline=METRICS/f'primary-ultra-{modality}'/'fold_0.json'
        spec['comparisons'].append({'label':'LinkGDA-'+modality+' vs ULTRA-'+modality,'paired_files':[[str(p),str(baseline)] for p in files(f'primary-linkgda-{modality}-calsel')]})
    dump(METRICS/'excluded-comparisons.json',compare(spec))
    strata={};rows=[]
    for name,group in groups:
        objects=[json.loads(p.read_text()) for p in files(group)]
        strata[group]={}
        for status in ['seen','unseen']:
            values=[x['settings']['disease_loo']['within_training_status'][status] for x in objects]
            mrs=[x['unfiltered']['pair_average']['mr'] for x in values]
            strata[group][status]={'fold_mean_rank':mrs,'mean_rank':float(np.mean(mrs)),
                                  'random_mean_rank':float(np.mean([x['tied_control_mean_rank'] for x in values])),
                                  'pairs':sum(x['unfiltered']['pairs'] for x in values)}
        rows.append([name,*[f"{strata[group][s]['mean_rank']:.2f}" for s in ['seen','unseen']]])
    dump(METRICS/'within-status-summary.json',strata)
    table(tables/'revised-status.tex','Mean rank after reranking only among candidates sharing the true gene\'s training-association status. Calibration retains the original reference panel. Fold-averaged random-order ranks are 1,890.1 for seen and 4,923.4 for unseen genes; there are 3,085 and 3,486 test pairs, respectively.','tab:status',['Method','Seen genes','Unseen genes'],rows)
    rows=[]
    for arm in ['rawsel','calsel']:
        for setting in ['raw','disease_loo','fixed_training']:
            m=summaries['main-linkgda-pfs-'+arm]['settings'][setting]['equal_fold_weight']['pair_average']
            rows.append([arm.replace('sel','-selected'),setting.replace('_',' '),f"{m['mr']['mean']:.2f}",f"{100*m['hits@10']['mean']:.2f}"])
    table(tables/'revised-calibration.tex','Checkpoint selection and score calibration for LinkGDA-pfs at fixed hyperparameters. Values are equal-fold means. The fixed training reference contains one vector per distinct training disease.','tab:calibration',['Checkpoint','Scoring','MR','H@10 (\\%)'],rows)
    spec={'title':'Main benchmark: calibrated scores','series':[{'label':name,'setting':'disease_loo','files':[str(p) for p in files(group)]} for name,group in groups if name in ['LinkGDA','INDIGENA','ConvKB-D','ULTRA, zero-shot','Resnik-BMA']]}
    dump(METRICS/'main-cdf-spec.json',spec)
    figure=PAPER/'fig/revised_rank_cdf.pdf'
    if figure.exists() and not REPLACE:raise FileExistsError('Figure exists; pass --replace to regenerate')
    plot(spec,figure)
    print('Final tables, comparisons, status controls and CDF generated.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace',action='store_true',help='Deliberately replace generated tables and figures')
    parser.add_argument('--workspace',type=Path,required=True,help='Data workspace containing revised-metrics/')
    parser.add_argument('--out-dir',type=Path,required=True,help='Output root for tables/ and fig/')
    args=parser.parse_args()
    configure(args.workspace,args.out_dir,args.replace)
    main()
