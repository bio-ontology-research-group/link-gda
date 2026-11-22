"""Deterministic clinical ranks, historical LOO calibration and fixed-grid selection."""
import numpy as np


def calibrate(values):
    values=np.asarray(values,dtype=np.float64);values=values-values[:1];n=len(values)-1
    assert n>=1
    mean=(values.sum(0)-values)/n;variance=(values*values).sum(0)/n-values*values/n-mean*mean
    return (values-mean)/(np.sqrt(np.maximum(variance,0))+1e-12)


def aggregate(ranks):
    ranks=np.asarray(ranks,dtype=np.float64);assert len(ranks) and np.isfinite(ranks).all() and np.all(ranks>=1)
    return {'hits@1':float(np.mean(ranks<=1)),'hits@10':float(np.mean(ranks<=10)),'mrr':float(np.mean(1/ranks)),'mr':float(np.mean(ranks))}


def measure(scores,answers,candidate_ids,filtered=False,seen_genes=None):
    assert np.isfinite(scores).all() and len(scores)==len(answers)
    columns={int(gene):i for i,gene in enumerate(candidate_ids)};all_ranks=[];per_query=[];seen=[]
    for row,genes in zip(scores,answers):
        ranks=[];known=[columns[gene] for gene in genes]
        for gene in genes:
            index=columns[gene];target=row[index];mask=np.ones(len(row),dtype=bool);mask[known if filtered else [index]]=False
            rank=float(1+(row[mask]>target).sum()+.5*(row[mask]==target).sum());ranks.append(rank)
            if seen_genes is not None:seen.append(gene in seen_genes)
        per_query.append(aggregate(ranks));all_ranks.extend(ranks)
    result={'queries':len(answers),'associations':len(all_ranks),'answer_micro':aggregate(all_ranks),'query_macro':{key:float(np.mean([row[key] for row in per_query])) for key in ('hits@1','hits@10','mrr','mr')},'ranks':all_ranks}
    if seen_genes is not None:
        seen=np.asarray(seen,dtype=bool);ranks=np.asarray(all_ranks)
        for label,mask in [('TRAIN_seen_target_gene',seen),('TRAIN_unseen_target_gene',~seen)]:
            result[label]={'associations':int(mask.sum()),**aggregate(ranks[mask])} if mask.any() else {'associations':0}
    return result



def selector(curve, score_type):
    assert score_type in ('raw', 'calibrated')
    assert curve and curve[0]['step'] == 0
    ordered = sorted(curve, key=lambda row: row['step'])
    assert len({row['step'] for row in ordered}) == len(ordered)
    initial = ordered[0]['validation']['real'][score_type]['answer_micro']
    eligible = []
    for row in ordered:
        value = row['validation']['real'][score_type]['answer_micro']
        if value['hits@1'] >= initial['hits@1'] and value['hits@10'] >= initial['hits@10']:
            eligible.append((value['hits@10'], value['hits@1'], value['mrr'], -value['mr'], -row['step'], row['step']))
    assert eligible
    return {'step': max(eligible)[-1], 'score_type': score_type,
            'criterion': 'Own step0 H1/H10 floor; H10,H1,MRR,-MR; earliest exact ties; step0 fallback',
            'eligible_steps': [row[-1] for row in eligible]}
