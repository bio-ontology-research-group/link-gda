"""Predeclared paired fold tests and excluded disease-cluster uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon


def holm(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p);out=np.empty_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out.tolist()


def aligned_ranks(left,right,setting='disease_loo'):
    lk=[(x['gene'],x['disease']) for x in left['keys']]
    rk=[(x['gene'],x['disease']) for x in right['keys']]
    if len(lk)!=len(set(lk)) or len(rk)!=len(set(rk)) or set(lk)!=set(rk):
        raise ValueError('Paired comparisons require identical unique gene-disease rows')
    for obj in (left,right):
        if setting not in obj['settings']:raise ValueError(f'Missing setting {setting}')
    def candidate_hash(obj):
        return obj['provenance'].get('restricted_pool',obj['provenance']['candidates'])['sha256']
    if candidate_hash(left)!=candidate_hash(right):raise ValueError('Candidate pools differ')
    lookup={k:i for i,k in enumerate(rk)}
    a=np.asarray(left['settings'][setting]['ranks'],float)
    b=np.asarray(right['settings'][setting]['ranks'],float)[[lookup[k] for k in lk]]
    if len(a)!=len(lk) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid ranks')
    return a,b,lk


def cluster_interval(differences,diseases,repeats=10000,seed=20261013,alpha=.025):
    differences=np.asarray(differences,float)
    unique,inverse=np.unique(diseases,return_inverse=True)
    sums=np.bincount(inverse,weights=differences);counts=np.bincount(inverse)
    rng=np.random.default_rng(seed);draws=[]
    for start in range(0,repeats,100):
        indices=rng.integers(0,len(unique),size=(min(100,repeats-start),len(unique)))
        draws.extend((sums[indices].sum(axis=1)/counts[indices].sum(axis=1)).tolist())
    return {'mean_rank_difference':float(differences.mean()),
            'confidence_level':1-alpha,'interval':np.quantile(draws,[alpha/2,1-alpha/2]).tolist(),
            'bootstrap_replicates':repeats,'bootstrap_seed':seed,'disease_clusters':len(unique)}


def compare(spec):
    mode=spec['mode'];setting=spec.get('setting','disease_loo');records=[]
    for comparison in spec['comparisons']:
        differences=[];keys0=None
        for files in comparison['paired_files']:
            a,b,keys=aligned_ranks(json.loads(Path(files[0]).read_text()),json.loads(Path(files[1]).read_text()),setting)
            if mode=='excluded':
                if keys0 is None:keys0=keys
                elif set(keys)!=set(keys0):raise ValueError('Seed cohorts differ')
                lookup={k:i for i,k in enumerate(keys)}
                differences.append((a-b)[[lookup[k] for k in keys0]])
            else:differences.append(float((a-b).mean()))
        if len(differences)!=10:raise ValueError('Protocol requires ten matched folds or seeds')
        if mode=='main':
            d=np.asarray(differences)
            p=1. if np.all(d==0) else float(wilcoxon(d,alternative='two-sided',method='auto').pvalue)
            result={'mean_rank_difference':float(d.mean()),'fold_differences':differences,'p_two_sided':p}
        elif mode=='excluded':
            d=np.stack(differences);result=cluster_interval(d.mean(axis=0),[disease for _,disease in keys0])
            result['seed_mean_differences']=d.mean(axis=1).tolist()
            result['seed_sd_difference']=float(d.mean(axis=1).std(ddof=1))
        else:raise ValueError('Unknown comparison mode')
        records.append({'comparison':comparison['label'],**result})
    required=3 if mode=='main' else 2
    if len(records)!=required:raise ValueError(f'Protocol requires {required} comparisons in this family')
    if mode=='main':
        for record,p in zip(records,holm([r['p_two_sided'] for r in records])):record['p_holm']=p
    return {'mode':mode,'setting':setting,'difference_direction':'left minus right; negative favors left',
            'comparisons':records,'specification':spec}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.write_text(json.dumps(compare(json.loads(a.spec.read_text())),indent=2,allow_nan=False)+'\n')
