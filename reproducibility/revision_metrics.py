"""Strict prediction-to-metrics audit for the revised LinkGDA evaluation.

Requires explicit candidate identities. Writes per-pair ranks for paired tests;
never treats seed variation as uncertainty over diseases.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/code'))
from link_gda.calibration import standardize


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_scores(path,candidate_file):
    genes=Path(candidate_file).read_text().splitlines()
    if len(genes)<2 or len(genes)!=len(set(genes)):raise ValueError('Empty/duplicate candidate IDs')
    scores=[];keys=[];indices=[]
    with Path(path).open() as f:
        for number,line in enumerate(f,1):
            parts=line.rstrip('\n').split('\t')
            if len(parts)!=len(genes)+3:raise ValueError(f'Row {number}: score width mismatch')
            i=int(parts[2])
            if not 0<=i<len(genes) or genes[i]!=parts[0]:raise ValueError(f'Row {number}: true-gene index mismatch')
            keys.append((parts[0],parts[1]));indices.append(i);scores.append(np.asarray(parts[3:],float))
    if not keys or len(keys)!=len(set(keys)):raise ValueError('Empty/duplicate gene-disease rows')
    scores=np.vstack(scores)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite scores')
    return scores,np.array(indices),keys,genes


def ranks(scores,indices,keys,filtered=False):
    positives=defaultdict(set)
    for i,(_,d) in zip(indices,keys):positives[d].add(i)
    result=[]
    for row,i,(_,d) in zip(scores,indices,keys):
        use=np.ones(len(row),bool)
        if filtered:
            for j in positives[d]-{i}:use[j]=False
        greater=np.count_nonzero(row[use]>row[i]);equal=np.count_nonzero(row[use]==row[i])
        result.append(greater+(equal+1)/2)
    return np.array(result)


def summarize(r,keys,n):
    disease_rows=defaultdict(list)
    for i,(_,d) in enumerate(keys):disease_rows[d].append(i)
    terms={'mr':r,'mrr':1/r,'auc':(n-r)/(n-1)}
    thresholds=sorted(set([1,3,10,100,int(np.ceil(.01*n)),int(np.ceil(.05*n)),n]))
    terms.update({f'hits@{k}':(r<=k).astype(float) for k in thresholds})
    return {'pair_average':{k:float(v.mean()) for k,v in terms.items()},
            'disease_average':{k:float(np.mean([v[ix].mean() for ix in disease_rows.values()])) for k,v in terms.items()},
            'median_pair_rank':float(np.median(r)), 'pairs':len(r),'diseases':len(disease_rows),'candidates':n,
            'percentile_rank_thresholds':{'1_percent':int(np.ceil(.01*n)),'5_percent':int(np.ceil(.05*n))}}


def status_controls(matrix, indices, keys, genes, seen):
    """Rerank against same-status candidates, retaining the full-panel scores."""
    candidate_seen=np.array([g in seen for g in genes])
    result={}
    for label,status in [('seen',True),('unseen',False)]:
        columns=np.flatnonzero(candidate_seen==status)
        rows=np.array([i for i,(g,_) in enumerate(keys) if (g in seen)==status],int)
        if not len(rows):continue
        if len(columns)<2:raise ValueError('Status group has fewer than two candidates')
        lookup={int(old):new for new,old in enumerate(columns)}
        subkeys=[keys[i] for i in rows]
        rr=ranks(matrix[np.ix_(rows,columns)],np.array([lookup[int(indices[i])] for i in rows]),subkeys)
        result[label]={'row_indices':rows.tolist(),'ranks':rr.tolist(),
                       'unfiltered':summarize(rr,subkeys,len(columns)),
                       'tied_control_mean_rank':(len(columns)+1)/2}
    return result


def analyze(path,candidate_file,pool_file=None,reference_stats=None,exclude_diseases=None,training_file=None):
    scores,indices,keys,genes=read_scores(path,candidate_file)
    original_rows=len(keys)
    if exclude_diseases:
        excluded=set(Path(exclude_diseases).read_text().splitlines())
        use=np.array([d not in excluded for _,d in keys])
        scores=scores[use];indices=indices[use];keys=[k for k,keep in zip(keys,use) if keep]
        if not keys:raise ValueError('No queries remain')
    selected=np.arange(len(genes))
    if pool_file:
        pool=Path(pool_file).read_text().splitlines()
        if len(pool)!=len(set(pool)) or len(pool)<2:raise ValueError('Invalid requested pool')
        mapping={g:i for i,g in enumerate(genes)}
        absent=set(pool)-set(genes)
        if absent:raise ValueError(f'{len(absent)} requested candidates were not scored')
        selected=np.array([mapping[g] for g in pool]);lookup={g:i for i,g in enumerate(pool)}
        if any(g not in lookup for g,_ in keys):raise ValueError('Requested pool omits a true gene')
        scores=scores[:,selected];indices=np.array([lookup[g] for g,_ in keys]);genes=pool
    diseases=[d for _,d in keys]
    settings={'raw':scores}
    if len(set(diseases))>=2:settings['disease_loo']=standardize(scores,diseases)
    provenance={'predictions':{'path':str(path),'sha256':sha(path)},'candidates':{'path':str(candidate_file),'sha256':sha(candidate_file)}}
    seen=None
    if training_file:
        with Path(training_file).open() as f:training=list(csv.DictReader(f,delimiter='\t'))
        seen={row['Gene'] for row in training}
        training_diseases={row['Disease'] for row in training}
        if training_diseases&set(diseases):raise ValueError('Test diseases overlap training')
        provenance['training']={'path':str(training_file),'sha256':sha(training_file)}
    if reference_stats:
        ref_candidates=Path(str(reference_stats)+'.candidates.txt')
        # Require identical original order; a same-length wrong pool is not valid.
        if ref_candidates.read_text().splitlines()!=Path(candidate_file).read_text().splitlines():
            raise ValueError('Fixed-reference candidate ordering differs')
        metadata=json.loads(Path(str(reference_stats)+".metadata.json").read_text())
        if metadata.get("calibration_unit")!="disease" or metadata.get("ddof")!=0:
            raise ValueError("Fixed reference must use unique diseases and population standard deviation")
        reference_diseases=metadata.get('reference_diseases',[])
        if not reference_diseases or len(reference_diseases)!=len(set(reference_diseases)):
            raise ValueError('Fixed reference must identify unique reference diseases')
        if set(reference_diseases)&set(diseases):raise ValueError('Test disease in fixed reference')
        if training_file and set(reference_diseases)!=training_diseases:
            raise ValueError('Fixed reference differs from training diseases')
        data=np.loadtxt(reference_stats,skiprows=1,ndmin=2)
        if len(data)!=len(Path(candidate_file).read_text().splitlines()) or not np.array_equal(data[:,0],np.arange(len(data))):
            raise ValueError('Fixed-reference index mismatch')
        if not np.isfinite(data).all() or (data[:,2]<0).any():raise ValueError('Invalid reference statistics')
        settings['fixed_training']=(scores-data[selected,1])/(data[selected,2]+1e-12)
        provenance['reference_stats']={'path':str(reference_stats),'sha256':sha(reference_stats),'candidate_sha256':sha(ref_candidates)}
    for label,p in [('restricted_pool',pool_file),('excluded_diseases',exclude_diseases)]:
        if p:provenance[label]={'path':str(p),'sha256':sha(p)}
    result={'protocol':'unique-disease reference; deterministic average-rank ties; unfiltered primary ranks',
            'provenance':provenance,'source_rows':original_rows,'evaluated_rows':len(keys),'settings':{},
            'keys':[{'gene':g,'disease':d} for g,d in keys]}
    for setting,matrix in settings.items():
        rawr=ranks(matrix,indices,keys);filtered=ranks(matrix,indices,keys,True)
        result['settings'][setting]={'unfiltered':summarize(rawr,keys,len(genes)),
            'ranks':rawr.tolist(),'filtered_sensitivity':{'mr':float(filtered.mean()),'mrr':float((1/filtered).mean()),'ranks':filtered.tolist()}}
        if seen is not None:
            result['settings'][setting]['within_training_status']=status_controls(matrix,indices,keys,genes,seen)
    if seen is not None:
        candidate_seen=np.array([g in seen for g in genes],float)
        result['status_only_controls']={}
        for label,vector in [('seen_first',candidate_seen),('unseen_first',-candidate_seen)]:
            rr=ranks(np.broadcast_to(vector,(len(keys),len(genes))),indices,keys)
            result['status_only_controls'][label]=summarize(rr,keys,len(genes))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--candidate-file',type=Path)
    p.add_argument('--pool-file',type=Path);p.add_argument('--reference-stats',type=Path)
    p.add_argument('--exclude-diseases',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--training-file',type=Path)
    a=p.parse_args();candidate_file=a.candidate_file or Path(str(a.predictions)+'.candidates.txt')
    result=analyze(a.predictions,candidate_file,a.pool_file,a.reference_stats,a.exclude_diseases,a.training_file)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
