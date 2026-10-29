"""Descriptive rank strata, using training rows and label-free annotation counts."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from revision_metrics import sha,summarize


def annotation_bins(counts):
    """Zero plus positive-count quartiles; cutpoints use every candidate gene."""
    counts=np.asarray(counts,int)
    positive=counts[counts>0]
    cuts=np.unique(np.quantile(positive,[.25,.5,.75])).tolist() if len(positive) else []
    labels=['zero' if n==0 else f'positive_bin_{int(np.searchsorted(cuts,n,side="left"))+1}' for n in counts]
    return labels,cuts


def analyze(metric_file,training_file,data_dir):
    metrics=json.loads(Path(metric_file).read_text())
    keys=[(x['gene'],x['disease']) for x in metrics['keys']]
    pool=metrics['provenance'].get('restricted_pool',metrics['provenance']['candidates'])
    pool_file=Path(pool['path'])
    if sha(pool_file)!=pool['sha256']:raise ValueError('Candidate pool changed')
    genes=pool_file.read_text().splitlines();ix={g:i for i,g in enumerate(genes)}
    if len(ix)!=len(genes) or any(g not in ix for g,_ in keys):raise ValueError('Invalid candidate identities')
    training=pd.read_csv(training_file,sep='\t',dtype=str)
    seen=set(training.Gene)
    if set(training.Disease)&{d for _,d in keys}:raise ValueError('Test diseases present in training')
    groups={'training_association':['seen' if g in seen else 'unseen' for g,_ in keys]}
    sources={};cutpoints={}
    for name,file in [('phenotype','gene_phenotypes.csv'),('function','gene_functions.csv'),('expression','gene_site.csv')]:
        p=Path(data_dir)/file
        frame=pd.read_csv(p,dtype=str).drop_duplicates()
        n=frame.groupby('Gene').size();counts=[int(n.get(g,0)) for g in genes]
        labels,cuts=annotation_bins(counts)
        groups[name+'_count']=[labels[ix[g]] for g,_ in keys]
        cutpoints[name]={'positive_count_quantile_cuts':cuts,'counts':'unique prepared annotation pairs before ontology coverage filtering'}
        sources[name]={'path':str(p),'sha256':sha(p)}
    out={}
    for setting,values in metrics['settings'].items():
        r=np.asarray(values['ranks'],float)
        if len(r)!=len(keys) or not np.isfinite(r).all():raise ValueError('Invalid ranks')
        out[setting]={}
        for dimension,labels in groups.items():
            out[setting][dimension]={}
            for label in sorted(set(labels)):
                indices=[i for i,x in enumerate(labels) if x==label]
                out[setting][dimension][label]=summarize(r[indices],[keys[i] for i in indices],len(genes))
    return {'metrics':{'path':str(metric_file),'sha256':sha(metric_file)},'training':{'path':str(training_file),'sha256':sha(training_file)},'annotation_sources':sources,'cutpoints':cutpoints,'settings':out,'interpretation':'Descriptive within-run strata; candidate annotation quantiles use no disease-association labels. Not a causal decomposition.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',type=Path,required=True);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.write_text(json.dumps(analyze(a.metrics,a.training,a.data),indent=2,allow_nan=False)+'\n')
