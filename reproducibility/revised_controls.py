"""Disease-independent annotation/status controls in frozen revised pools."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from link_gda.data import create_train_val_split
sys.path.insert(0,str(Path(__file__).resolve().parent))
from revision_metrics import summarize,sha


def controls(data,folds,pool):
    data=Path(data);folds=Path(folds);genes=Path(pool).read_text().splitlines();ix={g:i for i,g in enumerate(genes)}
    if len(ix)!=len(genes):raise ValueError('Duplicate candidates')
    counts={}
    sources={'phenotype':'gene_phenotypes.csv','function':'gene_functions.csv','expression':'gene_site.csv'}
    for label,file in sources.items():
        frame=pd.read_csv(data/file,dtype=str).drop_duplicates();n=frame.groupby('Gene').size()
        counts[label]=np.array([n.get(g,0) for g in genes],float)
    results=[]
    for fold in sorted(folds.glob('fold_*')):
        train,val=create_train_val_split(pd.read_csv(fold/'train.csv',sep='\t'),random_seed=0)
        test=pd.read_csv(fold/'test.csv',sep='\t');keys=list(zip(test.Gene,test.Disease))
        if any(g not in ix for g,_ in keys):raise ValueError('True gene missing from candidate pool')
        true=np.array([ix[g] for g,_ in keys]);associated=set(train.Gene)
        predictors={'tied':np.zeros(len(genes)), 'training_association_present':np.array([g in associated for g in genes],float),
                    'MGI_annotation_present':(counts['phenotype']>0).astype(float),**{label+'_count':n for label,n in counts.items()}}
        summaries={}
        for label,scores in predictors.items():
            directions=[('high_first',1)] if label=='tied' else [('high_first',1),('low_first',-1)]
            for direction,sign in directions:
                ranks=rankdata(-sign*scores,method='average')[true]
                summaries[label+'_'+direction]=summarize(ranks,keys,len(genes))
        results.append({'fold':fold.name,'train_pairs':len(train),'validation_pairs':len(val),
                        'test_pairs':len(test),'test_pairs_with_gene_seen_training':int(test.Gene.isin(associated).sum()),'controls':summaries})
    return {'pool':str(pool),'pool_sha256':sha(pool),'candidates':len(genes),
            'annotation_counts':'unique prepared annotation pairs before ontology coverage filtering',
            'input_sha256':{name:sha(data/file) for name,file in sources.items()},'folds':results}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--folds',type=Path,required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.write_text(json.dumps(controls(a.data,a.folds,a.pool),indent=2)+'\n')
