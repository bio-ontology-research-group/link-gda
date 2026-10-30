"""Query-independent pool controls on recovered inputs and reconstructed folds."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'code'))
from link_gda.data import create_train_val_split

def control(pool, train_genes, true_genes):
    high=set(pool)-set(train_genes)
    n=len(pool); k=len(high)
    # Average rank within the two tied tiers, missing training edge ranked first.
    ranks=np.array([(k+1)/2 if g in high else k+(n-k+1)/2 for g in true_genes])
    return {'candidates':n,'top_tier_genes':k,'pairs':len(ranks),
            'true_genes_in_top_tier_fraction':float(np.mean([g in high for g in true_genes])),
            'mr':float(ranks.mean()),'auc':float((n-ranks.mean())/(n-1)),
            'h10_average_rank_ties':float((ranks<=10).mean())}

pool=(ROOT/'main-candidate-order.txt').read_text().splitlines()
folds=[]
for f in range(10):
    d=ROOT/f'rebuilt-main/data/folds/fold_{f}'
    outer=pd.read_csv(d/'train.csv',sep='\t');test=pd.read_csv(d/'test.csv',sep='\t')
    train,val=create_train_val_split(outer,random_seed=0)
    folds.append({'fold':f,'outer_association_absence':control(pool,outer.Gene,test.Gene),
                  'training_graph_association_absence':control(pool,train.Gene,test.Gene)})
d=ROOT/'rebuilt-excluded/data/folds/fold_0'
outer=pd.read_csv(d/'train.csv',sep='\t');test=pd.read_csv(d/'test.csv',sep='\t')
train,val=create_train_val_split(outer,random_seed=0)
expool=(ROOT/'excluded-candidate-order.txt').read_text().splitlines()
result={'scope':'Prepared-input controls on reconstructed folds; no model predictions required. val_seed=0. Average-rank ties.',
        'main_folds':folds,'main_summary':{},
        'excluded':{'outer_association_absence':control(expool,outer.Gene,test.Gene),
                    'training_graph_association_absence':control(expool,train.Gene,test.Gene)}}
for name in ['outer_association_absence','training_graph_association_absence']:
 result['main_summary'][name]={k:{'mean':float(np.mean([r[name][k] for r in folds])),
                               'sd':float(np.std([r[name][k] for r in folds],ddof=1))}
                              for k in ['top_tier_genes','true_genes_in_top_tier_fraction','mr','auc']}
(ROOT/'structural-controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='main_folds'},indent=2))
