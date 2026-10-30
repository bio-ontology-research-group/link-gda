"""Descriptive paired rank decomposition and historical-tuning sensitivity."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
sys.path.insert(0,str(Path(__file__).resolve().parent))
from paired_comparisons import aligned_ranks,holm

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace',type=Path,required=True)
parser.add_argument('--validation-file',type=Path,required=True)
args=parser.parse_args()
metrics=args.workspace.resolve()/'revised-metrics'
validation=args.validation_file.resolve()
excluded=set(pd.read_csv(validation,sep='\t').Disease)
out={};sensitivity={}
for group in ['main-ultra-pfs','main-convkb-pfs-calsel','main-indigena-pfs-calsel']:
    left=[];right=[];diffs=[];removed=0
    for fold in range(10):
        x=json.loads((metrics/'main-linkgda-pfs-calsel'/f'fold_{fold}.json').read_text())
        y=json.loads((metrics/group/f'fold_{fold}.json').read_text())
        a,b,keys=aligned_ranks(x,y)
        left.extend(a);right.extend(b)
        keep=np.array([d not in excluded for _,d in keys]);removed+=int((~keep).sum())
        diffs.append(float((a[keep]-b[keep]).mean()))
    a=np.asarray(left);b=np.asarray(right);d=a-b
    out[group]={'pairs':len(a),'linkgda_better_fraction':float((d<0).mean()),
                'linkgda_worse_fraction':float((d>0).mean()),'ties_fraction':float((d==0).mean()),
                'linkgda_median':float(np.median(a)),'comparator_median':float(np.median(b)),
                'pooled_difference':float(d.mean()),
                'comparator_beyond_5000_contribution':float(d[b>5000].sum()/len(d)),
                'remaining_contribution':float(d[b<=5000].sum()/len(d)),
                'cdf':{str(k):{'LinkGDA':float((a<=k).mean()),'comparator':float((b<=k).mean())} for k in [100,500,1000,2000,5000]}}
    sensitivity[group]={'removed_pairs':removed,'removed_diseases':len(excluded),
                        'fold_differences':diffs,'mean_difference':float(np.mean(diffs)),
                        'p_two_sided':float(wilcoxon(diffs,alternative='two-sided').pvalue)}
for record,p in zip(sensitivity.values(),holm([x['p_two_sided'] for x in sensitivity.values()])):record['p_holm']=p
(metrics/'paired-rank-diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
(metrics/'tuning-reuse-sensitivity.json').write_text(json.dumps({'protocol':'Exclude reconstructed fold-0 validation diseases from evaluation; retain original scores/calibration. Historical validation identity is unverified.','comparisons':sensitivity},indent=2)+'\n')
print(json.dumps(sensitivity,indent=2))
