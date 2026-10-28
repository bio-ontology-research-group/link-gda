"""Generate reviewer-requested controls and secondary endpoints from ranks."""
import argparse,json
from pathlib import Path
import numpy as np
import finalize_results as final

root=Path(__file__).resolve().parent;metrics=root/'revised-metrics';tables=root.parents[1]/'tables'
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replace',action='store_true')
final.REPLACE=parser.parse_args().replace
groups=[('SimGIC','resnik_simgic'),('Resnik-BMA','resnik_resnik_bma'),('Lin-BMA','resnik_lin_bma'),('Phive','phive'),
        ('INDIGENA','main-indigena-pfs-calsel'),('ULTRA','main-ultra-pfs'),('ConvKB-D','main-convkb-pfs-calsel'),('LinkGDA','main-linkgda-pfs-calsel')]
rows=[]
for pool in ['main','excluded-primary','excluded-secondary']:
    data=json.loads((root/f'revised-{pool}-controls.json').read_text())
    for name in ['phenotype','function','expression']:
        if pool!='main' and name=='phenotype':continue
        for direction in ['high_first','low_first']:
            values=[x['controls'][f'{name}_count_{direction}']['pair_average'] for x in data['folds']]
            rows.append([pool.replace('excluded-',''),name,'More first' if direction=='high_first' else 'Fewer first',
                         f"{np.mean([v['mr'] for v in values]):.2f}",f"{100*np.mean([v['hits@10'] for v in values]):.2f}",
                         f"{100*np.mean([v['hits@100'] for v in values]):.2f}"])
final.table(tables/'revised-controls.tex','Query-independent prepared-annotation count controls with average-rank ties. Primary and secondary denote the no-MGI and full-GO excluded pools.','tab:count_controls',['Pool','Count','Ordering','MR','H@10 (\\%)','H@100 (\\%)'],rows)
for scope in ['main','primary']:
    if scope=='primary':groups=[('LinkGDA-f','primary-linkgda-f-calsel'),('ULTRA-f','primary-ultra-f'),('LinkGDA-fs','primary-linkgda-fs-calsel'),('ULTRA-fs','primary-ultra-fs')]
    rows=[]
    for name,group in groups:
        objects=[json.loads(p.read_text()) for p in sorted((metrics/group).glob('fold_*.json'))]
        v=[o['settings']['disease_loo'] for o in objects];n=v[0]['unfiltered']['candidates']
        cols=[]
        for k in [1,3,100,int(np.ceil(.01*n)),int(np.ceil(.05*n))]:
            cols.append(f"{100*np.mean([x['unfiltered']['pair_average'][f'hits@{k}'] for x in v]):.2f}")
        cols += [f"{np.mean([x['unfiltered']['disease_average']['mr'] for x in v]):.2f}",
                 f"{np.mean([x['filtered_sensitivity']['mr'] for x in v]):.2f}",
                 f"{np.mean([x['unfiltered']['median_pair_rank'] for x in v]):.1f}"]
        rows.append([name,*cols])
    caption=('Main-benchmark secondary metrics, averaged over folds. ' if scope=='main' else 'Primary excluded-cohort secondary metrics, averaged over seeds; ULTRA is deterministic. ')
    caption+='All scores use disease-level calibration. Hits are percentages; the final column averages within-fold or within-seed medians. The 1\\% and 5\\% thresholds are '+str(int(np.ceil(.01*n)))+' and '+str(int(np.ceil(.05*n)))+' candidates.'
    path=tables/f'revised-secondary-{scope}.tex'
    final.table(path,caption,'tab:secondary_'+scope,['Method','H@1','H@3','H@100','H@1\\%','H@5\\%','Disease MR','Filtered MR','Median'],rows)
    text=path.read_text().replace('\\begin{tabular}',r'\resizebox{\textwidth}{!}{\begin{tabular}').replace('\\end{tabular}',r'\end{tabular}}')
    path.write_text(text)
permuted=json.loads((metrics/'query-permutation.json').read_text());rows=[]
for group,obj in permuted['groups'].items():
    s=obj['summary'];name=group.replace('main-','Main: ').replace('primary-','No-MGI: ').replace('-calsel','').replace('-pfs','')
    rows.append([name,*[f"{s[setting][key]:.2f}" for setting in ['raw','disease_loo'] for key in ['actual_pair_mean_rank','wrong_query_pair_mean_rank']]])
final.table(tables/'revised-query-control.tex','Exact mean-rank expectation when each association is scored against every other distinct disease in its fold or cohort. Calibration panels remain fixed. Averages give equal weight to folds or seeds. Random-order MR is 6,813 for main and 2,848.5 for no-MGI. This is a descriptive control.','tab:query_control',['Model / pool','Raw: actual','Raw: other','Cal.: actual','Cal.: other'],rows)
print('Control, secondary-endpoint and exact query-permutation tables generated.')
