"""Aggregate ten disease-disjoint main folds without treating pairs as replicates."""
import argparse
import json
from pathlib import Path
import numpy as np
from revision_metrics import summarize,sha


def aggregate(files):
    if len(files)!=10 or len(set(files))!=10:raise ValueError('Ten distinct fold files required')
    objects=[json.loads(Path(f).read_text()) for f in files]
    hashes={o['provenance'].get('restricted_pool',o['provenance']['candidates'])['sha256'] for o in objects}
    if len(hashes)!=1:raise ValueError('Candidate pools differ')
    keys=[];seen=set()
    for o in objects:
        k=[(x['gene'],x['disease']) for x in o['keys']];d={x[1] for x in k}
        if seen&d:raise ValueError('Test diseases overlap across folds')
        if len(k)!=len(set(k)):raise ValueError('Duplicate association rows')
        seen.update(d);keys.extend(k)
    settings=set(objects[0]['settings'])
    if any(set(o['settings'])!=settings for o in objects):raise ValueError('Scoring settings differ')
    result={}
    for setting in sorted(settings):
        fold=[o['settings'][setting]['unfiltered'] for o in objects]
        n={s['candidates'] for s in fold}
        if len(n)!=1:raise ValueError('Candidate counts differ')
        ranks=[np.asarray(o['settings'][setting]['ranks'],float) for o in objects]
        for o,r in zip(objects,ranks):
            if len(r)!=len(o['keys']) or not np.isfinite(r).all():raise ValueError('Invalid ranks')
        means={}
        for unit in ['pair_average','disease_average']:
            means[unit]={}
            for metric in fold[0][unit]:
                values=np.asarray([s[unit][metric] for s in fold],float)
                means[unit][metric]={'mean':float(values.mean()),'sd_across_folds':float(values.std(ddof=1)),'fold_values':values.tolist()}
        result[setting]={'equal_fold_weight':means,'pooled_associations':summarize(np.concatenate(ranks),keys,n.pop())}
    return {'fold_files':[{'path':str(f),'sha256':sha(f)} for f in files],'candidate_sha256':hashes.pop(),'pairs':len(keys),'diseases':len(seen),'settings':result}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folds',nargs=10,type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.write_text(json.dumps(aggregate(a.folds),indent=2,allow_nan=False)+'\n')
