"""Exact expectation over all incorrect disease queries, with fixed calibration.

Each true association is evaluated against every other unique disease vector
in its fold/cohort. This is a descriptive query-permutation control, not a
significance test. Calibration panels remain those of the original evaluation.
"""
import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_metrics import read_scores,standardize,sha

ROOT=Path(__file__).resolve().parent


def expected_wrong_query_ranks(ranked,inverse,indices):
    """Exact uniform expectation, excluding each association's own disease."""
    if len(ranked)<2:raise ValueError('At least two diseases required')
    return (ranked.sum(axis=0)[indices]-ranked[inverse,indices])/(len(ranked)-1)


def evaluate(path):
    obj=json.loads(Path(path).read_text())
    prov=obj['provenance']
    scores,indices,keys,genes=read_scores(prov['predictions']['path'],prov['candidates']['path'])
    if 'restricted_pool' in prov:
        pool=Path(prov['restricted_pool']['path']).read_text().splitlines()
        index={g:i for i,g in enumerate(genes)}
        columns=[index[g] for g in pool]
        scores=scores[:,columns];genes=pool;index={g:i for i,g in enumerate(genes)}
        indices=np.array([index[g] for g,_ in keys])
    else:columns=np.arange(len(genes))
    diseases=np.array([d for _,d in keys]);unique,first,inverse=np.unique(diseases,return_index=True,return_inverse=True)
    assert len(unique)>1
    matrices={'raw':scores,'disease_loo':standardize(scores,diseases)}
    if 'reference_stats' in prov:
        ref=np.loadtxt(prov['reference_stats']['path'],skiprows=1,ndmin=2)
        matrices['fixed_training']=(scores-ref[columns,1])/(ref[columns,2]+1e-12)
    out={}
    for setting,matrix in matrices.items():
        ranked=rankdata(-matrix[first],axis=1,method='average')
        actual=ranked[inverse,indices]
        np.testing.assert_array_equal(actual,obj['settings'][setting]['ranks'])
        wrong=expected_wrong_query_ranks(ranked,inverse,indices)
        # Uniformly choose another disease for each unique query; all its true
        # genes share that substituted query. Linearity gives this exact MR.
        out[setting]={'actual_pair_mean_rank':float(actual.mean()),
                      'wrong_query_pair_mean_rank':float(wrong.mean()),
                      'actual_minus_wrong_query':float((actual-wrong).mean()),
                      'wrong_query_expected_rank_per_pair':wrong.tolist()}
    return {'metrics':str(path),'metrics_sha256':sha(path),'pairs':len(keys),'diseases':len(unique),
            'candidates':len(genes),'random_mean_rank':(len(genes)+1)/2,'settings':out}


def main(workers, workspace):
    ROOT=Path(workspace).resolve()
    groups=['main-linkgda-pfs-calsel','main-indigena-pfs-calsel','main-convkb-pfs-calsel','main-ultra-pfs',
            'primary-linkgda-f-calsel','primary-linkgda-fs-calsel','primary-ultra-f','primary-ultra-fs']
    output={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for group in groups:
            paths=sorted((ROOT/'revised-metrics'/group).glob('fold_*.json'))
            expected=1 if group.startswith('primary-ultra') else 10
            assert len(paths)==expected,(group,len(paths))
            runs=list(pool.map(evaluate,paths))
            summary={setting:{key:float(np.mean([run['settings'][setting][key] for run in runs]))
                              for key in ['actual_pair_mean_rank','wrong_query_pair_mean_rank','actual_minus_wrong_query']}
                     for setting in runs[0]['settings']}
            output[group]={'runs':runs,'summary':summary}
            print(group,summary,flush=True)
    result={'protocol':__doc__,'groups':output}
    (ROOT/'revised-metrics/query-permutation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--workspace',type=Path,required=True)
    args=parser.parse_args()
    main(args.workers,args.workspace)
