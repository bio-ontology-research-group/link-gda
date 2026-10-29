import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('paired',Path(__file__).resolve().parents[1]/'reproducibility/paired_comparisons.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_cluster_resamples_whole_diseases_and_preserves_pair_weighting():
    # The two-gene disease contributes twice as much to pair MR as the singleton.
    out=m.cluster_interval([2,2,-1],['a','a','b'],repeats=1000)
    assert out['mean_rank_difference']==1
    assert out['interval']==[-1,2]
    assert out['disease_clusters']==2
    assert m.cluster_interval([3]*3,['a','a','b'],repeats=100)['interval']==[3,3]

def test_holm_monotonic_adjustment():
    np.testing.assert_allclose(m.holm([.04,.01,.03]),[.06,.03,.06])

def test_alignment_checks_candidate_identity_and_reorders_pairs():
    def obj(keys,ranks):return {'keys':[{'gene':g,'disease':d} for g,d in keys],
        'provenance':{'candidates':{'sha256':'a'}},'settings':{'disease_loo':{'ranks':ranks}}}
    a=obj([('g1','d1'),('g2','d2')],[1,3]);b=obj([('g2','d2'),('g1','d1')],[5,2])
    x,y,keys=m.aligned_ranks(a,b);np.testing.assert_equal(x-y,[-1,-2])
    b['provenance']['candidates']['sha256']='b'
    with pytest.raises(ValueError,match='pools differ'):m.aligned_ranks(a,b)
