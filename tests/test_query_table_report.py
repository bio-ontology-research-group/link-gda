import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from analysis.summarize_query_tables import diagnostics, metadata, metrics


def test_wrong_query_and_status_diagnostics_with_ties():
    scores = np.array([[4,4,1,0],[0,3,3,2],[1,0,1,1]],dtype=float)
    di,gi = np.array([0,0,1,2]),np.array([0,2,3,1])
    seen = np.array([True,True,False,False])
    within,other = diagnostics(scores,di,gi,seen)
    expected_within,expected_other = [],[]
    for row,column in zip(di,gi):
        eligible = [j for j in range(4) if seen[j]==seen[column]]
        ranked = rankdata(-scores[row,eligible],method='average')
        expected_within.append(ranked[eligible.index(column)])
        expected_other.append(np.mean([rankdata(-scores[q],method='average')[column]
                                       for q in range(3) if q!=row]))
    np.testing.assert_array_equal(within,expected_within)
    np.testing.assert_array_equal(other,expected_other)


def test_metadata_uses_training_gene_set_without_lookup_mutation(tmp_path):
    (tmp_path/'candidates.txt').write_text('g0\ng1\ng2\ng3\n')
    for name,rows in [('training',[('g0','d0'),('g1','d1')]),
                       ('validation',[('g2','d2')]),
                       ('test',[('g3','d3'),('g0','d4'),('g3','d4')])]:
        pd.DataFrame(rows,columns=['Gene','Disease']).to_csv(tmp_path/(name+'.tsv'),sep='\t',index=False)
    (tmp_path/'profiles.json').write_text(json.dumps({
        'd0':['p0','p1'],'d1':['p2'],'d2':['p3'],'d3':['p2'],'d4':['p0']}))
    genes,rows,seen,candidate_seen,strata = metadata(tmp_path)
    np.testing.assert_array_equal(seen,[False,True,False])
    np.testing.assert_array_equal(candidate_seen,[True,True,False,False])
    np.testing.assert_array_equal(strata['seen_high'],[False,True,False])
    np.testing.assert_array_equal(strata['training_profile_match'],[True,False,False])
    np.testing.assert_array_equal(strata['multi_gene'],[False,True,True])
    assert metrics([1,10,20])['hits@10']==2/3
