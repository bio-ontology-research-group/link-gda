"""Exercise real prediction writing and calibration with a tiny CPU scorer."""
import numpy as np
import pandas as pd
import pytest
th=pytest.importorskip('torch')
from link_gda.evaluation import evaluate_by_graph

class Scorer:
    device=th.device('cpu')
    def __init__(self):self.calls=0
    def eval(self):return self
    def predict_hrt(self, triples):
        self.calls+=1
        forward=(triples[:,2]-2)*2+1
        return th.where(triples[:,0]==0,forward,6-forward).float().reshape(-1,1)

def test_prediction_sidecars_and_unique_reference_statistics(tmp_path):
    rows=pd.DataFrame({'Disease':['d1','d1','d2','d3'],'Gene':['g1','g2','g1','g2']})
    baseline=tmp_path/'baseline.tsv';prefix=str(tmp_path/'scores')
    scorer=Scorer()
    result=evaluate_by_graph(scorer,rows,{'d1':['p1'],'d2':['p2'],'d3':['p3']},['g1','g2'],
        entity_to_id={'g1':0,'g2':1,'p1':2,'p2':3,'p3':4},relation_to_id={'causes_phenotype':0},
        output_file_prefix=prefix,baseline_out=str(baseline),calibrate=True,calibration_unit='disease')
    assert scorer.calls==3
    assert result[0]['mr']==pytest.approx(1.625)
    stored=np.loadtxt(baseline,skiprows=1)
    np.testing.assert_allclose(stored[:,1],[3,3])
    np.testing.assert_allclose(stored[:,2],[np.sqrt(8/3)]*2)
    for f in ['scores_by_graph_bma.tsv.candidates.txt','scores_by_graph_bmm.tsv.candidates.txt','baseline.tsv.candidates.txt']:
        assert (tmp_path/f).read_text()=='g1\ng2\n'
    first=(tmp_path/'scores_by_graph_bma.tsv').read_text().splitlines()[0].split('\t')
    assert first[:3]==['g1','d1','0']
    np.testing.assert_allclose(np.array(first[3:],float),[1,5])


def test_query_without_represented_phenotype_fails():
    rows=pd.DataFrame({'Disease':['d1'],'Gene':['g1']})
    with pytest.raises(ValueError,match='no represented phenotypes'):
        evaluate_by_graph(Scorer(),rows,{'d1':['missing']},['g1','g2'],
            entity_to_id={'g1':0,'g2':1},relation_to_id={'causes_phenotype':0},calibration_unit='disease')


def test_similarity_zero_dot_is_real_similarity_and_padding_is_ignored():
    from link_gda.evaluation import compare_vectorized
    genes=th.tensor([[[1.,0.],[0.,0.]],[[0.,1.],[1.,1.]]])
    disease=th.tensor([[0.,1.]])
    counts=th.tensor([1.,2.])
    result=compare_vectorized(genes,disease,counts)
    th.testing.assert_close(result,th.tensor([.5,th.sigmoid(th.tensor(1.)).item()]))
    # Ranking a subset must preserve each gene's score exactly.
    th.testing.assert_close(compare_vectorized(genes[:1],disease,counts[:1]),result[:1])
