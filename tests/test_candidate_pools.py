import pytest
from link_gda.candidates import select_candidates,write_candidates

def test_annotation_pool_does_not_depend_on_association_pool(tmp_path):
    (tmp_path/'upheno_edges.tsv').write_text('p\tr\tq\n')
    (tmp_path/'gene_phenotypes.csv').write_text('Gene,Phenotype\ng2,p\ng1,p\ng2,q\ng3,p\n')
    (tmp_path/'gene_diseases.csv').write_text('Gene,Disease\ng2,d\n')
    assert select_candidates(tmp_path,'phenotype',required_genes=['g2'])==['g1','g2','g3']
    (tmp_path/'gene_diseases.csv').write_text('Gene,Disease\ng1,d\ng2,d\n')
    assert select_candidates(tmp_path,'phenotype')==['g1','g2','g3']

def test_unrepresented_candidate_or_missing_answer_is_error(tmp_path):
    (tmp_path/'gene_functions.csv').write_text('Gene,Function\ng1,f\ng2,f\n')
    with pytest.raises(ValueError,match='absent from training graph'):
        select_candidates(tmp_path,'function',available_entities=['g1'])
    with pytest.raises(ValueError,match='evaluated genes absent'):
        select_candidates(tmp_path,'function',required_genes=['g3'])

def test_candidate_order_is_preserved_in_sidecar(tmp_path):
    p=tmp_path/'candidates.txt';write_candidates(p,['g2','g1'])
    assert p.read_text()=='g2\ng1\n'
    with pytest.raises(ValueError):write_candidates(p,['g1','g1'])


def test_phenotype_pool_requires_a_represented_profile(tmp_path):
    (tmp_path/'upheno_edges.tsv').write_text('p\tr\tq\n')
    (tmp_path/'gene_phenotypes.csv').write_text('Gene,Phenotype\ng1,p\ng2,obsolete\n')
    assert select_candidates(tmp_path,'phenotype')==['g1']
