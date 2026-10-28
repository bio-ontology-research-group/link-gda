import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]/'code/baselines/native_phrank'
spec=importlib.util.spec_from_file_location('adapter',ROOT/'run.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
from phrank import Phrank


def test_native_parity_full_candidates_and_zero_evidence(tmp_path):
 dag=tmp_path/'dag.tsv';dag.write_text('p1\troot\np2\troot\np3\tp1\n')
 ann=tmp_path/'ann.tsv';ann.write_text('p1\tg1\np3\tg1\np2\tg2\np3\tg3\n')
 engine=Phrank(str(dag),geneannotationsfile=str(ann));genes=['g3','unsupported','g1','g2']
 profiles={'d1':['p3','p3'],'d2':['p2'],'d3':['absent']};diseases=list(profiles)
 actual=a.cached_scores(engine,profiles,diseases,genes)
 for i,d in enumerate(diseases):
  reference=dict((g,s) for s,g in engine.rank_genes_directly(genes,profiles[d]))
  np.testing.assert_array_equal(actual[i],[reference[g] for g in genes])
 assert np.all(actual[:,1]==0) and np.all(actual[2]==0)
 # Unknown profile ties all four candidates, with average rank 2.5.
 assert 1+(actual[2]>actual[2,1]).sum()+((actual[2]==actual[2,1]).sum()-1)/2==2.5


def test_annotations_only_training_disease_sources():
 H=a.HPO;profiles={'train':[H+'1'],'val':[H+'2'],'test':[H+'3']}
 ann,sources=a.training_annotations([('g','train')],[('g','val')],[('g','test')],profiles,{'g','unsupported'})
 assert ann=={'g':{H+'1'}} and sources['g',H+'1']=={'train'}
 assert H+'2' not in ann['g'] and H+'3' not in ann['g']
 with pytest.raises(ValueError,match='overlap'):
  a.training_annotations([('g','train')],[],[('g','train')],profiles,{'g'})


def test_extract_asserted_hpo_only_not_restrictions_or_crossspecies(tmp_path):
 owl=tmp_path/'ont.owl';H=a.HPO
 owl.write_text(f'''<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
 <owl:Class rdf:about="{H}2"><rdfs:subClassOf rdf:resource="{H}1"/><rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/MP_1"/><rdfs:subClassOf><owl:Restriction><owl:someValuesFrom rdf:resource="{H}3"/></owl:Restriction></rdfs:subClassOf></owl:Class>
 <owl:Class rdf:about="{H}1"/></rdf:RDF>''')
 info=a.extract_hpo(owl,tmp_path/'extract')
 assert info['edge_count']==1 and info['node_count']==2
 assert (tmp_path/'extract/hpo-dag.tsv').read_text()==f'{H}2\t{H}1\n'


def test_equal_overlap_stable_sum_despite_different_annotation_construction():
 from types import SimpleNamespace
 from collections import defaultdict
 engine=SimpleNamespace(_gene_pheno_map={'g1':['a','b','c','outside1'],'g2':['outside2','c','b','a']},
                        _child_to_parent=defaultdict(list),_marginal_IC={'a':1e16,'b':1.,'c':1.,'outside1':8.,'outside2':9.})
 result=a.cached_scores(engine,{'d':['c','a','b']},['d'],['g1','g2'])
 assert result[0,0]==result[0,1]==1e16+2
