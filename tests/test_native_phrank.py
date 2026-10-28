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


def test_full_fold_disease_max_matches_author_and_differs_from_union(tmp_path):
 import json,phrank
 H=a.HPO;data=tmp_path/'data/fold0';data.mkdir(parents=True)
 ontology=tmp_path/'ontology';ontology.mkdir()
 (ontology/'hpo-dag.tsv').write_text(''.join(H+p+'\t'+H+'root\n' for p in ['a','b','c']))
 (ontology/'hpo-nodes.txt').write_text('\n'.join(H+p for p in ['root','a','b','c'])+'\n')
 a.write(ontology/'ontology.json',{'dag_sha256':a.sha(ontology/'hpo-dag.tsv'),
                                 'nodes_sha256':a.sha(ontology/'hpo-nodes.txt')})
 profiles={'one':[H+'a'],'two':[H+'b'],'both':[H+'a',H+'b'],'three':[H+'c'],
           'validation':[H+'c'],'testAB':[H+'a',H+'b'],'testC':[H+'c']}
 a.write(data/'profiles.json',profiles)
 genes=['g1','g2','g3','unsupported'];(data/'candidates.txt').write_text('\n'.join(genes)+'\n')
 splits={'training':[('g1','one'),('g1','two'),('g2','both'),('g3','three')],
         'validation':[('g3','validation')],'test':[('g1','testAB'),('g3','testC')]}
 for name,pairs in splits.items():
  (data/(name+'.tsv')).write_text('Gene\tDisease\n'+''.join(g+'\t'+d+'\n' for g,d in pairs))
 a.write(data/'preparation.json',{label+'_sha256':a.sha(data/name) for name,label in
         [('training.tsv','training'),('validation.tsv','validation'),('test.tsv','test'),
          ('profiles.json','profiles'),('candidates.txt','candidates')]})
 summary=ROOT.parents[1]/'analysis/summarize_ultra_campaign.py'
 author_root=Path(phrank.__file__).resolve().parents[1]
 for mode in ['direct','disease-max']:
  a.run_fold(data.parent,ontology,author_root,tmp_path/mode,0,summary,mode)
 direct=tmp_path/'direct/fold0';dmax=tmp_path/'disease-max/fold0'
 with np.load(direct/'scores.npz') as z:union=z['scores'];ds=z['diseases'].tolist()
 with np.load(dmax/'scores.npz') as z:maximum=z['scores'];assert z['genes'].tolist()==genes
 i=ds.index('testAB')
 assert union[i,0]==union[i,1] and 0<maximum[i,0]<maximum[i,1]
 assert maximum[i,1]==union[i,1] and np.all(maximum[:,3]==0)
 engine=Phrank(str(ontology/'hpo-dag.tsv'),diseaseannotationsfile=str(dmax/'training-disease-hpo.tsv'),
               diseasegenefile=str(dmax/'training-gene-disease.tsv'))
 for i,d in enumerate(ds):
  reference={g:s for s,g in engine.rank_genes_using_disease(set(genes),profiles[d])}
  np.testing.assert_allclose(maximum[i],[reference.get(g,0.) for g in genes],rtol=1e-12,atol=1e-12)
 for path,mode in [(direct,'direct'),(dmax,'disease-max')]:
  evidence=json.loads((path/'provenance.json').read_text())
  assert evidence['scoring_mode']==mode and evidence['held_out_annotation_sources']==0
  assert evidence['author_score_parity_checks']==8
  for name,digest in json.loads((path/'checksums.json').read_text()).items():assert a.sha(path/name)==digest
 with pytest.raises(FileExistsError):
  a.run_fold(data.parent,ontology,author_root,tmp_path/'disease-max',0,summary,'disease-max')
 (ontology/'hpo-dag.tsv').write_text('changed')
 with pytest.raises(ValueError,match='Ontology input changed'):
  a.run_fold(data.parent,ontology,author_root,tmp_path/'corrupted',0,summary,'disease-max')
