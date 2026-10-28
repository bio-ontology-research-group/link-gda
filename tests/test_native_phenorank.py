import importlib.util,sys,json,pickle,csv,subprocess
import pytest
from pathlib import Path
import numpy as np
from scipy import sparse
P=Path(__file__).resolve().parents[1]/'code/baselines/native_phenorank';sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('phenorank_adapter',P/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def test_projection_uses_only_ontology_and_most_specific(tmp_path):
 graph=tmp_path/'graph';graph.write_text('http://x/HP_1\thttp://superclassof\thttp://x/HP_2\nhttp://x/HP_2\thttp://superclassof\thttp://x/HP_3\nhttp://x/HP_1\thttp://superclassof\thttp://x/HP_3\nhttp://x/HP_4\tcauses_phenotype\thttp://x/HP_3\n')
 out,m=r.make_profile_mapping({'d':['http://x/HP_3'],'e':['http://x/HP_5']},{'HP:1':0,'HP:2':1,'HP:4':2},{1:[0]},graph)
 assert out=={'d':[1],'e':[]}

def test_native_scoring_matches_explicit_formula():
 anc={0:[],1:[0],2:[0]};ic=np.array([0.,2.,3.]);mat=sparse.csr_matrix([[0,0],[2,0],[0,3]])
 gh=sparse.eye(2,format='csr');gm=sparse.csr_matrix((2,2));W=sparse.csr_matrix([[0.,1.],[1.,0.]])
 initial,prop,rank=r.native.score_genes([1],anc,ic,np.array([2.,3.]),mat,gh,gm,W,.1,20)
 np.testing.assert_array_equal(initial,[1.,0.]);expected=initial.copy()
 for _ in range(20):expected=.9*(W@expected)+.1*initial
 np.testing.assert_array_equal(prop,expected);np.testing.assert_array_equal(rank,[2.,1.])

def test_simulation_pvalues_use_rank_and_floor(monkeypatch):
 model={'candidates':['a','b'],'outix':np.array([0,1]),'co':{0:{1.:[0]}}}
 seq=iter([(None,np.array([4.,1.]),np.array([2.,1.])),(None,None,np.array([1.,2.])),(None,None,np.array([1.,2.]))])
 monkeypatch.setattr(r,'score',lambda *args:next(seq));raw,p=r.evaluate_one(model,[0],17,2)
 np.testing.assert_array_equal(raw,[4.,1.]);np.testing.assert_array_equal(p,[-.5,-1.])

def test_unsupported_profile_keeps_every_candidate():
 a,b=r.evaluate_one({'candidates':['a','b','c']},[],0)
 np.testing.assert_array_equal(a,[0,0,0]);np.testing.assert_array_equal(b,[-1,-1,-1])

def make_inputs(tmp_path):
 data=tmp_path/'data';data.mkdir();res=tmp_path/'resources';p=res/'original';p.mkdir(parents=True)
 (p/'phenorank_genes.tsv').write_text('ENSG1\nENSG2\n');(p/'phenorank_phenotypes.tsv').write_text('HP:1\nHP:2\nHP:3\n');(p/'phenorank_conditions.tsv').write_text('mouse0\n')
 (p/'phenotype_ancestors.tsv').write_text('key\tvalue\n0\t\n1\t0\n2\t0\n');(p/'cp_m.tsv').write_text('key\tvalue\n0\t1\n')
 for name,x in [('gc_m',sparse.csr_matrix([[1],[0]])),('W_generic',sparse.eye(2,format='csr'))]:
  with (p/(name+'.pickle')).open('wb') as f:pickle.dump(x,f)
 (res/'hgnc_complete_set.txt').write_text('status\tentrez_id\tensembl_gene_id\nApproved\t1\tENSG1\nApproved\t2\tENSG2\n')
 (data/'candidates.txt').write_text('http://mowl.borg/1\nhttp://mowl.borg/2\nhttp://mowl.borg/3\n')
 r.write(data/'profiles.json',{'train':['http://x/HP_2'],'val':['http://x/HP_3'],'test':['http://x/HP_3']})
 for split,d,g in [('training','train','1'),('validation','val','2'),('test','test','3')]:
  (data/(split+'.tsv')).write_text(f'Gene\tDisease\nhttp://mowl.borg/{g}\t{d}\n')
 (data/'raw').mkdir();(data/'raw/train.txt').write_text('http://x/HP_1\thttp://superclassof\thttp://x/HP_2\n')
 r.write(data/'preparation.json',{k+'_sha256':r.sha(data/n) for n,k in [('profiles.json','profiles'),('training.tsv','training'),('validation.tsv','validation'),('test.tsv','test'),('candidates.txt','candidates'),('raw/train.txt','graph')]})
 return data,res

def test_training_mask_and_zero_information(tmp_path):
 data,res=make_inputs(tmp_path);model=r.build(data,res)
 assert model['audit']['reference_human_diseases']==['train']
 assert model['ic'][2]==0 and model['matrix'][2].nnz==0
 assert set(model['co'])=={1}
 assert len(model['candidates'])==3
 assert model['audit']['appended_candidates']==1

def test_zero_information_union_uses_native_fallback(tmp_path):
 data,res=make_inputs(tmp_path);model=r.build(data,res)
 before=np.geterr().copy()
 raw,probability=r.evaluate_one(model,model['profiles']['test'],0,nperm=2)
 np.testing.assert_array_equal(raw,np.zeros(3))
 np.testing.assert_array_equal(probability,-np.ones(3))
 assert np.geterr()==before

@pytest.mark.parametrize('mutation',['resource','preparation','missing-audit'])
def test_cli_resume_binds_partial_queries_to_inputs(tmp_path,mutation):
 data,res=make_inputs(tmp_path)
 (res/'original/cp_m.tsv').write_text('key\tvalue\n0\t2\n')
 profiles=r.read(data/'profiles.json');profiles['test']=['http://x/HP_2'];r.write(data/'profiles.json',profiles)
 identity=r.read(data/'preparation.json');identity['profiles_sha256']=r.sha(data/'profiles.json');r.write(data/'preparation.json',identity)
 data.rename(tmp_path/'fold0')
 out=tmp_path/'output'
 cmd=[sys.executable,str(P/'run.py'),'--root',str(tmp_path),'--data-root',str(tmp_path),'--resources',str(res),'--output-root',str(out),'--fold','0','--pilot','--pilot-simulations','2']
 first=subprocess.run(cmd,capture_output=True,text=True)
 assert first.returncode==0,first.stderr
 fold=out/'pilot-fold0';chunk=fold/'partial/query0000.npz';stamp=fold/'partial/query0000.json';audit=fold/'resource-audit.json'
 before={p:p.read_bytes() for p in [chunk,stamp,audit]}
 mtime=chunk.stat().st_mtime_ns
 resumed=subprocess.run(cmd,capture_output=True,text=True)
 assert resumed.returncode==0,resumed.stderr
 assert chunk.stat().st_mtime_ns==mtime
 assert all(p.read_bytes()==b for p,b in before.items())
 if mutation=='resource':
  (res/'original/cp_m.tsv').write_text('key\tvalue\n0\t1\n')
 elif mutation=='preparation':
  prep=tmp_path/'fold0/preparation.json';identity=r.read(prep);identity['provenance_revision']='changed';r.write(prep,identity)
 else:audit.unlink()
 rejected=subprocess.run(cmd,capture_output=True,text=True)
 assert rejected.returncode!=0
 assert ('Resume input/resource provenance changed' if mutation!='missing-audit' else 'Partial queries lack input/resource provenance') in rejected.stderr
 assert chunk.read_bytes()==before[chunk] and stamp.read_bytes()==before[stamp]
 if mutation!='missing-audit':assert audit.read_bytes()==before[audit]
 else:assert not audit.exists()
