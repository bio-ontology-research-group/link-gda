import argparse,collections,json,sys,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phenoquery.projection import phenotype,allowed,digest
VARIANTS=['bridge','plain','entity','typed','structure']
def load(p):
 with p.open() as f:return {tuple(l.rstrip().split('\t')) for l in f}
def disease(s):return s.startswith('http://mowl.borg/') and not s.rsplit('/',1)[-1].isdigit()
def gene(s):return s.startswith('http://mowl.borg/') and s.rsplit('/',1)[-1].isdigit()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--ultra-root',type=Path,required=True);p.add_argument('--folds',type=int,nargs='+',default=list(range(10)));a=p.parse_args();sys.path.insert(0,str(a.ultra_root));from ultra.tasks import build_relation_graph
 if not a.folds or len(set(a.folds))!=len(a.folds) or min(a.folds)<0 or max(a.folds)>9:p.error('Choose distinct fold IDs in 0..9')
 root=a.root;proj=root/'projection';manifest=json.loads((proj/'manifest.json').read_text());
 for n,h in manifest['files'].items():assert digest(proj/n)==h
 extra={k:load(proj/(k+'.tsv')) for k in ['backbone','entity','typed','structure']}
 for fold in a.folds:
  d=a.base/'data'/f'fold{fold}';out=root/'data'/f'fold{fold}';out.mkdir(parents=True,exist_ok=False);old=json.loads((d/'preparation.json').read_text())
  for filename,key in [('profiles.json','profiles'),('candidates.txt','candidates'),('training.tsv','training'),('validation.tsv','validation'),('test.tsv','test'),('raw/train.txt','graph')]:assert digest(d/filename)==old[key+'_sha256']
  for filename in ['profiles.json','candidates.txt','training.tsv','validation.tsv','test.tsv']:shutil.copy2(d/filename,out/filename)
  prior=load(d/'raw/train.txt');base=set();bridge=set();removed=collections.Counter()
  for h,r,t in prior:
   if r=='expressed_in':removed['expression']+=1;continue
   if disease(h) or disease(t):
    if r in ['associated_with','has_symptom']:bridge.add((h,r,t))
    removed['disease']+=1;continue
   if not ((allowed(h) or gene(h)) and (allowed(t) or gene(t))):removed['outside_core']+=1;continue
   # All phenotype↔ontology EQ routes are removed from the baseline.
   if phenotype(h) or phenotype(t):
    if gene(h) and phenotype(t) and r in ['has_phenotype','causes_phenotype']:base.add((h,r,t))
    elif phenotype(h) and phenotype(t) and r in ['http://subclassof','http://superclassof']:base.add((h,r,t))
    else:removed['phenotype_projection']+=1
   elif r!='http://superclassof':base.add((h,r,t))
  # Canonical taxonomy direction; automatic inverse edges are created below.
  base={(t,'http://subclassof',h) if r=='http://superclassof' else (h,r,t) for h,r,t in base}|extra['backbone']
  genes=(out/'candidates.txt').read_text().splitlines();profiles=json.loads((out/'profiles.json').read_text());allps={p for ps in profiles.values() for p in ps}
  for variant in VARIANTS:
   target=out/variant;target.mkdir(exist_ok=True)
   es=base|(bridge if variant=='bridge' else extra.get(variant,set()))
   nodes=sorted({x for h,r,t in es for x in [h,t]}|set(genes)|allps);ei={x:i for i,x in enumerate(nodes)};rs=sorted({r for h,r,t in es});ri={r:i for i,r in enumerate(rs)};triples=sorted(es);edge_index=torch.tensor([[ei[h],ei[t]] for h,r,t in triples],dtype=torch.long).T;types=torch.tensor([ri[r] for h,r,t in triples]);graph=Data(edge_index=torch.cat([edge_index,edge_index.flip(0)],1),edge_type=torch.cat([types,types+len(rs)]),num_nodes=len(nodes),num_relations=2*len(rs));graph=build_relation_graph(graph)
   assert all(g in ei for g in genes)
   if variant!='bridge':assert not any(disease(x) for x in nodes)
   torch.save(graph,target/'graph.pt');(target/'vocabulary.json').write_text(json.dumps({'entities':ei,'relations':ri},separators=(',',':')))
   (target/'identity.json').write_text(json.dumps({'fold':fold,'variant':variant,'nodes':len(nodes),'direct_edges':len(es),'relations':len(rs),'graph_sha256':digest(target/'graph.pt'),'vocabulary_sha256':digest(target/'vocabulary.json'),'projection_manifest_sha256':digest(proj/'manifest.json'),'removed':dict(removed)},indent=2)+'\n');print('PREPARED',fold,variant,len(nodes),len(es),flush=True)
  (out/'identity.json').write_text(json.dumps({'prior':old,'files':{n:digest(out/n) for n in ['profiles.json','candidates.txt','training.tsv','validation.tsv','test.tsv']},'projection_manifest_sha256':digest(proj/'manifest.json')},indent=2)+'\n')
 (root/'PREPARED.json').write_text(json.dumps({'folds':len(a.folds),'fold_ids':a.folds,'variants':VARIANTS,'projection_manifest_sha256':digest(proj/'manifest.json'),'source_sha256':{'prepare_phenoquery.py':digest(Path(__file__))}},indent=2)+'\n')
if __name__=='__main__':main()
