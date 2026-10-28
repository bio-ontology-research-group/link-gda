#!/usr/bin/env python3
"""Fold-isolated PhenoRank adapter, preserving published scoring and simulation."""
import argparse,csv,hashlib,json,pickle,time,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
import native_scoring as native

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(p.read_text())
def rows(p):
 with p.open() as f:return [(r['Gene'],r['Disease']) for r in csv.DictReader(f,delimiter='\t')]
def dictionary(p):
 return {int(a):[int(t) for t in b.split('|') if t] for a,b in (r.split('\t') for r in p.read_text().splitlines()[1:])}
def load(p):
 with p.open('rb') as f:return pickle.load(f,encoding='latin1')
def term(t):return t.rsplit('/',1)[-1].replace('_',':')

def make_profile_mapping(profiles,known,anc,graph):
 parents=defaultdict(set)
 with graph.open() as f:
  for row in f:
   h,r,t=row.rstrip('\n').split('\t')
   if r=='http://superclassof':parents[term(t)].add(term(h))
   elif r in ('is_a','http://www.w3.org/2000/01/rdf-schema#subClassOf'):parents[term(h)].add(term(t))
 mapping={}
 for t in sorted({term(x) for ps in profiles.values() for x in ps}):
  if t in known:mapping[t]=[known[t]];continue
  seen=set();todo=[t];found=set()
  while todo:
   x=todo.pop()
   if x in seen:continue
   seen.add(x)
   if x in known:found.add(known[x])
   else:todo.extend(parents.get(x,()))
  # Keep most specific available terms; ontology only, no gene labels involved.
  mapping[t]=sorted(found-{a for x in found for a in anc.get(x,[]) if a!=x})
 return {d:sorted({i for t in ps for i in mapping[term(t)]}) for d,ps in profiles.items()},mapping

def build(data,resources):
 p=resources/'original';genes=(p/'phenorank_genes.tsv').read_text().splitlines();phenos=(p/'phenorank_phenotypes.tsv').read_text().splitlines();pi={t:i for i,t in enumerate(phenos)};anc=dictionary(p/'phenotype_ancestors.tsv')
 profiles,mapping=make_profile_mapping(read(data/'profiles.json'),pi,anc,data/'raw/train.txt');train=rows(data/'training.tsv');val=rows(data/'validation.tsv');test=rows(data/'test.tsv')
 train_d={d for _,d in train};held={d for _,d in val+test}
 if train_d&held:raise ValueError('Disease holdout violated')
 candidates=(data/'candidates.txt').read_text().splitlines();idmap={};reverse=defaultdict(set)
 with (resources/'hgnc_complete_set.txt').open() as f:
  for r in csv.DictReader(f,delimiter='\t'):
   if r['status']=='Approved' and r['entrez_id'] and r['ensembl_gene_id']:reverse[r['entrez_id']].add(r['ensembl_gene_id'])
 for g in candidates:
  hits=reverse.get(g.rsplit('/',1)[-1],set());idmap[g]=next(iter(hits)) if len(hits)==1 else 'UNMAPPED:'+g
 if len(set(idmap.values()))!=len(idmap):raise ValueError('Multiple candidate genes map to same Ensembl ID')
 original_count=len(genes);genes+=sorted(set(idmap.values())-set(genes));gi={g:i for i,g in enumerate(genes)};outix=np.array([gi[idmap[g]] for g in candidates])
 mouse=dictionary(p/'cp_m.tsv');oldconditions=(p/'phenorank_conditions.tsv').read_text().splitlines();mousekeys=sorted(k for k,v in mouse.items() if v);humankeys=sorted(d for d in train_d if profiles[d]);conds=[mouse[k] for k in mousekeys]+[profiles[d] for d in humankeys];ci={d:i+len(mousekeys) for i,d in enumerate(humankeys)}
 rr=[];cc=[]
 for j,ps in enumerate(conds):
  expanded=set(native.add_ancestors(ps,anc));rr.extend(expanded);cc.extend([j]*len(expanded))
 annotations=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(phenos),len(conds)));freq=np.asarray(annotations.sum(1)).ravel();ic=np.zeros(len(phenos));use=freq>0;ic[use]=-np.log(freq[use]/len(conds));matrix=annotations.multiply(ic[:,None]).tocsr();condition_ic=np.asarray(matrix.sum(0)).ravel()
 hr=[];hc=[]
 for g,d in train:
  if d in ci:hr.append(gi[idmap[g]]);hc.append(ci[d])
 gh=sparse.csr_matrix((np.ones(len(hr)),(hr,hc)),shape=(len(genes),len(conds)));gh.data[:]=1
 gm0=load(p/'gc_m.pickle')[:,mousekeys];gm=sparse.vstack([sparse.hstack([gm0,sparse.csr_matrix((original_count,len(humankeys)))]),sparse.csr_matrix((len(genes)-original_count,len(conds)))]).tocsr()
 W0=load(p/'W_generic.pickle');W=sparse.block_diag((W0,sparse.csr_matrix((len(genes)-original_count,)*2)),format='csr')
 rr=[];cc=[]
 for j,d in enumerate(humankeys):rr.extend(profiles[d]);cc.extend([j]*len(profiles[d]))
 direct=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(phenos),len(humankeys)))
 co=(direct@direct.T).tocsr();cooccur={}
 for k in np.flatnonzero(np.diff(co.indptr)):
  a,b=co.indptr[k:k+2];vals=co.data[a:b];groups=defaultdict(list)
  for idx,v in zip(co.indices[a:b],vals):groups[float(v/vals.sum())].append(int(idx))
  cooccur[int(k)]=dict(groups)
 if not cooccur:raise ValueError('No training simulation reference')
 identity=read(data/'preparation.json')
 for n,k in [('profiles.json','profiles'),('training.tsv','training'),('validation.tsv','validation'),('test.tsv','test'),('candidates.txt','candidates'),('raw/train.txt','graph')]:
  if sha(data/n)!=identity[k+'_sha256']:raise ValueError('Prepared input changed '+n)
 audit={'training_diseases':sorted(train_d),'reference_human_diseases':humankeys,'held_out_diseases':sorted(held),'mouse_conditions':len(mousekeys),'gene_count':len(genes),'candidate_count':len(candidates),'native_supported_candidates':int((outix<original_count).sum()),'appended_candidates':int((outix>=original_count).sum()),'hgnc_ambiguous_or_missing':sum(v.startswith('UNMAPPED:') for v in idmap.values()),'candidate_mapping':idmap,'phenotype_projection':{t:[phenos[i] for i in ids] for t,ids in mapping.items() if t not in pi},'unmapped_terms':[t for t,v in mapping.items() if not v],'unsupported_profiles':[d for d,ps in profiles.items() if not ps],'source_sha256':{f.name:sha(f) for f in p.iterdir() if f.is_file()},'hgnc_sha256':sha(resources/'hgnc_complete_set.txt'),'preparation_sha256':sha(data/'preparation.json')}
 no_evidence=np.asarray((gh.sum(1)+gm.sum(1))).ravel()==0;no_evidence &= np.asarray(W.sum(1)).ravel()==0
 audit['no_evidence_candidates']=[candidates[j] for j,i in enumerate(outix) if no_evidence[i]]
 return {'no_evidence':no_evidence[outix],'profiles':profiles,'anc':anc,'ic':ic,'condition_ic':condition_ic,'matrix':matrix,'gh':gh,'gm':gm,'W':W,'co':cooccur,'outix':outix,'candidates':candidates,'test':test,'audit':audit}

def score(model,profile):
 return native.score_genes(profile,model['anc'],model['ic'],model['condition_ic'],model['matrix'],model['gh'],model['gm'],model['W'],.1,20)

def evaluate_one(model,profile,seed,nperm=1000):
 if not profile:return np.zeros(len(model['candidates'])),np.full(len(model['candidates']),-1.)
 np.random.seed(seed);_,raw,observed=score(model,profile);counts=np.zeros(len(observed),np.int32)
 for _ in range(nperm):
  simulated=native.simulate_disease(len(profile),model['co']);_,_,ranks=score(model,simulated);counts+=ranks>=observed
 p=np.maximum(counts,1)/float(nperm);ix=model['outix'];result=-p[ix];result[model.get('no_evidence',np.zeros(len(ix),bool))]=-1.;return raw[ix],result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--data-root',type=Path);ap.add_argument('--resources',type=Path);ap.add_argument('--output-root',type=Path);ap.add_argument('--summary-path',type=Path,default=Path(__file__).resolve().parents[2]/'analysis/summarize_ultra_campaign.py');ap.add_argument('--pilot',action='store_true');ap.add_argument('--pilot-simulations',type=int,default=1000);a=ap.parse_args();start=time.time();base=a.output_root or a.root/'baselines/phenorank';base.mkdir(parents=True,exist_ok=True);data=(a.data_root or a.root/'data')/f'fold{a.fold}';resources=a.resources or a.root/'baselines/phenorank/resources';protocol=Path(__file__).with_name('PROTOCOL.md');out=base/(f'pilot-fold{a.fold}' if a.pilot else f'fold{a.fold}');out.mkdir(exist_ok=True)
 config={'fold':a.fold,'seed':0,'nperm':a.pilot_simulations if a.pilot else 1000,'restart':.1,'iterations':20,'technical_pilot':a.pilot,'run_sha256':sha(Path(__file__)),'native_scoring_sha256':sha(Path(__file__).with_name('native_scoring.py')),'protocol_sha256':sha(protocol)}
 if (out/'config.json').exists() and read(out/'config.json')!=config:raise ValueError('Resume configuration/source changed')
 if (out/'completion.json').exists():raise ValueError('Completed fold must not be rerun')
 write(out/'config.json',config);(out/'PROTOCOL.md').write_bytes(protocol.read_bytes());model=build(data,resources);write(out/'resource-audit.json',model['audit']);print('BUILT',time.time()-start,flush=True)
 partial=out/'partial';partial.mkdir(exist_ok=True)
 diseases=sorted({d for _,d in model['test']});diseases=diseases[:1] if a.pilot else diseases;raw=[];native_scores=[]
 for j,d in enumerate(diseases):
  t=time.time();seed=int.from_bytes(hashlib.sha256(f'fold{a.fold}:seed0:{d}'.encode()).digest()[:4],'little');chunk=partial/f'query{j:04d}.npz';stamp=partial/f'query{j:04d}.json'
  if chunk.exists() and stamp.exists():
   meta=read(stamp)
   if meta!={'disease':d,'seed':seed,'sha256':sha(chunk)}:raise ValueError('Partial query provenance mismatch')
   with np.load(chunk,allow_pickle=False) as z:r=z['raw'];n=z['native']
  else:
   r,n=evaluate_one(model,model['profiles'][d],seed,a.pilot_simulations if a.pilot else 1000)
   temp=partial/f'query{j:04d}.tmp.npz';np.savez_compressed(temp,raw=r,native=n);temp.replace(chunk);write(stamp,{'disease':d,'seed':seed,'sha256':sha(chunk)})
  if r.shape!=(len(model['candidates']),) or n.shape!=r.shape or not np.isfinite(r).all() or not np.isfinite(n).all():raise ValueError('Invalid partial score')
  raw.append(r);native_scores.append(n)
  with (out/'progress.jsonl').open('a') as f:f.write(json.dumps({'query':j+1,'total':len(diseases),'disease':d,'terms':len(model['profiles'][d]),'seconds':time.time()-t})+'\n')
  print('QUERY',j+1,len(diseases),'seconds',time.time()-t,flush=True)
 if a.pilot:write(out/'pilot-completion.json',{'technical_pilot':True,'seconds':time.time()-start,'queries':len(diseases)});return
 import importlib.util
 spec=importlib.util.spec_from_file_location('summary',a.summary_path);summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
 def save(directory,scores):
  directory.mkdir(exist_ok=True);np.savez_compressed(directory/'scores.npz',scores=np.array(scores),diseases=np.array(diseases),genes=np.array(model['candidates']))
  for name in ['candidates.txt','pairs.tsv']:(directory/name).write_bytes((data/('test.tsv' if name=='pairs.tsv' else name)).read_bytes())
  ms={mode:summary.metrics(summary.rank_pairs(values,diseases,model['candidates'],model['test'])) for mode,values in [('raw',np.array(scores)),('calibrated',summary.calibrated(scores))]};write(directory/'metrics.json',ms);write(directory/'checksums.json',{n:sha(directory/n) for n in ['scores.npz','candidates.txt','pairs.tsv','metrics.json']})
 save(out,native_scores);save(out/'raw-propagation',raw)
 (out/'raw-propagation-checksums.json').write_bytes((out/'raw-propagation/checksums.json').read_bytes())
 manifest=read(out/'checksums.json');manifest.update({n:sha(out/n) for n in ['config.json','resource-audit.json','PROTOCOL.md','raw-propagation-checksums.json']});write(out/'checksums.json',manifest)
 write(out/'completion.json',{'status':'complete','fold':a.fold,'checksums_sha256':sha(out/'checksums.json'),'preparation_sha256':sha(data/'preparation.json'),'resource_audit_sha256':sha(out/'resource-audit.json'),'native_primary':'negative empirical simulated-query pvalue; floor1/1000; average ties','seconds':time.time()-start,'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}})
if __name__=='__main__':main()
