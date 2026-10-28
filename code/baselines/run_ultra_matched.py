import argparse,json,logging,math,sys,time
from pathlib import Path
import numpy as np
import torch
from support import *

def setup(a,c):
 d=a.base/'data'/f"fold{c['fold']}";identity=read(d/'preparation.json')
 for n,k in [('profiles.json','profiles'),('candidates.txt','candidates'),('training.tsv','training'),('validation.tsv','validation'),('test.tsv','test'),('raw/train.txt','graph'),('processed/data.pt','processed'),('vocabulary.json','vocabulary')]:assert sha(d/n)==identity[k+'_sha256'],n
 from score_ultra_base import build_dataset,load_model_config
 if c['graph']=='original':ds,entities,rels=build_dataset(str(a.base/'source/ULTRA'),str(a.base/'data'),f"fold{c['fold']}");graph=ds[0]
 else:
  g=a.root/'graphs'/f"fold{c['fold']}";gi=read(g/'identity.json');assert gi['source_preparation']==identity
  for n,h in gi['files'].items():assert sha(g/n)==h
  v=read(g/'vocabulary.json');entities=v['entities'];rels=v['relations'];graph=torch.load(g/'graph.pt',map_location='cpu')
 from ultra.models import Ultra
 mc=load_model_config(str(a.base/'source/ULTRA'));model=Ultra(rel_model_cfg=mc['relation_model'],entity_model_cfg=mc['entity_model']).cuda().eval();ck=a.base/'source/ULTRA/ckpts'/('ultra_4g.pth' if c['cohort']=='general' else 'ultraquery.pth');state=load_initial(ck,c['cohort']);model.load_state_dict(state,strict=True)
 genes=(d/'candidates.txt').read_text().splitlines();profiles=read(d/'profiles.json');rows={s:pairs(d/(s+'.tsv')) for s in ['training','validation','test']};assert len(genes)==13625;assert not {di for g,di in rows['validation']+rows['test']}&set(entities)
 return d,identity,graph.cuda(),entities,rels,genes,profiles,rows,model,sha(ck)
def output(a,c):return a.root/c['cohort']/c['name']
def phase_scores(model,graph,e,r,genes,profiles,rows):
 from score_ultra_base import score_phenotypes
 ps=sorted({p for g,d in rows for p in profiles[d]});x=score_phenotypes(model,graph,[e[p] for p in ps],[e[g] for g in genes],r['causes_phenotype']+len(r),8,'cuda');return x,ps

def cache(a,c):
 out=output(a,c);out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():return
 d,identity,graph,e,r,genes,profiles,rows,model,h=setup(a,c);x,ps=phase_scores(model,graph,e,r,genes,profiles,rows['validation']);np.savez_compressed(out/'validation.npz',scores=x,phenotypes=np.array(ps),genes=np.array(genes));grids={}
 for kind,params in [('pool',np.linspace(0,1,11)),('and',INPUT_TEMPS)]:
  grids[kind]=[{'parameter':float(p),'metrics':evaluate(x,ps,profiles,genes,rows['validation'],kind,float(p))} for p in params]
 if c['graph']=='original' and c['cohort']=='general':
  old=read(a.base/'runs'/f"cache-fold{c['fold']}"/'completion.json');expected={(round(v['weight'],6),v['scoring']):v['metrics'] for v in old['grid']}
  for v in grids['pool']:
   for mode,m in v['metrics'].items():
    for k,value in m.items():assert np.isclose(value,expected[(round(v['parameter'],6),mode)][k],rtol=1e-12,atol=1e-12),(mode,k,value,expected[(round(v['parameter'],6),mode)][k])
  write(out/'historical-parity.json',{'all_22_validation_settings_match':True})
 write(out/'COMPLETE.json',{'config':c,'grids':grids,'identity':identity,'checkpoint_sha256':h,'cache_sha256':sha(out/'validation.npz'),'source_sha256':sha(a.root/'source.sha256'),'protocol_sha256':sha(a.root/'PROTOCOL.md')});print('CACHE_COMPLETE',c['name'],flush=True)

def train(a,c):
 out=output(a,c);out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():return
 pilot=c['phase']=='pilot';islink=c.get('training_phase',c['phase'])=='link';torch.manual_seed(0);np.random.seed(0);rng=np.random.default_rng(0);d,identity,graph,e,r,genes,profiles,rows,model,initialhash=setup(a,c);cell=c['graph'];co=a.root/c['cohort'];kind=c.get('readout','pool')
 if pilot:options={'pool':{'parameter':0.,'scoring':'calibrated'},'and':{'parameter':1.,'scoring':'calibrated'}}
 else:options=read(co/'stageA.json')['settings'][cell]
 param=options[kind]['parameter'];weight=max(.001,min(.999,param)) if kind=='pool' else .5;pool=torch.nn.Parameter(torch.tensor(math.log(weight/(1-weight)),device='cuda'),requires_grad=not islink and kind=='pool');params=list(model.parameters())+([pool] if pool.requires_grad else []);optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.)
 if not islink and not pilot:
  stage2=read(co/'stageB.json');mode=stage2['settings'][cell][kind]['scoring'];parent=co/f"link-{cell}-fold{c['fold']}";pc=read(parent/'COMPLETE.json');keyname=kind+'_'+mode;path=parent/(keyname+'.pt');assert sha(path)==pc['best'][keyname]['checkpoint_sha256'];model.load_state_dict(torch.load(path,map_location='cpu')['model']);write(out/'parent.json',{'checkpoint_sha256':sha(path),'parent_stage':sha(co/'stageB.json'),'key':keyname})
 validation=rows['validation']
 if pilot:
  chosen=set(sorted({di for g,di in validation})[:4]);validation=[(g,di) for g,di in validation if di in chosen]
 forward=r['causes_phenotype'];candidateids=[e[g] for g in genes];positive={};dg={};sources={};candidate_set=set(candidateids)
 for h,t in graph.edge_index[:,graph.edge_type==forward].T.tolist():
  if h in candidate_set:positive.setdefault(t,set()).add(h)
 for g,di in rows['training']:dg.setdefault(di,set()).add(e[g])
 for di in dg:
  for p in set(profiles[di]):sources.setdefault(p,set()).add(e[di])
 ds=sorted(dg);best={};curve=[];last=0;elapsed=0;started=time.time();initialweights=modelhash(model)
 def state(step):return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'pool':float(pool.detach()),'step':step,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'best':best,'curve':curve,'elapsed':elapsed+time.time()-started}
 def validate(step):
  model.eval();x,ps=phase_scores(model,graph,e,r,genes,profiles,validation);entry={'step':step,'metrics':{}}
  for rd in READOUTS if islink else [kind]:
   p=options[rd]['parameter'] if islink or rd=='and' else float(pool.sigmoid());both=evaluate(x,ps,profiles,genes,validation,rd,p)
   for mode,m in both.items():
    tag=rd+'_'+mode;entry['metrics'][tag]=m
    if tag not in best or key(m)>key(best[tag]['metrics']):
     path=out/(tag+'.pt');save(path,{'model':model.state_dict(),'pool':float(pool.detach()),'parameter':float(p),'step':step});best[tag]={'step':step,'metrics':m,'parameter':float(p),'checkpoint_sha256':sha(path)}
  curve.append(entry);write(out/'progress.json',{'step':step,'best':best,'curve':curve,'elapsed_seconds':elapsed+time.time()-started});save(out/'resume.pt',state(step));print('VALIDATION',c['name'],step,entry['metrics'],flush=True)
 if (out/'resume.pt').exists():
  st=torch.load(out/'resume.pt',map_location='cpu');model.load_state_dict(st['model']);optimizer.load_state_dict(st['optimizer']);pool.data.fill_(st['pool']);rng.bit_generator.state=st['rng'];torch.set_rng_state(st['torch_rng']);torch.cuda.set_rng_state(st['cuda_rng']);best=st['best'];curve=st['curve'];last=st['step'];elapsed=st['elapsed']
 else:validate(0)
 from ultra_objectives import prepare_mask,stage2_backward,profile_backward
 for step in range(last+1,c['steps']+1):
  di=ds[int(rng.integers(len(ds)))];ps=profiles[di]
  if islink:ps=[ps[int(rng.integers(len(ps)))]]
  masked=prepare_mask(graph,[e[p] for p in ps],forward,disease_id=sorted(set().union(*(sources[p] for p in ps))));optimizer.zero_grad(set_to_none=True);tic=time.time()
  if islink:stats=stage2_backward(model,masked,e[ps[0]],sorted(positive[e[ps[0]]]),candidateids,rng)
  elif kind=='pool':stats=profile_backward(model,masked,[e[p] for p in ps],sorted(dg[di]),candidateids,pool,rng,temperature=c['loss_temperature'])
  else:stats=and_backward(model,masked,[e[p] for p in ps],sorted(dg[di]),candidateids,pool,rng,param,c['loss_temperature'])
  if not np.isfinite(stats['loss']):raise FloatingPointError('Nonfinite loss')
  norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();del masked
  with (out/'training.jsonl').open('a') as f:f.write(json.dumps({'step':step,'disease':di,'phenotypes':len(ps),'loss':stats['loss'],'gradient_norm':float(norm),'seconds':time.time()-tic})+'\n')
  if step%10==0 or pilot:
   save(out/'resume.pt',state(step));write(out/'heartbeat.json',{'step':step,'steps':c['steps'],'updated':time.time(),'loss':stats['loss']});print('TRAIN',c['name'],step,stats['loss'],flush=True)
  if step%c['eval_every']==0 or step==c['steps']:validate(step)
 if pilot:assert initialweights!=modelhash(model)
 write(out/'COMPLETE.json',{'config':c,'best':best,'identity':identity,'initial_sha256':initialhash,'parameter_count':sum(p.numel() for p in model.parameters()),'source_sha256':sha(a.root/'source.sha256'),'protocol_sha256':sha(a.root/'PROTOCOL.md'),'elapsed':elapsed+time.time()-started,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'nonzero_update':initialweights!=modelhash(model)});print('TRAIN_COMPLETE',c['name'],flush=True)

def score(a,c):
 co=a.root/c['cohort'];freeze=read(co/'FINAL-SELECTION.json');assert sha(co/'FINAL-SELECTION.json')==c['selection_sha256'];out=output(a,c);out.mkdir(parents=True,exist_ok=True);stageA=read(co/'stageA.json');records={}
 for graphname in GRAPHS:
  local={**c,'graph':graphname};d,identity,graph,e,r,genes,profiles,rows,model,_=setup(a,local);x,ps=phase_scores(model,graph,e,r,genes,profiles,rows['test']);models={modelhash(model):(x,ps)};bundles=[]
  for rd in READOUTS:
   settings=stageA['settings'][graphname][rd];bundles.append((graphname+'_'+rd+'_frozen',x,ps,rd,settings['parameter'],settings['scoring']))
   selected=freeze['runs'][graphname+'_'+rd+f"-fold{c['fold']}"];p=co/selected['directory']/(rd+'_'+selected['scoring']+'.pt');assert sha(p)==selected['checkpoint_sha256'];st=torch.load(p,map_location='cpu');model.load_state_dict(st['model']);mh=modelhash(model)
   if mh not in models:models[mh]=phase_scores(model,graph,e,r,genes,profiles,rows['test'])
   xx,pp=models[mh];bundles.append((graphname+'_'+rd+'_adapted',xx,pp,rd,st['parameter'],selected['scoring']))
  for name,xx,pp,rd,parameter,mode in bundles:
   directory=out/name;directory.mkdir(exist_ok=True);ds=sorted({di for g,di in rows['test']});values=aggregate(xx,pp,profiles,ds,rd,parameter);rr={m:ranks(v,ds,genes,rows['test']) for m,v in [('raw',values),('calibrated',cal(values))]};seen={g for g,di in rows['training']};np.savez_compressed(directory/'scores.npz',scores=values,diseases=np.array(ds),genes=np.array(genes));np.savez_compressed(directory/'ranks.npz',**rr,primary=rr[mode],genes=np.array([g for g,di in rows['test']]),diseases=np.array([di for g,di in rows['test']]),seen=np.array([g in seen for g,di in rows['test']]))
   records[name]={'metrics':{k:metrics(v) for k,v in rr.items()},'primary_mode':mode,'files':{n:sha(directory/n) for n in ['scores.npz','ranks.npz']}};write(directory/'COMPLETE.json',records[name])
  # Exact historical reference: original selected profile checkpoint and its pool.
  if graphname=='original' and c['cohort']=='general':
   old=a.base/'runs'/f"profile-fold{c['fold']}";oldc=read(old/'completion.json');mode=read(a.base/'selection-stage3.json')['scoring'];p=old/Path(oldc['best'][mode]['checkpoint']).name;st=torch.load(p,map_location='cpu');model.load_state_dict(st['model']);mh=modelhash(model)
   if mh not in models:models[mh]=phase_scores(model,graph,e,r,genes,profiles,rows['test'])
   xx,pp=models[mh];ds=sorted({di for g,di in rows['test']});v=aggregate(xx,pp,profiles,ds,'pool',st['weight']);expected=a.base/'runs'/f"score-fold{c['fold']}"/'profile';z=np.load(expected/'scores.npz');assert z['genes'].astype(str).tolist()==genes and z['diseases'].astype(str).tolist()==ds;assert np.allclose(v,z['scores'],rtol=1e-5,atol=1e-6)
   prior=read(expected/'metrics.json');got={m:metrics(ranks(vv,ds,genes,rows['test'])) for m,vv in [('raw',v),('calibrated',cal(v))]}
   for m in got:
    for k,val in got[m].items():assert np.isclose(val,prior[m][k],rtol=1e-12,atol=1e-12),(m,k,val,prior[m][k])
   write(out/'historical-parity.json',{'metrics':got,'all_metrics_match':True,'score_max_abs_error':float(np.abs(v-z['scores']).max())})
  del model,graph;torch.cuda.empty_cache()
 write(out/'COMPLETE.json',{'config':c,'selection_sha256':c['selection_sha256'],'records':records});print('SCORE_COMPLETE',c['name'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.base/'code-v1'));sys.path.insert(0,str(a.base/'source/ULTRA'));logging.basicConfig(level=logging.INFO);torch.set_num_threads(8);c=read(a.config);assert c['protocol_sha256']==sha(a.root/'PROTOCOL.md');({'cache':cache,'link':train,'profile':train,'pilot':train,'score':score}[c['phase']])(a,c)
