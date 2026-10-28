import argparse,copy,json,logging,math,sys,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from phenoquery.common import *
from phenoquery.model import QueryBackbone,backward
from phenoquery.matched_support import load_initial

def restore_cuda_rng(state):
 if state is not None:torch.cuda.set_rng_state(state)

def save(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');torch.save(x,tmp);tmp.replace(p)
def setup(a,arm,fold):
 root=a.root;d=root/'data'/f'fold{fold}';ident=read(d/'identity.json');variant,mode=arm.split('_');gdir=d/variant;gid=read(gdir/'identity.json')
 assert sha(root/'projection/manifest.json')==gid['projection_manifest_sha256']
 for n,h in ident['files'].items():assert sha(d/n)==h
 assert sha(gdir/'graph.pt')==gid['graph_sha256'] and sha(gdir/'vocabulary.json')==gid['vocabulary_sha256']
 graph=torch.load(gdir/'graph.pt',map_location='cpu',weights_only=False).to(a.device);voc=read(gdir/'vocabulary.json');entities=voc['entities'];relations=voc['relations'];genes=(d/'candidates.txt').read_text().splitlines();profiles=read(d/'profiles.json');rows={s:pairs(d/(s+'.tsv')) for s in ['training','validation','test']}
 assert len(genes)==a.expected_candidates
 held={di for g,di in rows['validation']+rows['test']};assert not held&set(entities)
 if variant!='bridge':assert not {di for g,di in rows['training']}&set(entities)
 from phenoquery.ultra_io import load_model_config
 cfg=load_model_config(str(a.ultra_root));model=QueryBackbone(cfg).to(a.device).eval();ck=a.ultra_root/'ckpts'/(a.initialization+'.pth');model.ultra.load_state_dict(load_initial(ck, 'query' if a.initialization == 'ultraquery' else 'general'));return d,ident,gid,graph,entities,relations,genes,profiles,rows,model,cfg,sha(ck)

def score_rows(model,graph,entities,rel,genes,profiles,rows,mode,weight):
 from phenoquery.ultra_io import score_phenotypes
 ds=sorted({d for g,d in rows});ps=sorted({p for d in ds for p in profiles[d]});x=score_phenotypes(model,graph,[entities[p] for p in ps],[entities[g] for g in genes],rel,2,str(graph.edge_index.device));values=aggregate(x,ps,profiles,ds,mode,weight);both={k:metrics(ranks(v,ds,genes,rows)) for k,v in [('raw',values),('calibrated',cal(values))]};return values,ds,both

def train(a):
 cfg=read(a.config);arm=cfg['arm'];fold=cfg['fold'];pilot=cfg.get('pilot',False);mode=arm.split('_')[1];out=a.root/('pilots' if pilot else 'training')/(arm+f'-fold{fold}');out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():
  previous=read(out/'identity.json')
  if previous['config']!=cfg or previous.get('initialization','ultra_4g')!=a.initialization or previous.get('seed',0)!=a.seed:
   raise ValueError('Completed run belongs to another configuration; choose a new output root')
  completion=read(out/'COMPLETE.json')
  if completion['source_manifest_sha256']!=sha(a.root/'source.sha256'):raise ValueError('Completed run source mismatch')
  if previous['initial_checkpoint_sha256']!=sha(a.ultra_root/'ckpts'/(a.initialization+'.pth')):raise ValueError('Initial checkpoint changed')
  if sha(out/'best.pt')!=completion['checkpoint_sha256']:raise ValueError('Completed checkpoint checksum mismatch')
  return
 torch.manual_seed(a.seed);np.random.seed(a.seed);rng=np.random.default_rng(a.seed);d,ident,gid,graph,entities,relations,genes,profiles,rows,model,modelcfg,initialsha=setup(a,arm,fold)
 pool=torch.nn.Parameter(torch.tensor(0.,device=a.device),requires_grad=mode=='pool');params=list(model.parameters())+([pool] if mode=='pool' else []);optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0.)
 outmeta={'config':cfg,'protocol_sha256':sha(a.root/'PROTOCOL.md'),'source_manifest_sha256':sha(a.root/'source.sha256'),'identity':ident,'graph_identity':gid,'initial_checkpoint_sha256':initialsha,'initialization':a.initialization,'seed':a.seed}
 if (out/'identity.json').exists() and read(out/'identity.json')!=outmeta:raise ValueError('Run provenance changed; use a new output root')
 write(out/'identity.json',outmeta)
 forward=relations['causes_phenotype'];rel=forward+len(relations);candidateids=[entities[g] for g in genes];trainmap={};sources={}
 for g,di in rows['training']:trainmap.setdefault(di,set()).add(entities[g])
 for di in trainmap:
  for p in set(profiles[di]):
   if di in entities:sources.setdefault(p,set()).add(entities[di])
 tds=sorted(trainmap);validation=rows['validation']
 if pilot:
  keep=set(sorted({di for g,di in validation})[:4]);validation=[(g,di) for g,di in validation if di in keep]
  from ultra.models import Ultra
  reference=Ultra(rel_model_cfg=modelcfg['relation_model'],entity_model_cfg=modelcfg['entity_model']).to(a.device).eval();reference.load_state_dict(model.ultra.state_dict());p=entities[profiles[validation[0][1]][0]];tails=torch.tensor(candidateids[:64],device=a.device);batch=torch.stack([torch.full_like(tails,p),tails,torch.full_like(tails,rel)],-1)[None]
  with torch.no_grad():one=model(graph,batch);two=reference(graph,batch);err=float((one-two).abs().max())
  assert torch.allclose(one,two,rtol=1e-5,atol=1e-6),err
  write(out/'parity.json',{'max_abs_error':err,'candidate_count':len(genes),'parameters':sum(p.numel() for p in model.parameters())});del reference;torch.cuda.empty_cache()
 curve=[];best=None;baseline=None;last=0;started=time.time();elapsed=0
 def checkpoint(step):
  return {'model':model.state_dict(),'pool':float(pool.detach()),'step':step,'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if str(a.device).startswith('cuda') else None,'curve':curve,'best':best,'baseline':baseline,'elapsed':elapsed+time.time()-started}
 def validate(step):
  nonlocal best,baseline
  model.eval();values,ds,both=score_rows(model,graph,entities,rel,genes,profiles,validation,mode,float(pool.sigmoid()));m=both['calibrated'];baseline=m if baseline is None else baseline;chosen=best is None or (eligible(m,baseline) and key(m)>key(best['metrics']))
  if chosen:
   best={'step':step,'metrics':m,'both_modes':both};save(out/'best.pt',{'model':model.state_dict(),'pool':float(pool.detach()),'step':step});np.savez_compressed(out/'validation.npz',scores=values,diseases=np.array(ds),genes=np.array(genes))
  if step==0:save(out/'initial.pt',{'model':model.state_dict(),'pool':float(pool.detach()),'step':0})
  curve.append({'step':step,'both_modes':both,'eligible':eligible(m,baseline),'selected':chosen});write(out/'progress.json',{'step':step,'best':best,'baseline':baseline,'curve':curve,'elapsed_seconds':elapsed+time.time()-started});save(out/'resume.pt',checkpoint(step));print('VALIDATION',arm,fold,step,m,'selected',chosen,flush=True)
 if (out/'resume.pt').exists():
  state=torch.load(out/'resume.pt',map_location='cpu');model.load_state_dict(state['model']);pool.data.fill_(state['pool']);optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng']);restore_cuda_rng(state['cuda_rng']);curve=state['curve'];best=state['best'];baseline=state['baseline'];last=state['step'];elapsed=state['elapsed']
 else:validate(0)
 initialweights=[p.detach().clone() for p in params] if pilot else None
 from phenoquery.objectives import prepare_mask
 for step in range(last+1,cfg['steps']+1):
  di=tds[int(rng.integers(len(tds)))];ps=sorted(set(profiles[di]));ps=sorted(rng.choice(ps,min(8,len(ps)),replace=False).tolist());m=prepare_mask(graph,[entities[p] for p in ps],forward,disease_id=sorted(set().union(*(sources.get(p,set()) for p in ps))))
  optimizer.zero_grad(set_to_none=True);tic=time.time();loss=backward(model,m,[entities[p] for p in ps],trainmap[di],candidateids,rel,pool,mode,rng);norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();del m
  with (out/'training.jsonl').open('a') as f:f.write(json.dumps({'step':step,'disease':di,'phenotypes':len(ps),'loss':loss,'gradient_norm':float(norm),'seconds':time.time()-tic})+'\n')
  if step%10==0 or pilot:
   save(out/'resume.pt',checkpoint(step));write(out/'heartbeat.json',{'step':step,'steps':cfg['steps'],'updated_unix':time.time(),'loss':loss,'gradient_norm':float(norm)});print('TRAIN',arm,fold,step,'loss',loss,'seconds',time.time()-tic,flush=True)
  if step%100==0 or step==cfg['steps']:validate(step)
 if pilot:
  delta=sum(float((p.detach()-old).abs().sum()) for p,old in zip(params,initialweights));assert delta>0;write(out/'update-check.json',{'parameter_absolute_change':delta,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
 write(out/'COMPLETE.json',{'arm':arm,'fold':fold,'pilot':pilot,'steps':cfg['steps'],'best':best,'baseline':baseline,'seconds':elapsed+time.time()-started,'identity_sha256':sha(out/'identity.json'),'checkpoint_sha256':sha(out/'best.pt'),'initial_checkpoint_sha256':sha(out/'initial.pt'),'validation_sha256':sha(out/'validation.npz'),'protocol_sha256':sha(a.root/'PROTOCOL.md'),'source_manifest_sha256':sha(a.root/'source.sha256'),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30});print('COMPLETE',arm,fold,flush=True)

def score(a):
 cfg=read(a.config);fold=cfg['fold'];freeze=read(a.root/'SELECTION.json');assert sha(a.root/'SELECTION.json')==cfg['selection_sha256']
 for arm in cfg.get('arms',ARMS):
  out=a.root/'test'/f'fold{fold}'/arm;out.mkdir(parents=True,exist_ok=True)
  if (out/'COMPLETE.json').exists():
   completed=read(out/'COMPLETE.json')
   if completed['selection_sha256']!=cfg['selection_sha256']:raise ValueError('Test selection changed')
   for filename,digest in completed['files'].items():
    if sha(out/filename)!=digest:raise ValueError('Completed test score checksum mismatch')
   continue
  parent=a.root/'training'/(arm+f'-fold{fold}');frozen=freeze['runs'][arm+f'-fold{fold}'];assert sha(parent/'COMPLETE.json')==frozen['completion_sha256'] and sha(parent/'best.pt')==frozen['checkpoint_sha256']
  d,ident,gid,graph,entities,relations,genes,profiles,rows,model,_,_=setup(a,arm,fold);state=torch.load(parent/'best.pt',map_location='cpu');model.load_state_dict(state['model']);mode=arm.split('_')[1];weight=1/(1+math.exp(-state['pool']));values,ds,both=score_rows(model,graph,entities,relations['causes_phenotype']+len(relations),genes,profiles,rows['test'],mode,weight);seen={g for g,d in rows['training']};rr={k:ranks(v,ds,genes,rows['test']) for k,v in [('raw',values),('calibrated',cal(values))]}
  reference=np.load(parent/'validation.npz',allow_pickle=False)
  np.savez_compressed(out/'scores.npz',scores=values,diseases=np.array(ds),genes=np.array(genes),reference_scores=reference['scores'],reference_diseases=reference['diseases']);reference.close();np.savez_compressed(out/'ranks.npz',**rr,genes=np.array([g for g,d in rows['test']]),diseases=np.array([d for g,d in rows['test']]),seen=np.array([g in seen for g,d in rows['test']]))
  write(out/'COMPLETE.json',{'arm':arm,'fold':fold,'metrics':both,'selection_sha256':cfg['selection_sha256'],'files':{n:sha(out/n) for n in ['scores.npz','ranks.npz']}});print('SCORED',arm,fold,both,flush=True);del model,graph;torch.cuda.empty_cache()
 write(a.root/'test'/f'fold{fold}'/'COMPLETE.json',{'fold':fold,'arms':cfg.get('arms',ARMS),'selection_sha256':cfg['selection_sha256']})
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,required=True,help='Campaign directory with data/configuration and output artifacts')
 p.add_argument('--base',type=Path,required=True,help='Benchmark workspace containing data/foldN')
 p.add_argument('--config',type=Path,required=True,help='Frozen JSON run configuration')
 p.add_argument('--ultra-root',type=Path,required=True,help='Pinned upstream ULTRA checkout containing ckpts/')
 p.add_argument('--device',default='cuda')
 p.add_argument('--expected-candidates',type=int,default=13625)
 p.add_argument('--historical-parity',action='store_true',help='Also verify against archived predecessor runs')
 a=p.parse_args()
 sys.path.insert(0,str(a.ultra_root.resolve()))
 logging.basicConfig(level=logging.INFO)
 torch.set_num_threads(8)
 cfg=read(a.config)
 a.seed=int(cfg.get('seed',0))
 a.initialization=cfg.get('initialization','ultra_4g')
 if a.initialization not in ['ultra_4g','ultraquery']:p.error('Unknown checkpoint initialization')
 if cfg['protocol_sha256']!=sha(a.root/'PROTOCOL.md'):raise ValueError('Protocol checksum mismatch')
 ({'train':train,'score':score}[cfg['phase']])(a)

if __name__=='__main__':
 main()
