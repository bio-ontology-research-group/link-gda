import argparse,copy,json,logging,math,sys,time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from phenoquery.common import *
from phenoquery.model import QueryBackbone,backward
from phenoquery.matched_support import load_initial
from phenoquery.campaign import verify_source,verify_global,verify_gates
from phenoquery.resume import publish_selection,pending_validation,require_complete_validation

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
  require_complete_validation(read(out/'progress.json'),cfg['steps'],completion['best'])
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
 curve=[];best=None;baseline=None;last=0;started=time.time();elapsed=0;selected_artifacts={}
 def checkpoint(step):
  return {'model':model.state_dict(),'pool':float(pool.detach()),'step':step,'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if str(a.device).startswith('cuda') else None,'curve':curve,'best':best,'baseline':baseline,'selection_artifacts':selected_artifacts,'elapsed':elapsed+time.time()-started}
 def validate(step):
  nonlocal best,baseline,selected_artifacts
  model.eval();values,ds,both=score_rows(model,graph,entities,rel,genes,profiles,validation,mode,float(pool.sigmoid().detach()));m=both['calibrated'];baseline=m if baseline is None else baseline;chosen=best is None or (eligible(m,baseline) and key(m)>key(best['metrics']))
  if chosen:
   best={'step':step,'metrics':m,'both_modes':both}
   model_file=f'selected-step{step:04d}.pt';scores_file=f'validation-step{step:04d}.npz'
   save(out/model_file,{'model':model.state_dict(),'pool':float(pool.detach()),'step':step})
   np.savez_compressed(out/scores_file,scores=values,diseases=np.array(ds),genes=np.array(genes))
   selected_artifacts={name:{'file':file,'sha256':sha(out/file)} for name,file in [('best.pt',model_file),('validation.npz',scores_file)]}
   publish_selection(out,selected_artifacts)
  if step==0:
   save(out/'initial.pt',{'model':model.state_dict(),'pool':float(pool.detach()),'step':0})
   np.savez_compressed(out/'initial-validation.npz',scores=values,diseases=np.array(ds),genes=np.array(genes))
  curve.append({'step':step,'both_modes':both,'eligible':eligible(m,baseline),'selected':chosen});write(out/'progress.json',{'step':step,'best':best,'baseline':baseline,'curve':curve,'elapsed_seconds':elapsed+time.time()-started});save(out/'resume.pt',checkpoint(step));print('VALIDATION',arm,fold,step,m,'selected',chosen,flush=True)
 if (out/'resume.pt').exists():
  state=torch.load(out/'resume.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);pool.data.fill_(state['pool']);optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng'];torch.set_rng_state(state['torch_rng']);restore_cuda_rng(state['cuda_rng']);curve=state['curve'];best=state['best'];baseline=state['baseline'];last=state['step'];elapsed=state['elapsed']
  selected_artifacts=state['selection_artifacts'];publish_selection(out,selected_artifacts)
  write(out/'progress.json',{'step':curve[-1]['step'],'best':best,'baseline':baseline,'curve':curve,'elapsed_seconds':elapsed})
  log=out/'training.jsonl'
  if log.exists():log.write_text(''.join(line+'\n' for line in log.read_text().splitlines() if json.loads(line)['step']<=last))
  if pending_validation(curve,last,cfg['steps']):validate(last)
 else:validate(0)
 initialweights=None
 if pilot:
  origin=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
  initialweights=[origin['model'][name].to(p) for name,p in model.named_parameters()]
  if mode=='pool':initialweights.append(torch.tensor(origin['pool']).to(pool))
 from phenoquery.objectives import prepare_mask
 for step in range(last+1,cfg['steps']+1):
  di=tds[int(rng.integers(len(tds)))];ps=sorted(set(profiles[di]));ps=sorted(rng.choice(ps,min(8,len(ps)),replace=False).tolist());m=prepare_mask(graph,[entities[p] for p in ps],forward,disease_id=sorted(set().union(*(sources.get(p,set()) for p in ps))))
  optimizer.zero_grad(set_to_none=True);tic=time.time();loss=backward(model,m,[entities[p] for p in ps],trainmap[di],candidateids,rel,pool,mode,rng);norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();del m
  with (out/'training.jsonl').open('a') as f:f.write(json.dumps({'step':step,'disease':di,'phenotypes':len(ps),'loss':loss,'gradient_norm':float(norm),'seconds':time.time()-tic})+'\n')
  if step%10==0 or pilot:
   save(out/'resume.pt',checkpoint(step));write(out/'heartbeat.json',{'step':step,'steps':cfg['steps'],'updated_unix':time.time(),'loss':loss,'gradient_norm':float(norm)});print('TRAIN',arm,fold,step,'loss',loss,'seconds',time.time()-tic,flush=True)
  if step%100==0 or step==cfg['steps']:
   if getattr(a,'stop_before_validation',None)==step:
    save(out/'resume.pt',checkpoint(step));write(out/'STOPPED.json',{'step':step,'reason':'technical interruption before validation'});return
   validate(step)
  if a.stop_after is not None and step>=a.stop_after and step<cfg['steps']:
   save(out/'resume.pt',checkpoint(step))
   write(out/'STOPPED.json',{'step':step,'reason':'technical resumption check'})
   return
 if pilot:
  delta=sum(float((p.detach()-old).abs().sum()) for p,old in zip(params,initialweights));assert delta>0;write(out/'update-check.json',{'parameter_absolute_change':delta,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
 require_complete_validation(read(out/'progress.json'),cfg['steps'],best)
 publish_selection(out,selected_artifacts)
 write(out/'COMPLETE.json',{'arm':arm,'fold':fold,'pilot':pilot,'steps':cfg['steps'],'best':best,'baseline':baseline,'seconds':elapsed+time.time()-started,'identity_sha256':sha(out/'identity.json'),'checkpoint_sha256':sha(out/'best.pt'),'initial_checkpoint_sha256':sha(out/'initial.pt'),'initial_validation_sha256':sha(out/'initial-validation.npz'),'validation_sha256':sha(out/'validation.npz'),'protocol_sha256':sha(a.root/'PROTOCOL.md'),'source_manifest_sha256':sha(a.root/'source.sha256'),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30});print('COMPLETE',arm,fold,flush=True)

def score(a):
 cfg=read(a.config);fold=cfg['fold'];freeze=read(a.root/'SELECTION.json')
 if sha(a.root/'SELECTION.json')!=cfg['selection_sha256']:raise ValueError('Selection changed')
 records={}
 for arm in cfg.get('arms',ARMS):
  parent=a.root/'training'/(arm+f'-fold{fold}')
  frozen=freeze['runs'][arm+f'-fold{fold}']
  if sha(parent/'COMPLETE.json')!=frozen['completion_sha256']:raise ValueError('Training completion changed')
  completion=read(parent/'COMPLETE.json')
  if read(parent/'identity.json')['config'].get('global_selection_required',False):
   if 'global_selection_sha256' not in cfg:raise ValueError('All controlled selections must be frozen before test scoring')
   verify_global(a.root,cfg['global_selection_sha256'])
  checkpoints=[('selected','best.pt','validation.npz','checkpoint_sha256','validation_sha256')]
  if cfg.get('score_initial',False):
   checkpoints.append(('initial','initial.pt','initial-validation.npz','initial_checkpoint_sha256','initial_validation_sha256'))
  for label,checkpoint,reference_file,checkpoint_key,reference_key in checkpoints:
   if sha(parent/checkpoint)!=completion[checkpoint_key] or sha(parent/reference_file)!=completion[reference_key]:
    raise ValueError('Checkpoint or validation scores changed')
  d,ident,gid,graph,entities,relations,genes,profiles,rows,model,_,_=setup(a,arm,fold)
  cache=None
  for label,checkpoint,reference_file,checkpoint_key,reference_key in checkpoints:
   name=arm if label=='selected' else arm+'_frozen'
   out=a.root/'test'/f'fold{fold}'/name;out.mkdir(parents=True,exist_ok=True)
   if (out/'COMPLETE.json').exists():
    completed=read(out/'COMPLETE.json')
    if completed['selection_sha256']!=cfg['selection_sha256']:raise ValueError('Test selection changed')
    for filename,digest in completed['files'].items():
     if sha(out/filename)!=digest:raise ValueError('Completed test score checksum mismatch')
    records[name]=completed
    continue
   state=torch.load(parent/checkpoint,map_location='cpu',weights_only=False)
   model.load_state_dict(state['model']);mode=arm.split('_')[1]
   weight=1/(1+math.exp(-state['pool']))
   if label=='initial' and completion['best']['step']==0 and cache is not None:
    values,ds,both=cache
   else:
    values,ds,both=score_rows(model,graph,entities,relations['causes_phenotype']+len(relations),genes,profiles,rows['test'],mode,weight)
   cache=(values,ds,both)
   seen={g for g,d in rows['training']}
   rr={k:ranks(v,ds,genes,rows['test']) for k,v in [('raw',values),('calibrated',cal(values))]}
   with np.load(parent/reference_file,allow_pickle=False) as reference:
    if reference['genes'].astype(str).tolist()!=genes:raise ValueError('Validation candidate order changed')
    np.savez_compressed(out/'scores.npz',scores=values,diseases=np.array(ds),genes=np.array(genes),
                        reference_scores=reference['scores'],reference_diseases=reference['diseases'])
   np.savez_compressed(out/'ranks.npz',**rr,genes=np.array([g for g,d in rows['test']]),
                       diseases=np.array([d for g,d in rows['test']]),seen=np.array([g in seen for g,d in rows['test']]))
   completed={'arm':arm,'fold':fold,'checkpoint_kind':label,'checkpoint_sha256':sha(parent/checkpoint),
              'initialization':a.initialization,'seed':a.seed,'selected_step':state['step'],'metrics':both,
              'selection_sha256':cfg['selection_sha256'],'files':{n:sha(out/n) for n in ['scores.npz','ranks.npz']}}
   write(out/'COMPLETE.json',completed);records[name]=completed
   print('SCORED',name,fold,both,flush=True)
  del model,graph;torch.cuda.empty_cache()
 write(a.root/'test'/f'fold{fold}'/'COMPLETE.json',{'fold':fold,'arms':cfg.get('arms',ARMS),
       'selection_sha256':cfg['selection_sha256'],'records':records})

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,required=True,help='Campaign directory with data/configuration and output artifacts')
 p.add_argument('--base',type=Path,required=True,help='Benchmark workspace containing data/foldN')
 p.add_argument('--config',type=Path,required=True,help='Frozen JSON run configuration')
 p.add_argument('--ultra-root',type=Path,required=True,help='Pinned upstream ULTRA checkout containing ckpts/')
 p.add_argument('--device',default='cuda')
 p.add_argument('--stop-after',type=int,help='Technical gate: save state and stop before this configured run is complete')
 p.add_argument('--stop-before-validation',type=int,help='Technical gate: stop after an update but before its scheduled validation')
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
 verify_source(a.root,Path(__file__).resolve().parents[1])
 if cfg.get('global_selection_required',False):verify_gates(a.root.parent)
 ({'train':train,'score':score}[cfg['phase']])(a)

if __name__=='__main__':
 main()
