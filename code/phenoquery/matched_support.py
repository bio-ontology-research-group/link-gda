import math
import numpy as np
import torch
from torch.nn import functional as F
from .common import sha,read,write,pairs,ranks,metrics,key
GRAPHS=['original','augmented'];READOUTS=['pool','and'];TEMPS=[.01,.1,1.,10.]
INPUT_TEMPS=[.0625,.125,.25,.5,1,2,4,8,16,32,64]
def cal(x):
 x=np.asarray(x,dtype=np.float64);x=x-x[:1];n=len(x)-1
 if n<1:raise ValueError('Need multiple queries')
 mu=(x.sum(0)-x)/n;var=(np.square(x).sum(0)-np.square(x))/n-mu**2
 return (x-mu)/(np.sqrt(np.maximum(var,0))+1e-12)
def aggregate(matrix,ps,profiles,ds,readout,parameter):
 ix={p:i for i,p in enumerate(ps)};out=[]
 for d in ds:
  block=matrix[[ix[p] for p in profiles[d]]]
  if readout=='pool':v=parameter*block.max(0)+(1-parameter)*block.mean(0)
  else:v=-np.logaddexp(0,-block.astype('float64')/parameter).mean(0)
  out.append(v)
 return np.stack(out)
def evaluate(matrix,ps,profiles,genes,rows,readout,parameter):
 ds=sorted({d for g,d in rows});values=aggregate(matrix,ps,profiles,ds,readout,parameter)
 return {mode:metrics(ranks(v,ds,genes,rows)) for mode,v in [('raw',values),('calibrated',cal(values))]}
def load_initial(path,cohort):
 state=torch.load(path,map_location='cpu')['model']
 if cohort=='query':
  prefix='model.model.'
  if not all(k.startswith(prefix) for k in state):raise ValueError('Unexpected query checkpoint keys')
  state={k[len(prefix):]:v for k,v in state.items()}
 return state
def save(p,x):
 from pathlib import Path
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');torch.save(x,t);t.replace(p)
def modelhash(model):
 import hashlib
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def and_backward(model,graph,ps,pos,genes,pool,rng,input_temperature,loss_temperature):
 from .objectives import _batch_scores,_eligible,deterministic_gradient_mode
 relation=graph.mask_metadata['inverse_relation'];pos,eligible=_eligible(pos,genes);random=rng.choice(eligible,32,replace=False);rest=eligible[~np.isin(eligible,random)];proposal=rng.choice(rest,min(256,len(rest)),replace=False);allg=np.concatenate([pos,random,proposal])
 with deterministic_gradient_mode(model):
  with torch.no_grad():
   first=torch.stack([_batch_scores(model,graph,p,allg,relation) for p in ps]);v=F.logsigmoid(first/input_temperature).mean(0);ix=np.lexsort((proposal,-v[len(pos)+32:].cpu().numpy()))[:32]
  pick=np.concatenate([np.arange(len(pos)+32),len(pos)+32+ix]);gg=allg[pick];small=first[:,pick].detach().requires_grad_(True);v=F.logsigmoid(small/input_temperature).mean(0)/loss_temperature;loss=torch.logsumexp(v,0)-v[:len(pos)].mean();loss.backward();grad=small.grad.detach()
  for i,p in enumerate(ps):_batch_scores(model,graph,p,gg,relation).backward(grad[i])
 return {'loss':float(loss.detach())}
