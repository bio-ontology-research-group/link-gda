import copy
import torch
from torch import nn
from torch.nn import functional as F
class QueryBackbone(nn.Module):
 """Upstream QueryNBFNet one-hop projection with a logits interface for stable AND."""
 def __init__(self,config):
  super().__init__();from ultra.models import Ultra
  config=copy.deepcopy(config);config['entity_model']['class']='QueryNBFNet'
  self.ultra=Ultra(rel_model_cfg=config['relation_model'],entity_model_cfg=config['entity_model'])
 def forward(self,graph,batch):
  h=batch[:,:,0];r=batch[:,:,2];tails=batch[:,:,1]
  assert (h==h[:,:1]).all() and (r==r[:,:1]).all()
  rels=self.ultra.relation_model(graph.relation_graph,query=r[:,0]);q=rels[torch.arange(len(batch),device=batch.device),r[:,0]]
  start=F.one_hot(h[:,0],graph.num_nodes).to(q.dtype);features=start[:,:,None]*q[:,None,:]
  logits=self.ultra.entity_model(graph,features,rels,q)
  return logits.gather(1,tails)
def readout(x,mode,pool):
 if mode=='pool':return pool.sigmoid()*x.max(0).values+(1-pool.sigmoid())*x.mean(0)
 return F.logsigmoid(x).mean(0)
def scores(model,graph,p,genes,rel):
 genes=torch.as_tensor(genes,device=graph.edge_index.device);batch=torch.stack([torch.full_like(genes,p),genes,torch.full_like(genes,rel)],-1)[None]
 return model(graph,batch).reshape(-1)
def backward(model,graph,ps,positive,candidates,rel,pool,mode,rng):
 import numpy as np
 pos=np.array(sorted(set(positive)),dtype=np.int64);cand=np.asarray(candidates);eligible=cand[~np.isin(cand,pos)];random=rng.choice(eligible,32,replace=False);rest=eligible[~np.isin(eligible,random)];proposal=rng.choice(rest,min(256,len(rest)),replace=False);allg=np.concatenate([pos,random,proposal])
 model.eval()
 with torch.no_grad():
  first=torch.stack([scores(model,graph,p,allg,rel) for p in ps]);v=readout(first,mode,pool);order=np.lexsort((proposal,-v[len(pos)+32:].cpu().numpy()))[:32]
 chosen=np.concatenate([np.arange(len(pos)+32),len(pos)+32+order]);genes=allg[chosen];small=first[:,chosen].detach().requires_grad_(True);v=readout(small,mode,pool);loss=torch.logsumexp(v,0)-v[:len(pos)].mean()
 if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
 loss.backward();grad=small.grad.detach()
 for i,p in enumerate(ps):scores(model,graph,p,genes,rel).backward(grad[i])
 return float(loss.detach())
