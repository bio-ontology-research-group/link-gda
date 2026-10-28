import csv,json,hashlib
from pathlib import Path
import numpy as np
ARMS=['bridge_pool']+[g+'_'+m for g in ['plain','entity','typed','structure'] for m in ['pool','and']]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def pairs(p):
 with open(p) as f:return [(x['Gene'],x['Disease']) for x in csv.DictReader(f,delimiter='\t')]
def cal(x):
 x=np.asarray(x,dtype=np.float64);x=x-x[:1];n=len(x)-1
 if n<1:raise ValueError('Need multiple queries for calibration')
 mu=(x.sum(0)-x)/n;v=(x*x).sum(0)/n-x*x/n-mu*mu
 return (x-mu)/(np.sqrt(np.maximum(v,0))+1e-12)
def ranks(x,ds,genes,rows):
 if not np.isfinite(x).all():raise ValueError('Nonfinite scores')
 di={d:i for i,d in enumerate(ds)};gi={g:i for i,g in enumerate(genes)};rr=[]
 for g,d in rows:
  v=x[di[d]];s=v[gi[g]];rr.append(1+(v>s).sum()+.5*((v==s).sum()-1))
 return np.asarray(rr,float)
def metrics(r):return {'hits@1':float(np.mean(r<=1)),'hits@10':float(np.mean(r<=10)),'mrr':float(np.mean(1/r)),'mr':float(np.mean(r))}
def key(m):return m['hits@10'],m['hits@1'],m['mrr'],-m['mr']
def eligible(m,initial):return m['hits@1']>=initial['hits@1'] and m['hits@10']>=initial['hits@10']
def aggregate(x,ps,profiles,ds,mode,weight):
 ix={p:i for i,p in enumerate(ps)};result=[]
 for d in ds:
  block=x[[ix[p] for p in sorted(set(profiles[d]))]].astype('float64')
  result.append(weight*block.max(0)+(1-weight)*block.mean(0) if mode=='pool' else -np.logaddexp(0,-block).mean(0))
 return np.stack(result)
