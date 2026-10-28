"""Validation-only staged selection for the archived matched ULTRA campaign."""
import argparse,fcntl,hashlib,json,math,subprocess,time
from pathlib import Path
from .resume import require_complete_validation
SEED=0
GRAPHS=['original','augmented'];READOUTS=['pool','and'];TEMPS=[.01,.1,1.,10.]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def key(m):return m['hits@10'],m['hits@1'],m['mrr'],-m['mr']
def avg(ms):return {k:sum(m[k] for m in ms)/len(ms) for k in ['hits@1','hits@10','mrr','mr']}
def frozen(p,x):
 if p.exists():assert read(p)==x
 else:write(p,x)
def cmd(a):return subprocess.check_output(a,text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def cfg(root,co,phase,graph='original',fold=0,rd=None,tau=1.):
 name=f'{phase}-{graph}-fold{fold}'
 if phase=='profile':name=f'profile-{graph}-{rd}-t{tau}-fold{fold}'
 if phase=='score':name=f'score-fold{fold}'
 x={'seed':SEED,'cohort':co,'phase':phase,'graph':graph,'fold':fold,'name':name,'protocol_sha256':sha(root/'PROTOCOL.md')}
 if phase in ['link','profile','pilot']:x.update(steps=2 if phase=='pilot' else 1000 if phase=='link' else 500,eval_every=2 if phase=='pilot' else 250 if phase=='link' else 100,loss_temperature=tau)
 if rd:x['readout']=rd
 return x
def done(root,c):return root/c['cohort']/c['name']/'COMPLETE.json'
def verify(root,c):
 x=read(done(root,c));assert x['config']==c
 if c['phase']!='score':assert x['source_sha256']==sha(root/'source.sha256') and x['protocol_sha256']==sha(root/'PROTOCOL.md')
 if c['phase'] in ['link','profile','pilot']:
  parent=done(root,c).parent
  require_complete_validation(read(parent/'progress.json'),c['steps'],x['best'],c['eval_every'])
  for tag,record in x['best'].items():
   if sha(parent/(tag+'.pt'))!=record['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
 return x
def stageA(root,co,caches):
 settings={}
 for g in GRAPHS:
  settings[g]={};cs=[verify(root,c) for c in caches if c['graph']==g]
  for rd in READOUTS:
   choices=[]
   for i in range(11):
    p=cs[0]['grids'][rd][i]['parameter']
    for mode in ['raw','calibrated']:
     m=avg([v['grids'][rd][i]['metrics'][mode] for v in cs]);choices.append({'parameter':p,'scoring':mode,'metrics':m})
   def order(x):
    p=x['parameter'];distance=abs(p-.5) if rd=='pool' else abs(math.log(p))
    return key(x['metrics'])+(x['scoring']=='raw',-distance,-p)
   settings[g][rd]=max(choices,key=order)
 frozen(root/co/'stageA.json',{'settings':settings,'source':'development validation folds0,1,2','completions':{c['name']:sha(done(root,c)) for c in caches},'protocol_sha256':sha(root/'PROTOCOL.md')})
def stageB(root,co,links):
 for c in links:verify(root,c)
 settings={}
 for g in GRAPHS:
  settings[g]={}
  for rd in READOUTS:
   choices=[]
   for mode in ['raw','calibrated']:
    m=avg([verify(root,c)['best'][rd+'_'+mode]['metrics'] for c in links if c['graph']==g and c['fold']<3]);choices.append({'scoring':mode,'metrics':m})
   settings[g][rd]=max(choices,key=lambda x:key(x['metrics'])+(x['scoring']=='raw',))
 frozen(root/co/'stageB.json',{'settings':settings,'source':'development validation folds0,1,2','completions':{c['name']:sha(done(root,c)) for c in links},'stageA_sha256':sha(root/co/'stageA.json')})
def stageC(root,co,dev):
 settings={}
 for g in GRAPHS:
  settings[g]={}
  for rd in READOUTS:
   choices=[]
   for tau in TEMPS:
    cs=[verify(root,c) for c in dev if c['graph']==g and c['readout']==rd and c['loss_temperature']==tau]
    for mode in ['raw','calibrated']:choices.append({'loss_temperature':tau,'scoring':mode,'metrics':avg([x['best'][rd+'_'+mode]['metrics'] for x in cs])})
   settings[g][rd]=max(choices,key=lambda x:key(x['metrics'])+(x['scoring']=='raw',-abs(math.log(x['loss_temperature'])),-x['loss_temperature']))
 frozen(root/co/'stageC.json',{'settings':settings,'source':'development validation folds0,1,2','completions':{c['name']:sha(done(root,c)) for c in dev},'stageB_sha256':sha(root/co/'stageB.json')})
def finalfreeze(root,co,dev,prod):
 s=read(root/co/'stageC.json')['settings'];runs={}
 for g in GRAPHS:
  for rd in READOUTS:
   choice=s[g][rd]
   for f in range(10):
    c=cfg(root,co,'profile',g,f,rd,choice['loss_temperature']);x=verify(root,c);best=x['best'][rd+'_'+choice['scoring']];runs[g+'_'+rd+f'-fold{f}']={'directory':c['name'],'completion_sha256':sha(done(root,c)),'checkpoint_sha256':best['checkpoint_sha256'],'scoring':choice['scoring'],'best':best}
 frozen(root/co/'FINAL-SELECTION.json',{'runs':runs,'stageA_sha256':sha(root/co/'stageA.json'),'stageB_sha256':sha(root/co/'stageB.json'),'stageC_sha256':sha(root/co/'stageC.json'),'protocol_sha256':sha(root/'PROTOCOL.md')})
def tasks(root,co):
 caches=[cfg(root,co,'cache',g,f) for f in range(3) for g in GRAPHS];alltasks=caches[:];phase='aggregation-validation'
 if not all(done(root,c).exists() for c in caches):return alltasks,phase
 stageA(root,co,caches);links=[cfg(root,co,'link',g,f) for f in range(10) for g in GRAPHS];alltasks+=links;phase='link-adaptation'
 if not all(done(root,c).exists() for c in links):return alltasks,phase
 stageB(root,co,links);dev=[cfg(root,co,'profile',g,f,rd,t) for f in range(3) for g in GRAPHS for rd in READOUTS for t in TEMPS];alltasks+=dev;phase='profile-temperature-selection'
 if not all(done(root,c).exists() for c in dev):return alltasks,phase
 stageC(root,co,dev);s=read(root/co/'stageC.json')['settings'];prod=[cfg(root,co,'profile',g,f,rd,s[g][rd]['loss_temperature']) for f in range(3,10) for g in GRAPHS for rd in READOUTS];alltasks+=prod;phase='remaining-fold-adaptation'
 if not all(done(root,c).exists() for c in prod):return alltasks,phase
 finalfreeze(root,co,dev,prod);scores=[cfg(root,co,'score',fold=f) for f in range(10)]
 for c in scores:c['selection_sha256']=sha(root/co/'FINAL-SELECTION.json')
 alltasks+=scores;phase='test-scoring'
 if all(done(root,c).exists() for c in scores):phase='complete'
 return alltasks,phase
