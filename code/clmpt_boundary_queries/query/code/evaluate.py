"""Full-cohort saved evaluation isolated from the continuing training state."""
import copy,hashlib,json,os,pathlib,random,time
import numpy as np
import torch
from metrics import measure,calibrate
from state_compare import compare

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8388608),b''):h.update(block)
    return h.hexdigest()

def evaluate(model,sampler,bank,output,step,identity_sha256,checkpoint_sha256,should_stop=None):
    output=pathlib.Path(output);output.mkdir(exist_ok=True);prefix=output/('fit-step-%06d'%step);archive=prefix.with_suffix('.npz');metadata=prefix.with_suffix('.json')
    if metadata.exists():
        previous=json.loads(metadata.read_text());assert previous['identity_sha256']==identity_sha256 and previous['step']==step and previous['checkpoint_sha256']==checkpoint_sha256
        assert sha(archive)==previous['scores_sha256'];return previous
    for orphan in (archive,archive.with_suffix('.npz.tmp'),metadata.with_suffix('.json.tmp')):
        if orphan.exists():
            recovery=output/'orphaned';recovery.mkdir(exist_ok=True);os.replace(orphan,recovery/(str(time.time_ns())+'-'+orphan.name))
    before={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().clone(),'cuda':[x.clone() for x in torch.cuda.get_rng_state_all()],'sampler':copy.deepcopy(sampler.state())}
    was_training=model.training;versions=[p._version for p in model.parameters()];begun=time.time();model.eval()
    try:
        validation=[]
        for row in bank['validation']:
            hp=row['hp_ids'];validation.append({'type':'profile','hp_ids':hp,'constants':{'s'+str(i):h for i,h in enumerate(hp)},'atoms':[['s'+str(i),bank['relations']['inverse_has_symptom'],'e1'] for i in range(len(hp))]+[['e1',bank['relations']['inverse_associated_with'],'f']]})
        assert len(validation)==545 and sum(len(x['gene_ids']) for x in bank['validation'])==586
        empty={'type':'empty','hp_ids':[],'constants':{},'atoms':[['e1',bank['relations']['inverse_associated_with'],'f']]};candidate=torch.tensor(bank['candidate_ids'],device='cuda')
        with torch.no_grad():
            representations=[];scores=[]
            for start in range(0,len(validation),32):
                if should_stop is not None and should_stop():return None
                chunk=validation[start:start+32]
                representations.append(model.encode(chunk).cpu().numpy());scores.append(model.scores(chunk,candidate).cpu().numpy())
            if should_stop is not None and should_stop():return None
            representation=np.concatenate(representations);real=np.concatenate(scores);empty_rep=model.encode([empty]);empty_score=model.scores([empty],candidate).cpu().numpy()
        assert np.isfinite(real).all() and np.isfinite(empty_score).all()
        arrays={'validation_real':real,'validation_shuffled':real[bank['validation_shuffle']],'validation_empty':np.repeat(empty_score,545,axis=0),'validation_representation':representation,'empty_representation':empty_rep.cpu().numpy(),'candidate_ids':np.array(bank['candidate_ids'])}
        answers=[row['gene_ids'] for row in bank['validation']];seen={g for row in bank['training_profiles'] for g in row['answers']};metrics={}
        for control,key in [('real','validation_real'),('shuffled','validation_shuffled'),('empty','validation_empty')]:
            if should_stop is not None and should_stop():return None
            metrics[control]={'raw':measure(arrays[key],answers,bank['candidate_ids'],seen_genes=seen)}
            if control!='empty':metrics[control]['calibrated']=measure(calibrate(arrays[key]),answers,bank['candidate_ids'],seen_genes=seen)
        after={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'sampler':sampler.state()};invariant=compare(before,after);assert invariant['exact'] and versions==[p._version for p in model.parameters()]
        if should_stop is not None and should_stop():return None
        temp=archive.with_suffix('.npz.tmp')
        with temp.open('xb') as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,archive)
        result={'step':step,'identity_sha256':identity_sha256,'checkpoint_sha256':checkpoint_sha256,'validation':metrics,'raw_mrr':metrics['real']['raw']['answer_micro']['mrr'],'scores_sha256':sha(archive),'RNG_sampler_unchanged':invariant,'parameter_versions_unchanged':True,'elapsed_seconds':time.time()-begun,'scope':'Same full545/586 unfiltered average-rank metrics; Raw and calibrated PhenoQuery-style checkpoint selectors are kept separate; no early stopping in the fixed-budget phase. OwnLOO calibration/control/seen-strata descriptive; no TEST. Saved representations permit cosine replay, not independent Transformer reimplementation.'}
        temp=metadata.with_suffix('.json.tmp')
        with temp.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temp,metadata);return result
    finally:
        random.setstate(before['python']);np.random.set_state(before['numpy']);torch.set_rng_state(before['torch']);torch.cuda.set_rng_state_all(before['cuda']);model.train(was_training)
