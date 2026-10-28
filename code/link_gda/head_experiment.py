"""Isolated top-hit experiments; training and test scoring are separate phases."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from pykeen.training import SLCWATrainingLoop
from .calibration import standardize
from .checkpoint_identity import graph_identity
from .evaluation import evaluate_by_graph
from .head_selection import FixedBudgetHeadCallback
from .negative_sampling import GenePoolNegativeSampler


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=lambda x:x.item())+'\n');os.replace(tmp,path)


def pair_hash(frame):
    return hashlib.sha256(frame[['Gene','Disease']].to_csv(sep='\t',index=False).encode()).hexdigest()


def validate_splits(training,validation,test,candidates):
    sets=[set(x.Disease) for x in [training,validation,test]]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Disease leakage between train/validation/test')
    if len(candidates)!=len(set(candidates)):raise ValueError('Duplicate candidates')
    for frame in [training,validation,test]:
        if not set(frame.Gene)<=set(candidates):raise ValueError('Positive gene outside candidate universe')


def profile_metrics(score_map,pairs,candidates):
    """Unique-query calibration; average-rank ties; association-weighted metrics."""
    diseases=sorted(set(pairs.Disease));raw=np.stack([score_map[d] for d in diseases])
    if raw.shape!=(len(diseases),len(candidates)) or not np.isfinite(raw).all():raise ValueError('Invalid profile scores')
    lookup={d:i for i,d in enumerate(diseases)};gene_index={g:i for i,g in enumerate(candidates)}
    ans={}
    for setting,scores in [('raw',raw),('calibrated',standardize(raw,diseases))]:
        ranks=[]
        for row in pairs.itertuples(index=False):
            values=scores[lookup[row.Disease]];truth=values[gene_index[row.Gene]]
            ranks.append(1+int((values>truth).sum())+0.5*(int((values==truth).sum())-1))
        ranks=np.asarray(ranks);metrics={'mr':float(ranks.mean()),'mrr':float((1/ranks).mean())}
        for k in [1,3,10,100]:metrics[f'hits@{k}']=float((ranks<=k).mean())
        ans[setting]=(metrics,metrics.copy())
    return ans


def save_scores(directory,score_map,pairs,candidates):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    diseases=sorted(score_map)
    np.savez_compressed(directory/'scores.npz',scores=np.stack([score_map[d] for d in diseases]),diseases=np.array(diseases),genes=np.array(candidates))
    pairs.to_csv(directory/'pairs.tsv',sep='\t',index=False)
    (directory/'candidates.txt').write_text('\n'.join(candidates)+'\n')
    write_json(directory/'metrics.json',profile_metrics(score_map,pairs,candidates))
    write_json(directory/'checksums.json',{p.name:digest(p) for p in sorted(directory.iterdir()) if p.is_file()})


def run_experiment(config_path,model,factory,training,validation,test,disease2pheno,candidates,*,fold,seed,batch_size,learning_rate):
    cfg=json.loads(Path(config_path).read_text());phase=cfg['phase'];condition=cfg.get('condition','bundle' if phase=='score_bundle' else None)
    if cfg['fold']!=fold or cfg.get('seed',0)!=seed:raise ValueError('Configuration identity differs from trainer flags')
    validate_splits(training,validation,test,candidates)
    entities=set(factory.entity_to_id)
    if (set(validation.Disease)|set(test.Disease))&entities:raise ValueError('Held-out disease entity occurs in graph')
    identity={'graph_sha256':graph_identity(factory),'candidate_sha256':hashlib.sha256('\n'.join(candidates).encode()).hexdigest(),
              'training_sha256':pair_hash(training),'validation_sha256':pair_hash(validation),'test_sha256':pair_hash(test),
              'fold':fold,'seed':seed,'candidates':len(candidates),'training_diseases':training.Disease.nunique(),
              'validation_diseases':validation.Disease.nunique(),'test_diseases':test.Disease.nunique()}
    output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',cfg);write_json(output/'identity.json',identity)
    training.to_csv(output/'training.tsv',sep='\t',index=False);validation.to_csv(output/'validation.tsv',sep='\t',index=False)
    source_root=Path(__file__).resolve().parents[1]
    write_json(output/'source-sha256.json',{str(p.relative_to(source_root)):digest(p) for p in sorted(source_root.rglob('*.py'))})
    started=time.time()
    if phase=='graph':
        if condition not in ['U1','U32','T1','T32']:raise ValueError(condition)
        negatives=int(condition[1:]);kwargs={'num_negs_per_pos':negatives};sampler=None
        if condition.startswith('T'):
            sampler=GenePoolNegativeSampler
            kwargs.update(target_relations=[2*factory.relation_to_id['causes_phenotype']],gene_pool=[factory.entity_to_id[g] for g in candidates])
        evaluate=lambda:evaluate_by_graph(model,validation,disease2pheno,candidates,triples_factory=factory,calibration_unit='disease',dual_metrics=True)
        callback=FixedBudgetHeadCallback(model,evaluate,str(output/'checkpoint'),str(output/'validation.jsonl'),frequency=cfg.get('validation_frequency',20))
        optimizer=torch.optim.Adam(model.get_grad_params(),lr=learning_rate)
        loop=SLCWATrainingLoop(model=model,triples_factory=factory,optimizer=optimizer,negative_sampler=sampler,negative_sampler_kwargs=kwargs)
        epochs=cfg.get('epochs',1000)
        loop.train(triples_factory=factory,num_epochs=epochs,batch_size=batch_size,
                   sub_batch_size=cfg.get('sub_batch_size'),callbacks=[callback])
        if epochs%cfg.get('validation_frequency',20):callback.evaluate_epoch(epochs)
        write_json(output/'completion.json',{'phase':phase,'condition':condition,'epochs':epochs,'wall_seconds':time.time()-started,'best':callback.best,'identity':identity})
        return
    if phase=='score_bundle':
        _score_bundle(cfg,output,model,factory,training,validation,test,disease2pheno,candidates,identity,seed,started)
        return
    if phase in ['profile','score']:
        from .profile_ranking import ProfileRanker
        parent=Path(cfg['parent']);parent_id=json.loads((parent/'identity.json').read_text())
        if parent_id!=identity:raise ValueError('Warm-start/scoring identity mismatch')
        if not (parent/'completion.json').is_file():raise ValueError('Parent run is incomplete')
        if phase=='profile':
            if condition not in ['P-random','P-hard','P-pool']:raise ValueError('Unknown profile condition '+condition)
            checkpoint=parent/'checkpoint_raw_head.pt'
            model.load_state_dict(torch.load(checkpoint,map_location=next(model.parameters()).device,weights_only=True))
            hard=condition!='P-random'
            profile_training=training
            subset_metadata={'technical_pilot':bool(cfg.get('technical_pilot',False)), 'subset_applied':False}
            if 'max_profile_diseases' in cfg:
                if cfg.get('technical_pilot') is not True:
                    raise ValueError('max_profile_diseases is allowed only with technical_pilot=true')
                limit=cfg['max_profile_diseases']
                if isinstance(limit,bool) or not isinstance(limit,int) or limit<1:
                    raise ValueError('max_profile_diseases must be a positive integer')
                all_diseases=sorted(set(training.Disease))
                selected=sorted(np.random.default_rng(seed).choice(all_diseases,min(limit,len(all_diseases)),replace=False).tolist())
                profile_training=training.loc[training.Disease.isin(selected)].copy()
                profile_training.to_csv(output/'technical-profile-training.tsv',sep='\t',index=False)
                subset_metadata.update(subset_applied=True, requested_max_diseases=limit,
                                       selected_diseases=selected, selected_pair_count=len(profile_training),
                                       full_training_sha256=pair_hash(training), subset_training_sha256=pair_hash(profile_training),
                                       interpretation='Technical execution smoke only; not a scientific experiment result')
            ranker=ProfileRanker(model,factory,profile_training,disease2pheno,candidates,validation_pairs=validation,seed=seed,
                                learn_pooling=condition=='P-pool',temperature=1.,random_negatives=32 if hard else 64,
                                hard_negatives=32 if hard else 0,proposal_size=256)
            optimizer=torch.optim.Adam(ranker.parameters(),lr=cfg.get('profile_lr',0.0001))
            evaluate=lambda:profile_metrics(ranker.score_profiles(sorted(set(validation.Disease))),validation,candidates)
            callback=FixedBudgetHeadCallback(ranker,evaluate,str(output/'checkpoint'),str(output/'validation.jsonl'),frequency=1)
            write_json(output/'warm-start.json',{'path':str(checkpoint),'sha256':digest(checkpoint)})
            write_json(output/'profile-metadata.json',dict(ranker.metadata(),training_subset=subset_metadata))
            for epoch in range(1,cfg.get('epochs',20)+1):
                stats=ranker.train_epoch(optimizer,accumulation_steps=cfg.get('accumulation_steps',16))
                with (output/'training.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch,'stats':stats,'pooling_weight':ranker.pooling_weight()})+'\n')
                callback.evaluate_epoch(epoch)
                print('PROFILE_EPOCH',epoch,stats,flush=True)
            write_json(output/'completion.json',{'phase':phase,'condition':condition,'epochs':cfg.get('epochs',20),'wall_seconds':time.time()-started,'best':callback.best,'identity':identity})
            return
        selection,record=_read_selection(cfg)
        artifact={'condition':condition,'checkpoint':cfg['checkpoint']}
        if artifact not in record['allowed_artifacts']:raise ValueError('Checkpoint not permitted by frozen selection')
        _score_artifact(artifact,parent,output,model,factory,training,validation,test,disease2pheno,candidates,
                        identity,seed,selection,phase='score')
        return
    raise ValueError('Unknown phase '+phase)


def _read_selection(cfg):
    selection=Path(cfg['selection_record'])
    record=json.loads(selection.read_text())
    if record.get('selection_data')!='inner-validation-only':
        raise ValueError('Missing validation-only selection freeze')
    allowed=record.get('allowed_artifacts')
    if not isinstance(allowed,list) or not allowed:
        raise ValueError('No artifacts permitted by selection record')
    keys=[]
    for item in allowed:
        if set(item)!={'condition','checkpoint'}:
            raise ValueError('Selection artifacts require condition and checkpoint only')
        if item['condition'] not in ['U1','U32','T1','T32','P-random','P-hard','P-pool']:
            raise ValueError('Unknown selected condition')
        if item['checkpoint'] not in ['raw_head','calibrated_head','raw_mr','calibrated_mr']:
            raise ValueError('Unknown selected checkpoint')
        keys.append((item['condition'],item['checkpoint']))
    if len(set(keys))!=len(keys):raise ValueError('Duplicate selected artifacts')
    return selection,record


def _check_parent(parent,artifact,identity):
    parent=Path(parent)
    if json.loads((parent/'identity.json').read_text())!=identity:
        raise ValueError('Warm-start/scoring identity mismatch')
    if not (parent/'completion.json').is_file():raise ValueError('Parent run is incomplete')
    parent_cfg=json.loads((parent/'config.json').read_text())
    if parent_cfg['condition']!=artifact['condition']:
        raise ValueError('Parent condition differs from frozen selection artifact')
    if parent_cfg['phase'] not in ['graph','profile']:
        raise ValueError('Scoring parent must be a completed training run')
    checkpoint=parent/('checkpoint_'+artifact['checkpoint']+'.pt')
    if not checkpoint.is_file():raise ValueError('Selected checkpoint is missing')
    return parent_cfg,checkpoint


def _score_artifact(artifact,parent,output,model,factory,training,validation,test,disease2pheno,candidates,
                    identity,seed,selection,phase='score_bundle'):
    from .profile_ranking import ProfileRanker
    started=time.time()
    parent_cfg,checkpoint=_check_parent(parent,artifact,identity)
    is_profile=parent_cfg['phase']=='profile'
    if is_profile:
        hard=parent_cfg['condition']!='P-random'
        ranker=ProfileRanker(model,factory,training,disease2pheno,candidates,validation_pairs=validation,seed=seed,
                            learn_pooling=parent_cfg['condition']=='P-pool',random_negatives=32 if hard else 64,
                            hard_negatives=32 if hard else 0,proposal_size=256)
        ranker.load_state_dict(torch.load(checkpoint,map_location=next(model.parameters()).device,weights_only=True))
        scores=ranker.score_profiles(sorted(set(test.Disease)))
    else:
        model.load_state_dict(torch.load(checkpoint,map_location=next(model.parameters()).device,weights_only=True))
        scores=evaluate_by_graph(model,test,disease2pheno,candidates,triples_factory=factory,return_score_map=True)
    save_scores(output/'test',scores,test,candidates)
    completion={'phase':phase,**artifact,'wall_seconds':time.time()-started,'identity':identity,
                'parent':str(parent),'checkpoint_sha256':digest(checkpoint),'selection_sha256':digest(selection),
                'parent_config_sha256':digest(Path(parent)/'config.json'),
                'test_checksums_sha256':digest(output/'test'/'checksums.json'),
                'technical_pilot':bool(parent_cfg.get('technical_pilot',False))}
    write_json(output/'completion.json',completion)
    return completion


def _score_bundle(cfg,output,model,factory,training,validation,test,disease2pheno,candidates,identity,seed,started):
    selection,record=_read_selection(cfg)
    requested=cfg.get('artifacts',record['allowed_artifacts'])
    artifacts=[{k:a[k] for k in ['condition','checkpoint']} for a in requested]
    if not artifacts or any(a not in record['allowed_artifacts'] for a in artifacts):
        raise ValueError('Bundle contains artifact not permitted by frozen selection')
    labels=[a['condition']+'_'+a['checkpoint'] for a in artifacts]
    if len(set(labels))!=len(labels):raise ValueError('Duplicate bundle artifacts')
    parents=[Path(a['parent']) if 'parent' in a else Path(cfg['parents'][a['condition']]) for a in requested]
    # Validate every parent before scoring any outer test query.
    for artifact,parent in zip(artifacts,parents):
        _check_parent(parent,artifact,identity)
    results=[]
    for artifact,label,parent in zip(artifacts,labels,parents):
        directory=output/label
        directory.mkdir(parents=False,exist_ok=False)
        write_json(directory/'identity.json',identity)
        result=_score_artifact(artifact,parent,directory,model,factory,
                               training,validation,test,disease2pheno,candidates,identity,seed,selection)
        results.append({'artifact':artifact,'directory':label,'completion_sha256':digest(directory/'completion.json'),
                        'checkpoint_sha256':result['checkpoint_sha256']})
        print('SCORED_ARTIFACT',label,flush=True)
    write_json(output/'completion.json',{'phase':'score_bundle','identity':identity,'artifacts':results,
                                       'selection_sha256':digest(selection),'wall_seconds':time.time()-started})
