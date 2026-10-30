#!/usr/bin/env python3
"""Matched KGE graph comparison with train, global freeze, and test-score phases."""
import argparse
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phenoquery.common import sha, read, write

MODELS=('U1','T1','INDIGENA','CONVKB','PPOOL')
GRAPHS=('original','nobridge')
PRIMARY={'U1':'calibrated_mr','T1':'raw_head','INDIGENA':'raw_mr','CONVKB':'calibrated_mr','PPOOL':'raw_head'}


def directory(root,fold,graph,model):
    return root/f'fold{fold}'/graph/model


def validate_completion(path):
    complete=read(path/'TRAINED.json')
    if sha(path/'identity.json')!=complete['identity_sha256']:
        raise ValueError('Training identity changed')
    for name,digest in complete['files'].items():
        if Path(name).name!=name or sha(path/name)!=digest:
            raise ValueError('Trained artifact changed: '+name)
    if set(complete['selected'])!={'raw_head','raw_mr','calibrated_head','calibrated_mr'}:
        raise ValueError('Incomplete validation selections')
    return complete


def freeze(root,folds):
    if not folds or len(set(folds))!=len(folds) or min(folds)<0 or max(folds)>9:
        raise ValueError('Choose distinct fold IDs in 0..9')
    result={'schema':1,'folds':folds,'models':MODELS,'graphs':GRAPHS,'training':{}}
    campaign_settings={}
    for fold in folds:
        for graph in GRAPHS:
            for model in MODELS:
                out=directory(root,fold,graph,model)
                validate_completion(out)
                if (out/'SCORED.json').exists() or list(out.glob('scores-*.npz')):
                    raise ValueError('Test scores exist before global freeze')
                result['training'][str(out.relative_to(root))]=sha(out/'TRAINED.json')
        for model in MODELS:
            left=read(directory(root,fold,'original',model)/'identity.json')
            right=read(directory(root,fold,'nobridge',model)/'identity.json')
            for key in ['inputs','source','settings','torch','pykeen','gene_phenotypes_sha256','protocol_sha256']:
                if left.get(key)!=right.get(key):raise ValueError('Graph pair is not matched: '+key)
            common={key:left[key] for key in ['settings','source','torch','pykeen','protocol_sha256']}
            if model in campaign_settings and common!=campaign_settings[model]:
                raise ValueError('Model settings/source/environment differ across folds')
            campaign_settings[model]=common
    path=root/'FROZEN.json'
    if path.exists():
        if read(path)!=json.loads(json.dumps(result)):raise ValueError('Frozen campaign changed')
    else:write(path,result)
    print('FROZEN',len(result['training']),sha(path),flush=True)


def execute(args):
    import numpy as np
    import pandas as pd
    import torch
    from importlib.metadata import version
    from pykeen.models import TransD,ConvKB
    from pykeen.nn.init import PretrainedInitializer
    from pykeen.triples import TriplesFactory
    from pykeen.training import SLCWATrainingLoop
    from link_gda.checkpoint_identity import graph_identity
    from link_gda.filtered_negative_sampling import GenePoolNegativeSampler
    from link_gda.head_experiment import validate_splits,profile_metrics
    from link_gda.head_selection import FixedBudgetHeadCallback
    from link_gda.profile_ranking import ProfileRanker
    from link_gda.training_resume import train_resumable,train_profile_resumable
    from training.retrain_nobridge_kge import FastGraphScorer,FastIndigenaScorer

    out=directory(args.root,args.fold,args.graph,args.model);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data=args.data_root/f'fold{args.fold}'
    names=['raw/train.txt','candidates.txt','profiles.json','training.tsv','validation.tsv','test.tsv']
    inputs={name:sha(data/name) for name in names}
    if (data/'preparation.json').exists():
        prepared=read(data/'preparation.json')
        for name,key in zip(names,['graph','candidates','profiles','training','validation','test']):
            if inputs[name]!=prepared[key+'_sha256']:raise ValueError('Prepared input changed: '+name)
    genes=(data/'candidates.txt').read_text().splitlines();profiles=read(data/'profiles.json')
    if len(genes)!=args.expected_candidates:raise ValueError('Candidate count differs')
    train,val,test=[pd.read_csv(data/(split+'.tsv'),sep='\t') for split in ['training','validation','test']]
    validate_splits(train,val,test,genes)
    raw=pd.read_csv(data/'raw/train.txt',sep='\t',header=None,dtype=str,names=['h','r','t'])
    kept=raw if args.graph=='original' else raw[~raw.r.isin(['associated_with','has_symptom'])]
    factory=TriplesFactory.from_labeled_triples(kept.values,create_inverse_triples=True)
    if not set(genes)<=set(factory.entity_to_id):raise ValueError('Candidate missing from graph')
    heldout=set(val.Disease)|set(test.Disease)
    if heldout&set(factory.entity_to_id):raise ValueError('Held-out disease node in training graph')
    if args.graph=='nobridge' and set(train.Disease)&set(factory.entity_to_id):
        raise ValueError('Training disease node survived bridge removal')
    dimension=args.dimension or (400 if args.model=='INDIGENA' else 200)
    if args.epochs%args.validation_frequency:raise ValueError('Final epoch must be a validation epoch')
    settings={'model':args.model,'dimension':dimension,'seed':args.seed,'epochs':args.epochs,
              'validation_frequency':args.validation_frequency,'batch_size':args.batch_size,
              'sub_batch_size':args.sub_batch_size,'profile_epochs':args.profile_epochs,
              'checkpoint_frequency':args.checkpoint_frequency,'expected_candidates':args.expected_candidates,
              'device':args.device,'primary':PRIMARY[args.model],'early_stopping':False,
              't1_sampling':'candidate-typed, known training positives excluded'}
    source=Path(__file__).resolve().parents[1]
    identity={'schema':1,'fold':args.fold,'graph':args.graph,'settings':settings,'inputs':inputs,
              'graph_sha256':graph_identity(factory),'nodes':factory.num_entities,'forward_triples':len(kept),
              'torch':str(torch.__version__),'pykeen':version('pykeen'),
              'source':{str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py'))}}
    identity['protocol_sha256']=sha(source.parent/'reproducibility/protocols/matched-kge-20261029.md')
    if args.model=='INDIGENA':
        if args.gene_phenotypes is None:raise ValueError('INDIGENA needs --gene-phenotypes')
        identity['gene_phenotypes_sha256']=sha(args.gene_phenotypes)
    warm=None
    if args.model in ['CONVKB','PPOOL']:
        parent=directory(args.root,args.fold,args.graph,'U1');completion=validate_completion(parent)
        parent_id=read(parent/'identity.json')
        if any(parent_id[k]!=identity[k] for k in ['inputs','source','graph_sha256','torch','pykeen','protocol_sha256']):
            raise ValueError('Warm-start graph, inputs, source or environment differs')
        if parent_id['settings']['dimension']!=dimension or parent_id['settings']['seed']!=args.seed:
            raise ValueError('Warm-start dimension or seed differs')
        choice='calibrated_mr' if args.model=='CONVKB' else 'raw_head'
        warm=parent/completion['selected'][choice]['checkpoint']
        identity['warm_start_sha256']=sha(warm)
    if (out/'identity.json').exists():
        if read(out/'identity.json')!=identity:raise ValueError('Run identity changed')
    else:write(out/'identity.json',identity)
    trained=out/'TRAINED.json'
    if args.phase=='train' and trained.exists():
        validate_completion(out);print('VERIFIED_TRAINED',str(out),flush=True);return
    if args.phase=='score':
        frozen=read(args.root/'FROZEN.json');key=str(out.relative_to(args.root))
        if frozen['training'].get(key)!=sha(trained):raise ValueError('Selection absent from global freeze')
        completion=validate_completion(out)
        if (out/'SCORED.json').exists():
            scored=read(out/'SCORED.json')
            if scored['freeze_sha256']!=sha(args.root/'FROZEN.json'):raise ValueError('Freeze changed')
            for name,digest in scored['files'].items():
                if sha(out/name)!=digest:raise ValueError('Completed scores changed')
            return
    base=TransD(triples_factory=factory,embedding_dim=dimension,relation_dim=dimension,
                random_seed=args.seed).to(args.device)
    if warm is not None:base.load_state_dict(torch.load(warm,map_location=args.device,weights_only=True))
    model=base
    if args.model=='CONVKB':
        model=ConvKB(triples_factory=factory,embedding_dim=dimension,random_seed=args.seed,
            hidden_dropout_rate=0.,num_filters=100,
            entity_initializer=PretrainedInitializer(tensor=base.entity_representations[0](indices=None).detach()),
            relation_initializer=PretrainedInitializer(tensor=base.relation_representations[0](indices=None).detach())).to(args.device)
    scorer=(FastIndigenaScorer(factory,genes,profiles,args.gene_phenotypes,None) if args.model=='INDIGENA'
            else FastGraphScorer(factory,genes,profiles,eval_chunk_size=32768 if args.model=='CONVKB' else 200000))
    ranker=None
    if args.model=='PPOOL':
        ranker=ProfileRanker(model,factory,train,profiles,genes,validation_pairs=val,seed=args.seed,
            learn_pooling=True,temperature=1.,random_negatives=32,hard_negatives=32,proposal_size=256)
    selection_model=ranker or model
    val_diseases=sorted(set(val.Disease))
    def scores(diseases):
        return scorer.score_diseases(model,diseases,**({'pooling_weight':ranker.pooling_weight()} if ranker else {}))
    if args.phase=='train':
        callback=FixedBudgetHeadCallback(selection_model,lambda:profile_metrics(scores(val_diseases),val,genes),
            out/'selected',out/'validation.json',frequency=1 if ranker else args.validation_frequency,resume=True)
        optimizer=torch.optim.Adam(selection_model.parameters(),lr=1e-4 if args.model in ['CONVKB','PPOOL'] else .001)
        if ranker:
            state=train_profile_resumable(ranker,optimizer,factory,directory=out/'resume',identity=identity,
                callback=callback,total_epochs=args.profile_epochs,accumulation_steps=16,stop_after=args.stop_after)
        else:
            options={'negative_sampler_kwargs':{'num_negs_per_pos':1}}
            if args.model=='T1':
                options.update(negative_sampler=GenePoolNegativeSampler,
                    negative_sampler_kwargs={'num_negs_per_pos':1,
                      'target_relations':[2*factory.relation_to_id['causes_phenotype']],
                      'gene_pool':[factory.entity_to_id[g] for g in genes]})
            loop=SLCWATrainingLoop(model=model,triples_factory=factory,optimizer=optimizer,
                automatic_memory_optimization=False,**options)
            state=train_resumable(loop,factory,directory=out/'resume',identity=identity,callbacks=[callback],
                total_epochs=args.epochs,batch_size=args.batch_size,sub_batch_size=args.sub_batch_size,
                checkpoint_frequency=args.checkpoint_frequency,stop_after=args.stop_after)
        if not state['complete']:return
        selected={key:{**record,'checkpoint':Path(record['checkpoint']).name} for key,record in callback.best.items()}
        files={p.name:sha(p) for p in callback.checkpoint_files()}
        write(trained,{'identity_sha256':sha(out/'identity.json'),'selected':selected,'files':files,
                       'training':state,'primary':PRIMARY[args.model]})
        validate_completion(out)
        print('TRAINED',str(out),state['epochs'],flush=True)
    else:
        ds=sorted(set(test.Disease));tds=sorted(set(train.Disease));files={}
        for key,record in completion['selected'].items():
            selection_model.load_state_dict(torch.load(out/record['checkpoint'],map_location=args.device,weights_only=True))
            test_map=scores(ds);train_map=scores(tds)
            path=out/('scores-'+key+'.npz');temporary=path.with_suffix('.npz.partial')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,scores=np.stack([test_map[d] for d in ds]),
                    train_scores=np.stack([train_map[d] for d in tds]),diseases=np.asarray(ds),
                    train_diseases=np.asarray(tds),genes=np.asarray(genes))
            temporary.replace(path);files[path.name]=sha(path)
        write(out/'SCORED.json',{'freeze_sha256':sha(args.root/'FROZEN.json'),
              'training_sha256':sha(trained),'files':files,'primary':PRIMARY[args.model]})
        print('SCORED',str(out),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['train','freeze','score'],required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--fold',type=int)
    p.add_argument('--folds',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--graph',choices=GRAPHS)
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--gene-phenotypes',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--dimension',type=int)
    p.add_argument('--epochs',type=int,default=1000)
    p.add_argument('--profile-epochs',type=int,default=20)
    p.add_argument('--validation-frequency',type=int,default=20)
    p.add_argument('--checkpoint-frequency',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=65536)
    p.add_argument('--sub-batch-size',type=int,default=8192)
    p.add_argument('--expected-candidates',type=int,default=13625)
    p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    if any(value<1 for value in [a.epochs,a.profile_epochs,a.validation_frequency,
        a.checkpoint_frequency,a.batch_size,a.sub_batch_size,a.expected_candidates]):
        p.error('Training and candidate-count settings must be positive')
    if a.dimension is not None and a.dimension<1:p.error('Dimension must be positive')
    if a.seed<0:p.error('Seed must be nonnegative')
    if a.phase=='freeze':freeze(a.root,a.folds)
    else:
        if a.fold is None or a.data_root is None or a.graph is None or a.model is None:
            p.error('Training/scoring requires data-root, fold, graph and model')
        if not 0<=a.fold<=9:p.error('Fold must be in 0..9')
        execute(a)


if __name__=='__main__':main()
