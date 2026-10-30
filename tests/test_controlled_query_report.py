from pathlib import Path
import importlib.util
import shutil
import numpy as np
import pytest
from analysis import summarize_controlled_query as report
from phenoquery.common import cal, metrics, ranks, read, sha, write
from phenoquery.campaign import freeze_global


def test_profile_bootstrap_preserves_association_weights_and_pairs():
    values=np.array([[1,2],[0,3],[-1,4]],dtype=float)
    result=report.cluster_intervals(values,[('p1',),('p1',),('p2',)],repeats=37,seed=19,confidence=.8)
    draws=np.random.default_rng(19).integers(0,2,size=(37,2))
    members=[np.array([0,1]),np.array([2])]
    expected=np.stack([values[np.concatenate([members[j] for j in draw])].mean(0) for draw in draws])
    np.testing.assert_allclose(result['point'],values.mean(0),rtol=0,atol=0)
    np.testing.assert_allclose(result['lower'],np.quantile(expected,.1,axis=0))
    np.testing.assert_allclose(result['upper'],np.quantile(expected,.9,axis=0))
    assert result['profile_clusters']==2
    assert result['point'][0]!=np.mean([.5,-1.])


def make_campaign(tmp_path):
    root=tmp_path/'campaign';root.mkdir()
    protocol=Path(report.__file__).resolve().parents[2]/'reproducibility/protocols/controlled-query-20261028.md'
    names=[f'{i}-seed{s}' for i in report.INITS for s in report.SEEDS]
    rng=np.random.default_rng(12);genes=[f'g{i}' for i in range(17)]
    common=root/'data';common.mkdir()
    for fold in range(10):
        data=common/f'fold{fold}';data.mkdir()
        split_rows={'training':[('g0',f'train{fold}a'),('g1',f'train{fold}b')],
                    'validation':[('g0',f'val{fold}a'),('g3',f'val{fold}b')],
                    'test':[('g0',f'test{fold}a'),('g5',f'test{fold}a'),('g2',f'test{fold}b'),('g3',f'test{fold}c')]}
        (data/'candidates.txt').write_text('\n'.join(genes)+'\n')
        for split,rows in split_rows.items():
            (data/(split+'.tsv')).write_text('Gene\tDisease\n'+''.join(f'{g}\t{d}\n' for g,d in rows))
        profiles={d:[f'p{j%3}'] for j,(g,d) in enumerate(sum(split_rows.values(),[]))}
        write(data/'profiles.json',profiles)
    for init in report.INITS:
        for seed in report.SEEDS:
            child=root/f'{init}-seed{seed}';child.mkdir();(child/'data').symlink_to(common,target_is_directory=True)
            shutil.copyfile(protocol,child/'PROTOCOL.md');(child/'source.sha256').write_text('synthetic fixture source; no actual training\n')
            selection={'protocol_sha256':sha(protocol),'runs':{}}
            for fold in range(10):
                data=common/f'fold{fold}';ds=[f'test{fold}{s}' for s in 'abc'];val_ds=[f'val{fold}{s}' for s in 'ab']
                rows=[('g0',ds[0]),('g5',ds[0]),('g2',ds[1]),('g3',ds[2])];seen=np.array([True,False,False,False])
                data_hashes={n:sha(data/n) for n in ['training.tsv','validation.tsv','test.tsv','profiles.json','candidates.txt']}
                for arm in report.ARMS:
                    run=f'{arm}-fold{fold}';parent=child/'training'/run;parent.mkdir(parents=True)
                    config={'arm':arm,'fold':fold,'steps':500,'pilot':False,'seed':seed,'initialization':init,'global_selection_required':True}
                    identity={'config':config,'identity':{'files':data_hashes},'seed':seed,'initialization':init,'protocol_sha256':sha(protocol),'source_manifest_sha256':sha(child/'source.sha256')}
                    write(parent/'identity.json',identity)
                    for n in ['initial.pt','best.pt']:(parent/n).write_bytes((n+str(seed)).encode())
                    initial_ref=np.zeros((2,len(genes)));ref=initial_ref.copy()
                    if seed:ref[0,0]=1;ref[1,3]=1
                    for n,v in [('validation.npz',ref),('initial-validation.npz',initial_ref)]:np.savez_compressed(parent/n,scores=v,diseases=val_ds,genes=genes)
                    trained={'arm':arm,'fold':fold,'steps':500,'pilot':False,'identity_sha256':sha(parent/'identity.json'),'source_manifest_sha256':sha(child/'source.sha256'),'protocol_sha256':sha(protocol),'best':{'step':100 if seed else 0},'checkpoint_sha256':sha(parent/'best.pt'),'initial_checkpoint_sha256':sha(parent/'initial.pt'),'validation_sha256':sha(parent/'validation.npz'),'initial_validation_sha256':sha(parent/'initial-validation.npz')}
                    validation_rows=[('g0',val_ds[0]),('g3',val_ds[1])]
                    initial_metrics={m:metrics(ranks(x,val_ds,genes,validation_rows)) for m,x in [('raw',initial_ref),('calibrated',cal(initial_ref))]}
                    selected_metrics={m:metrics(ranks(x,val_ds,genes,validation_rows)) for m,x in [('raw',ref),('calibrated',cal(ref))]}
                    trained['baseline']=initial_metrics['calibrated']
                    trained['best'].update(metrics=selected_metrics['calibrated'],both_modes=selected_metrics)
                    curve=[{'step':step,'both_modes':initial_metrics if step==0 else selected_metrics,'eligible':True,'selected':step==0 or bool(seed and step==100)} for step in range(0,501,100)]
                    write(parent/'progress.json',{'step':500,'best':trained['best'],'baseline':trained['baseline'],'curve':curve})
                    write(parent/'COMPLETE.json',trained)
                    selection['runs'][run]={'completion_sha256':sha(parent/'COMPLETE.json'),'checkpoint_sha256':trained['checkpoint_sha256'],'best':trained['best']}
                    for kind in ['selected','initial']:
                        name=arm if kind=='selected' else arm+'_frozen';out=child/'test'/f'fold{fold}'/name;out.mkdir(parents=True)
                        if kind=='selected' or seed:
                            values=rng.integers(0,5,size=(3,len(genes))).astype(float)
                        rr={m:ranks(x,ds,genes,rows) for m,x in [('raw',values),('calibrated',cal(values))]}
                        np.savez_compressed(out/'scores.npz',scores=values,diseases=ds,genes=genes,reference_scores=ref if kind=='selected' else initial_ref,reference_diseases=val_ds)
                        np.savez_compressed(out/'ranks.npz',**rr,genes=[g for g,d in rows],diseases=[d for g,d in rows],seen=seen)
                        complete={'arm':arm,'fold':fold,'checkpoint_kind':kind,'initialization':init,'seed':seed,'checkpoint_sha256':trained['checkpoint_sha256' if kind=='selected' else 'initial_checkpoint_sha256'],'selected_step':trained['best']['step'] if kind=='selected' else 0,'metrics':{m:metrics(x) for m,x in rr.items()},'files':{n:sha(out/n) for n in ['scores.npz','ranks.npz']}}
                        write(out/'COMPLETE.json',complete)
            write(child/'SELECTION.json',selection)
            for p in (child/'test').glob('*/*/COMPLETE.json'):
                c=read(p);c['selection_sha256']=sha(child/'SELECTION.json');write(p,c)
    write(root/'CONTROLLED.json',{'campaigns':names,'arms':report.ARMS,'initializations':report.INITS,'seeds':report.SEEDS,'expected_runs':240,'runs_per_campaign':40,'protocol_sha256':sha(protocol),'source_sha256':sha(root/names[0]/'source.sha256')})
    freeze_global(root)
    return root


def test_complete_controlled_report_and_seed_metric_averaging(tmp_path):
    root=make_campaign(tmp_path);result=report.summarize(root,expected_candidates=17)
    assert result['score_bundles']==480 and result['training_runs']==240
    assert result['associations']==40 and result['profile_clusters']==3
    assert len(result['conditions'])==32 and len(result['primary_contrasts'])==8
    assert result['inference']['confidence_level']==.99375 and result['inference']['replicates']==10000
    contrast=result['primary_contrasts'][0];differences=[]
    for seed in range(3):
        sides=[]
        for arm in ['plain_pool','bridge_pool']:
            rr=[]
            for fold in range(10):
                with np.load(root/f'ultra_4g-seed{seed}'/'test'/f'fold{fold}'/arm/'ranks.npz') as z:rr.extend(z['calibrated'])
            sides.append(np.asarray(rr)<=10)
        differences.append((sides[0].astype(float)-sides[1]).mean()*100)
    np.testing.assert_allclose(contrast['per_seed_percentage_points'],differences)
    assert contrast['difference_percentage_points']==pytest.approx(np.mean(differences))
    assert result['selected_step_counts']=={0:80,100:160}
    generator_path=Path(report.__file__).resolve().parents[2]/'reproducibility/generate_controlled_query_tables.py'
    spec=importlib.util.spec_from_file_location('controlled_tables',generator_path)
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    destination=tmp_path/'tables';generator.generate(result,destination)
    assert {p.name for p in destination.iterdir()}=={'controlled-query-main.tex','controlled-query-full.tex','controlled-query-contrasts.tex'}
    main=(destination/'controlled-query-main.tex').read_text()
    assert main.count('Disease bridges, Pool &')==2
    selected=result['conditions']['ultra_4g/typed_and/selected/calibrated']
    assert generator.value(selected,'hits@10') in main
    assert '99.375' in (destination/'controlled-query-contrasts.tex').read_text()
    path=root/'ultra_4g-seed0/test/fold0/bridge_pool/scores.npz'
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='Checksum mismatch'):report.summarize(root,expected_candidates=17)


def test_global_freeze_is_required_before_reading_scores(tmp_path):
    root=make_campaign(tmp_path);(root/'GLOBAL-SELECTION.json').unlink()
    with pytest.raises(FileNotFoundError):report.summarize(root,expected_candidates=17)
    freeze_global(root)
    path=root/'ultraquery-seed2/SELECTION.json';selection=read(path);selection['runs'].pop('typed_and-fold9');write(path,selection)
    with pytest.raises(ValueError,match='selection changed'):report.summarize(root,expected_candidates=17)


@pytest.mark.parametrize('mutation',['candidate-order','retained-rank','seed','validation-choice'])
def test_report_rejects_internally_mislabeled_outputs(tmp_path,mutation):
    root=make_campaign(tmp_path)
    out=root/'ultra_4g-seed0/test/fold0/bridge_pool'
    complete=read(out/'COMPLETE.json')
    if mutation in ['candidate-order','retained-rank']:
        name='scores.npz' if mutation=='candidate-order' else 'ranks.npz'
        with np.load(out/name) as z:data={k:z[k].copy() for k in z.files}
        if mutation=='candidate-order':data['genes']=data['genes'][::-1]
        else:data['calibrated'][0]+=1
        np.savez_compressed(out/name,**data)
        complete['files'][name]=sha(out/name)
        write(out/'COMPLETE.json',complete)
        with pytest.raises(AssertionError):report.summarize(root,expected_candidates=17)
    elif mutation=='seed':
        complete['seed']=2;write(out/'COMPLETE.json',complete)
        with pytest.raises(ValueError,match='Score identity mismatch'):report.summarize(root,expected_candidates=17)
    else:
        path=root/'ultra_4g-seed0/training/bridge_pool-fold0/progress.json'
        progress=read(path);progress['curve'][2]['selected']=True;write(path,progress)
        with pytest.raises(ValueError,match='fixed selection rule'):report.summarize(root,expected_candidates=17)
