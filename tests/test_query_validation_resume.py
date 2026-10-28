"""Exercise the actual training coordinator at interrupted validation boundaries."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phenoquery.common import cal, metrics, ranks, read, sha, write
from phenoquery.resume import pending_validation, publish_selection, require_complete_validation, verify_boundary_state
import phenoquery.objectives as objectives

PATH=Path(__file__).resolve().parents[1]/'code/baselines/run_phenoquery.py'
spec=importlib.util.spec_from_file_location('query_training_for_test',PATH)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_validation_schedule_detects_pending_and_lost_evaluations():
    assert pending_validation([{'step':s} for s in [0,100,200,300]],400,500)
    assert pending_validation([{'step':s} for s in [0,100,200,300,400]],500,500)
    assert not pending_validation([{'step':0}],2,4)
    with pytest.raises(ValueError,match='missing or repeated'):pending_validation([{'step':s} for s in [0,100,200,300,500]],500,500)
    with pytest.raises(ValueError,match='final scheduled'):require_complete_validation({'step':400,'curve':[{'step':s} for s in [0,100,200,300,400]],'best':{}},500,{})


def setup_training(monkeypatch,tmp_path,name,arm):
    root=tmp_path/name;root.mkdir()
    (root/'PROTOCOL.md').write_text('Synthetic CPU resumption fixture\n');(root/'source.sha256').write_text('fixture\n')
    cfg={'phase':'train','arm':arm,'fold':0,'pilot':False,'steps':2}
    write(root/'config.json',cfg)
    args=SimpleNamespace(root=root,config=root/'config.json',device='cpu',seed=0,initialization='ultra_4g',stop_after=None,stop_before_validation=None,ultra_root=root)
    def setup(*unused):
        model=torch.nn.Linear(1,1,bias=False);model.weight.data.zero_()
        profiles={'train':['p'],'v0':['p'],'v1':['p']}
        rows={'training':[('g0','train')],'validation':[('g0','v0'),('g1','v1')],'test':[]}
        return root,{}, {},object(),{'p':0,'g0':1,'g1':2,'g2':3},{'causes_phenotype':0},['g0','g1','g2'],profiles,rows,model,{},'fixture-checkpoint'
    def backward(model,graph,phenos,positive,candidates,rel,pool,mode,rng):
        # Genuine optimizer updates and both random streams; validation consumes neither.
        factor=float(rng.uniform(.9,1.1))+float(torch.rand(()))
        loss=-model.weight.sum()*factor
        if mode=='pool':loss=loss+.2*pool
        loss.backward();return float(loss.detach())
    def score_rows(model,graph,entities,rel,genes,profiles,rows,mode,weight):
        v=float(model.weight.detach()[0,0]);values=np.array([[v,0,0],[0,v,0]],dtype=float);ds=['v0','v1']
        both={m:metrics(ranks(x,ds,genes,rows)) for m,x in [('raw',values),('calibrated',cal(values))]}
        return values,ds,both
    monkeypatch.setattr(runner,'setup',setup);monkeypatch.setattr(runner,'backward',backward)
    monkeypatch.setattr(runner,'score_rows',score_rows);monkeypatch.setattr(objectives,'prepare_mask',lambda graph,*a,**k:graph)
    return args


def compare_runs(left,right):
    a=torch.load(left/'resume.pt',weights_only=False);b=torch.load(right/'resume.pt',weights_only=False)
    for key in ['step','best','baseline','curve','rng','pool']:assert a[key]==b[key]
    for name,record in a['selection_artifacts'].items():
        assert record['file']==b['selection_artifacts'][name]['file']
        assert sha(left/record['file'])==record['sha256']
        assert sha(right/record['file'])==b['selection_artifacts'][name]['sha256']
    torch.testing.assert_close(a['torch_rng'],b['torch_rng'],rtol=0,atol=0)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    for key,state in a['optimizer']['state'].items():
        for name,value in state.items():torch.testing.assert_close(value,b['optimizer']['state'][key][name],rtol=0,atol=0)
    for file in ['validation.npz','initial-validation.npz']:
        with np.load(left/file) as x,np.load(right/file) as y:
            for k in x.files:np.testing.assert_array_equal(x[k],y[k])
    assert read(left/'COMPLETE.json')['best']==read(right/'COMPLETE.json')['best']


@pytest.mark.parametrize('arm',['typed_and','bridge_pool'])
@pytest.mark.parametrize('boundary',['before-validation','after-selection-before-commit'])
def test_actual_training_resumes_final_validation_and_selection(monkeypatch,tmp_path,boundary,arm):
    a=setup_training(monkeypatch,tmp_path,'uninterrupted',arm);runner.train(a)
    b=setup_training(monkeypatch,tmp_path,'interrupted',arm);out=b.root/'training'/f'{arm}-fold0'
    if boundary=='before-validation':
        b.stop_before_validation=2;runner.train(b)
        before=torch.load(out/'resume.pt',weights_only=False)
        runner.train(b)  # Requeued gate remains stopped at the same boundary.
        b.stop_before_validation=None
        assert not (out/'COMPLETE.json').exists()
        assert read(out/'progress.json')['step']==0
    else:
        save=runner.save
        def fail_commit(path,state):
            if Path(path).name=='resume.pt' and state['step']==2 and state['curve'][-1]['step']==2:
                raise RuntimeError('Injected interruption after selected files and progress were written')
            save(path,state)
        # Force a pre-validation checkpoint at update 2, as production does at100..500.
        b.stop_before_validation=2;runner.train(b);b.stop_before_validation=None
        monkeypatch.setattr(runner,'save',fail_commit)
        with pytest.raises(RuntimeError,match='Injected interruption'):runner.train(b)
        monkeypatch.setattr(runner,'save',save)
        assert torch.load(out/'best.pt',weights_only=False)['step']==2
        assert torch.load(out/'resume.pt',weights_only=False)['best']['step']==0
    runner.train(b)
    if boundary=='before-validation':
        after=torch.load(out/'resume.pt',weights_only=False)
        assert verify_boundary_state(before,after)['restored_state_bitwise_equal']
        after['optimizer']['state'][0]['exp_avg'].add_(1e-7)
        with pytest.raises(ValueError,match='Boundary restoration changed optimizer'):verify_boundary_state(before,after)
    compare_runs(a.root/'training'/f'{arm}-fold0',out)
    assert read(out/'progress.json')['step']==2
    state=torch.load(out/'resume.pt',weights_only=False)
    # Corrupting the committed generation must fail rather than silently reuse aliases.
    generation=out/state['selection_artifacts']['best.pt']['file'];generation.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):publish_selection(out,state['selection_artifacts'])
