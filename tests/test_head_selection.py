import json
import pytest
import torch
from link_gda.head_selection import FixedBudgetHeadCallback, head_key


def metrics(h10=.2,h1=.1,mrr=.15,mr=100):
    return {'hits@10':h10,'hits@1':h1,'mrr':mrr,'mr':mr}


def test_lexicographic_direction_and_finite():
    base = metrics()
    for kwargs in [dict(h10=.3,h1=0),dict(h1=.2,mrr=0),dict(mrr=.2,mr=1000),dict(mr=90)]:
        assert head_key(metrics(**kwargs)) > head_key(base)
    with pytest.raises(ValueError): head_key(metrics(mr=float('nan')))


def test_same_trajectory_different_head_and_mr_checkpoints(tmp_path):
    model = torch.nn.Linear(1,1,bias=False)
    current = [metrics()]
    def evaluate():
        assert not model.training
        assert not torch.is_grad_enabled()
        return {arm:(current[0],{}) for arm in ('raw','calibrated')}
    callback = FixedBudgetHeadCallback(model,evaluate,tmp_path/'run',tmp_path/'curve.json',frequency=2)
    model.train()
    with torch.no_grad(): model.weight.fill_(1)
    callback.post_epoch(1,1.0)
    assert not callback.best
    callback.post_epoch(2,1.0)
    assert model.training
    current[0] = metrics(h10=.3,mr=200)
    with torch.no_grad(): model.weight.fill_(2)
    callback.post_epoch(4,.5)
    assert callback.best['raw_mr']['epoch'] == 2
    assert callback.best['raw_head']['epoch'] == 4
    assert torch.load(tmp_path/'run_raw_mr.pt',weights_only=True)['weight'].item() == 1
    assert torch.load(tmp_path/'run_raw_head.pt',weights_only=True)['weight'].item() == 2
    callback.evaluate_epoch(5)  # equal metric tie retains earlier epoch
    assert callback.best['raw_head']['epoch'] == 4
    record=json.loads((tmp_path/'curve.json').read_text())
    assert len(record['curve']) == 3
    assert set(record['best']) == {'raw_mr','raw_head','calibrated_mr','calibrated_head'}
    with pytest.raises(FileExistsError):
        FixedBudgetHeadCallback(model,evaluate,tmp_path/'run',tmp_path/'curve.json')


def test_mode_restored_on_evaluation_failure(tmp_path):
    model=torch.nn.Linear(1,1)
    def fail(): raise RuntimeError('expected')
    callback=FixedBudgetHeadCallback(model,fail,tmp_path/'run',tmp_path/'curve.json')
    model.train()
    with pytest.raises(RuntimeError): callback.evaluate_epoch(20)
    assert model.training
    model.eval()
    with pytest.raises(RuntimeError): callback.evaluate_epoch(40)
    assert not model.training
