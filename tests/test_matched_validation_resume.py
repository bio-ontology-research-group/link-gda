"""Actual matched-training coordinator survives validation transaction failures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phenoquery.common import read, write
from phenoquery.resume import pending_validation
from phenoquery.matched_selection import verify
import phenoquery.objectives as objectives

spec = importlib.util.spec_from_file_location('matched_training_test',
    Path(__file__).resolve().parents[1] / 'code/baselines/run_ultra_matched.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(monkeypatch, root, phase):
    root.mkdir()
    (root / 'PROTOCOL.md').write_text('Synthetic CPU fixture\n')
    (root / 'source.sha256').write_text('fixture\n')
    args = SimpleNamespace(root=root, device='cpu', seed=0)
    config = {'name': 'pilot', 'cohort': 'general', 'graph': 'original', 'fold': 0,
              'phase': 'pilot', 'training_phase': phase, 'steps': 20,
              'eval_every': 10, 'readout': 'pool', 'loss_temperature': 1.}
    def setup(*unused):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()
        graph = SimpleNamespace(edge_index=torch.tensor([[1, 2], [0, 0]]),
                                edge_type=torch.tensor([0, 0]))
        profiles = {'train': ['p'], 'v0': ['p'], 'v1': ['p']}
        rows = {'training': [('g0', 'train')],
                'validation': [('g0', 'v0'), ('g1', 'v1')]}
        return root, {}, graph, {'p': 0, 'g0': 1, 'g1': 2, 'train': 3}, \
               {'causes_phenotype': 0}, ['g0', 'g1'], profiles, rows, model, 'fixture'
    def backward(model, graph, *args, **kwargs):
        rng = args[-1]
        loss = -model.weight.sum() * (float(rng.uniform(.9, 1.1)) + torch.rand(()))
        if phase == 'profile':
            loss = loss + .2 * args[-2]
        loss.backward()
        return {'loss': float(loss.detach())}
    def phase_scores(model, *unused):
        return float(model.weight.detach()[0, 0]), ['p']
    def evaluate(value, *unused):
        # Distinct improving validations ensure all selection aliases are replaced.
        m = {'hits@1': value, 'hits@10': value, 'mrr': value, 'mr': 2-value}
        return {'raw': m, 'calibrated': m}
    monkeypatch.setattr(runner, 'setup', setup)
    monkeypatch.setattr(runner, 'phase_scores', phase_scores)
    monkeypatch.setattr(runner, 'evaluate', evaluate)
    monkeypatch.setattr(objectives, 'prepare_mask', lambda graph, *a, **k: graph)
    monkeypatch.setattr(objectives, 'stage2_backward', backward)
    monkeypatch.setattr(objectives, 'profile_backward', backward)
    return args, config, root / 'general/pilot'


def equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('phase', ['link', 'profile'])
@pytest.mark.parametrize('step,boundary', [(10, 'before'), (20, 'before'), (10, 'after')])
def test_matched_resume_preserves_all_selected_readouts(monkeypatch, tmp_path, phase, step, boundary):
    a, c, full = fixture(monkeypatch, tmp_path / 'full', phase)
    runner.train(a, c)
    a, c, resumed = fixture(monkeypatch, tmp_path / 'resumed', phase)
    save = runner.save
    def interrupt(path, state):
        if Path(path).name == 'resume.pt' and state['step'] == step:
            validated = state['curve'][-1]['step'] == step
            if boundary == 'before' and not validated:
                save(path, state)
                raise RuntimeError('Injected interruption before validation')
            if boundary == 'after' and validated:
                raise RuntimeError('Injected interruption after selection before commit')
        save(path, state)
    monkeypatch.setattr(runner, 'save', interrupt)
    with pytest.raises(RuntimeError, match='Injected interruption'):
        runner.train(a, c)
    monkeypatch.setattr(runner, 'save', save)
    interrupted = torch.load(resumed / 'resume.pt', weights_only=False)
    assert pending_validation(interrupted['curve'], step, c['steps'], c['eval_every'])
    for changed in [{**c, 'steps': 30}, {**c, 'loss_temperature': 2.}]:
        with pytest.raises(ValueError, match='Resume configuration, source or scientific inputs changed'):
            runner.train(a, changed)
    original = (a.root / 'PROTOCOL.md').read_text()
    (a.root / 'PROTOCOL.md').write_text('Changed protocol\n')
    with pytest.raises(ValueError, match='Resume configuration, source or scientific inputs changed'):
        runner.train(a, c)
    (a.root / 'PROTOCOL.md').write_text(original)
    runner.train(a, c)
    left = torch.load(full / 'resume.pt', weights_only=False)
    right = torch.load(resumed / 'resume.pt', weights_only=False)
    for k in ['model', 'optimizer', 'pool', 'step', 'rng', 'torch_rng', 'cuda_rng', 'curve', 'initial_model_hash']:
        equal(left[k], right[k])
    for tag in left['best']:
        for k in ['step', 'metrics', 'parameter']:
            equal(left['best'][tag][k], right['best'][tag][k])
        equal(torch.load(full / (tag + '.pt'), weights_only=False),
              torch.load(resumed / (tag + '.pt'), weights_only=False))
    assert read(resumed / 'COMPLETE.json')['nonzero_update']
    runner.train(a, c)  # Completed reuse checks every scheduled validation.
    verify(a.root, c)  # Stage selection applies the same validation requirements.
    progress = read(resumed / 'progress.json')
    progress['curve'] = [r for r in progress['curve'] if r['step'] != 10]
    write(resumed / 'progress.json', progress)
    with pytest.raises(ValueError, match='missing or repeated'):
        runner.train(a, c)
    with pytest.raises(ValueError, match='missing or repeated'):
        verify(a.root, c)


def test_profile_resume_rejects_changed_upstream_checkpoint(monkeypatch, tmp_path):
    a, c, out = fixture(monkeypatch, tmp_path / 'profile', 'profile')
    c['phase'] = 'profile'
    cohort = a.root / 'general'
    parent = cohort / 'link-original-fold0'
    parent.mkdir(parents=True)
    write(cohort / 'stageA.json', {'settings': {'original': {'pool': {'parameter': .5, 'scoring': 'calibrated'}}}})
    write(cohort / 'stageB.json', {'settings': {'original': {'pool': {'scoring': 'calibrated'}}}})
    path = parent / 'pool_calibrated.pt'
    torch.save({'model': {'weight': torch.zeros((1, 1))}}, path)
    write(parent / 'COMPLETE.json', {'best': {'pool_calibrated': {'checkpoint_sha256': runner.sha(path)}}})
    save = runner.save
    def interrupt(path, state):
        save(path, state)
        if Path(path).name == 'resume.pt' and state['step'] == 10:
            raise RuntimeError('Injected interruption')
    monkeypatch.setattr(runner, 'save', interrupt)
    with pytest.raises(RuntimeError, match='Injected interruption'):
        runner.train(a, c)
    monkeypatch.setattr(runner, 'save', save)
    metadata_before = (out / 'parent.json').read_bytes()
    checkpoint_before = path.read_bytes()
    completion_before = (parent / 'COMPLETE.json').read_bytes()
    torch.save({'model': {'weight': torch.ones((1, 1))}}, path)
    write(parent / 'COMPLETE.json', {'best': {'pool_calibrated': {'checkpoint_sha256': runner.sha(path)}}})
    with pytest.raises(ValueError, match='Resume configuration, source or scientific inputs changed'):
        runner.train(a, c)
    assert (out / 'parent.json').read_bytes() == metadata_before
    path.write_bytes(checkpoint_before)
    (parent / 'COMPLETE.json').write_bytes(completion_before)
    runner.train(a, c)
    assert read(out / 'COMPLETE.json')['config'] == c
