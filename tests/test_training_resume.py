import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from pykeen.models import TransD
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import TriplesFactory
from link_gda.head_selection import FixedBudgetHeadCallback
from link_gda.training_resume import train_resumable
from link_gda.filtered_negative_sampling import GenePoolNegativeSampler
from link_gda.training_resume import train_profile_resumable
from link_gda.profile_ranking import ProfileRanker
from link_gda.head_experiment import profile_metrics
import pandas as pd
from training.retrain_nobridge_kge import CombinedValidationCallback, DualTracker


def setup(directory, sampler='untyped'):
    factory = TriplesFactory.from_labeled_triples(np.array([
        ['g1', 'causes_phenotype', 'p1'], ['g2', 'causes_phenotype', 'p2'],
        ['g3', 'causes_phenotype', 'p1'], ['g4', 'causes_phenotype', 'p2'],
        ['p1', 'subclass', 'root'], ['p2', 'subclass', 'root']]), create_inverse_triples=True)
    model = TransD(triples_factory=factory, embedding_dim=4, relation_dim=4, random_seed=4)
    kwargs = {'num_negs_per_pos': 2}
    if sampler == 'typed':
        kwargs.update(target_relations=[2*factory.relation_to_id['causes_phenotype']],
                      gene_pool=[factory.entity_to_id[f'g{i}'] for i in range(1,5)])
    loop = SLCWATrainingLoop(model=model, triples_factory=factory,
                            optimizer=torch.optim.Adam(model.get_grad_params(), lr=.01),
                            negative_sampler=GenePoolNegativeSampler if sampler == 'typed' else None,
                            negative_sampler_kwargs=kwargs, automatic_memory_optimization=False)
    def evaluate():
        values = model.predict_hrt(factory.mapped_triples).detach().flatten()
        ranks = 1 + (values[:, None] < values[None, :]).sum(1)
        result = {'mr': float(ranks.float().mean()), 'mrr': float((1/ranks).mean()),
                  'hits@1': float((ranks<=1).float().mean()), 'hits@10': float((ranks<=3).float().mean())}
        return {'raw': result, 'calibrated': result}
    callback = FixedBudgetHeadCallback(model, evaluate, directory/'selected', directory/'curve.json',
                                       frequency=2, resume=True)
    return factory, model, loop, callback


def run(directory, stop_after=None, sampler='untyped', budget=6):
    directory.mkdir(exist_ok=True)
    factory, model, loop, callback = setup(directory, sampler)
    result = train_resumable(loop, factory, directory=directory/'resume',
                             identity={'seed': 4, 'sampler': sampler}, callbacks=[callback],
                             total_epochs=budget, batch_size=4, sub_batch_size=2,
                             checkpoint_frequency=1, stop_after=stop_after)
    return result, model, loop, callback


def equal(a, b):
    if isinstance(a, torch.Tensor): torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b


@pytest.mark.parametrize('sampler', ['untyped', 'typed'])
def test_actual_pykeen_interruption_matches_uninterrupted(tmp_path, sampler):
    full, full_model, full_loop, full_cb = run(tmp_path/'full', sampler=sampler)
    full_torch_rng = torch.get_rng_state()
    full_numpy_draw = np.random.random(5)
    full_python_draw = random.random()
    partial, _, _, _ = run(tmp_path/'resumed', stop_after=3, sampler=sampler)
    assert partial == {'epochs': 3, 'early_stopped': False, 'complete': False}
    done, model, loop, callback = run(tmp_path/'resumed', sampler=sampler)
    assert done == full == {'epochs': 6, 'early_stopped': False, 'complete': True}
    equal(model.state_dict(), full_model.state_dict())
    equal(loop.optimizer.state_dict(), full_loop.optimizer.state_dict())
    equal(loop.losses_per_epochs, full_loop.losses_per_epochs)
    equal(torch.get_rng_state(), full_torch_rng)
    np.testing.assert_array_equal(np.random.random(5), full_numpy_draw)
    assert random.random() == full_python_draw
    assert callback.curve == full_cb.curve
    for arm in callback.best:
        assert callback.best[arm]['epoch'] == full_cb.best[arm]['epoch']
        equal(torch.load(callback.best[arm]['checkpoint'], weights_only=True),
              torch.load(full_cb.best[arm]['checkpoint'], weights_only=True))


def test_resume_rejects_changed_epoch_budget(tmp_path):
    run(tmp_path/'run', stop_after=3)
    with pytest.raises(ValueError, match='configuration or scientific inputs'):
        run(tmp_path/'run', budget=7)


def test_uncommitted_selection_does_not_replace_snapshot(tmp_path):
    _, full_model, _, full_cb = run(tmp_path/'full')
    _, model, _, callback = run(tmp_path/'resumed', stop_after=3)
    # Simulate a successful later validation followed by a crash before commit.
    for path in callback.checkpoint_paths.values():
        temporary = path.with_suffix('.pt.new')
        torch.save({'wrong': torch.tensor(42)}, temporary)
        temporary.replace(path)
    curve_tmp = callback.curve_path.with_suffix('.json.new')
    curve_tmp.write_text('uncommitted curve')
    curve_tmp.replace(callback.curve_path)
    _, model, _, callback = run(tmp_path/'resumed')
    equal(model.state_dict(), full_model.state_dict())
    assert callback.curve == full_cb.curve


def test_selected_weights_without_optimizer_state_are_rejected(tmp_path):
    root = tmp_path/'run'
    root.mkdir()
    torch.save({'weight': torch.tensor(1)}, root/'selected_raw_head.pt')
    with pytest.raises(ValueError, match='cannot substitute'):
        run(root)


def profile_run(directory, stop_after=None):
    directory.mkdir(exist_ok=True)
    genes = [f'g{i}' for i in range(8)]
    factory = TriplesFactory.from_labeled_triples(np.array([
        [g, 'causes_phenotype', f'p{i % 2}'] for i,g in enumerate(genes)
    ]), create_inverse_triples=True)
    model = TransD(triples_factory=factory, embedding_dim=4, relation_dim=4, random_seed=4)
    train = pd.DataFrame({'Gene': ['g0', 'g1', 'g2'], 'Disease': ['d0', 'd1', 'd2']})
    val = pd.DataFrame({'Gene': ['g3', 'g4'], 'Disease': ['d3', 'd4']})
    profiles = {f'd{i}': [f'p{i % 2}'] for i in range(5)}
    ranker = ProfileRanker(model, factory, train, profiles, genes, validation_pairs=val,
                           seed=4, learn_pooling=True, random_negatives=2,
                           hard_negatives=1, proposal_size=3)
    optimizer = torch.optim.Adam(ranker.parameters(), lr=.01)
    callback = FixedBudgetHeadCallback(ranker,
        lambda: profile_metrics(ranker.score_profiles(val.Disease), val, genes),
        directory/'selected', directory/'curve.json', frequency=1, resume=True)
    result = train_profile_resumable(ranker, optimizer, factory, directory=directory/'resume',
                                     identity={'seed': 4}, callback=callback, total_epochs=4,
                                     accumulation_steps=2, stop_after=stop_after)
    return result, ranker, optimizer, callback


def test_profile_pooling_optimizer_and_sampler_resume_exactly(tmp_path):
    result, full, full_opt, full_cb = profile_run(tmp_path/'full')
    full_rng = torch.get_rng_state()
    partial, _, _, _ = profile_run(tmp_path/'resumed', stop_after=2)
    assert partial['epochs'] == 2 and not partial['complete']
    resumed, ranker, opt, cb = profile_run(tmp_path/'resumed')
    equal(resumed, result)
    equal(ranker.state_dict(), full.state_dict())
    equal(opt.state_dict(), full_opt.state_dict())
    equal(ranker.rng_state(), full.rng_state())
    equal(torch.get_rng_state(), full_rng)
    assert cb.curve == full_cb.curve
    for arm in cb.best:
        assert cb.best[arm]['epoch'] == full_cb.best[arm]['epoch']
        equal(torch.load(cb.best[arm]['checkpoint'], weights_only=True),
              torch.load(full_cb.best[arm]['checkpoint'], weights_only=True))


def test_scientific_early_stopping_patience_survives_interruption(tmp_path):
    def execute(directory, stop=None):
        directory.mkdir(exist_ok=True)
        factory, model, loop, _ = setup(directory)
        pairs = pd.DataFrame({'Gene': ['g1', 'g2'], 'Disease': ['d0', 'd1']})
        class ConstantScorer:
            def score_diseases(self, model, diseases):
                return {d: np.zeros(4) for d in diseases}
        callback = CombinedValidationCallback(model, pairs, ['g1','g2','g3','g4'],
            ConstantScorer(), DualTracker(directory/'selected'), profile_metrics,
            frequency=2, patience_evals=2, total_epochs=10)
        result = train_resumable(loop, factory, directory=directory/'resume',
            identity={'early_stop_test': True}, callbacks=[callback], total_epochs=10,
            batch_size=4, sub_batch_size=2, checkpoint_frequency=1, stop_after=stop)
        return result, model, callback
    full, full_model, full_cb = execute(tmp_path/'full')
    assert full == {'epochs': 6, 'early_stopped': True, 'complete': True}
    partial, _, _ = execute(tmp_path/'resumed', stop=3)
    assert not partial['complete']
    result, model, cb = execute(tmp_path/'resumed')
    assert result == full
    equal(model.state_dict(), full_model.state_dict())
    assert cb.evals_without_improvement == full_cb.evals_without_improvement == 2
    # JSON serialization converts the metrics tuples to lists on restoration.
    assert json.loads(json.dumps(cb.curve)) == json.loads(json.dumps(full_cb.curve))
    already_done, again, again_cb = execute(tmp_path/'resumed')
    assert already_done == full
    equal(again.state_dict(), full_model.state_dict())
    assert len(again_cb.curve) == 3
