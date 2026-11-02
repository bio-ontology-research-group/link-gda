"""Model, source-parity, sampler and evaluator tests for code/compositional_gda."""
import collections
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from compositional_gda import evaluate, models
from compositional_gda.data import PreparedData, QueryIndex, Sampler
from compositional_gda.models import query_loss

KGREASONING = Path(os.environ.get('COMPOSITIONAL_KGREASONING', Path.home() / 'Git' / 'KGReasoning'))
GEOMETRE = Path(os.environ.get('COMPOSITIONAL_GEOMETRE', Path.home() / 'Git' / 'geometrE'))
SOURCE_MODULES = ('util', 'dataloader', 'models', 'box', 'embeddings')
STUBS = ('tqdm', 'pandas', 'matplotlib', 'matplotlib.pyplot', 'matplotlib.patches', 'scipy', 'scipy.stats')

GENES = [f'G{i}' for i in range(7)]
TRAIN_DISEASES = [f'D{i}' for i in range(4)]
PHENOTYPES = [f'HP:{i}' for i in range(6)]
TERMS = ['GO:0', 'GO:1']
TRAINING = [('G0', 'D0'), ('G1', 'D0'), ('G2', 'D1'), ('G3', 'D2'), ('G4', 'D3'), ('G5', 'D3')]
PROFILES = {'D0': ['HP:0', 'HP:1'], 'D1': ['HP:1', 'HP:2'], 'D2': ['HP:3'], 'D3': ['HP:4', 'HP:5', 'HP:5'],
            'V0': ['HP:0', 'HP:1', 'HP:1'], 'V1': ['HP:2', 'HP:3'], 'V2': ['HP:4', 'HP:5'], 'V3': ['HP:0', 'HP:5'],
            'T0': ['HP:2']}
VALIDATION = [('G1', 'V0'), ('G2', 'V1'), ('G3', 'V2'), ('G4', 'V2'), ('G0', 'V3')]
TEST = [('G5', 'T0')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_prepared(directory, extra_relations=()):
    """Write a small prepared fold in the data worker's format."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = GENES + TRAIN_DISEASES + PHENOTYPES + TERMS
    entities = {name: index for index, name in enumerate(names)}
    base = ['associated_with', 'has_symptom', 'has_function', 'subclassof', *extra_relations]
    relations = {name: index for index, name in enumerate(base)}
    relations.update({'inverse:' + name: index + len(base) for index, name in enumerate(base)})
    forward = [(g, 'associated_with', d) for g, d in TRAINING]
    forward += [(d, 'has_symptom', p) for d in TRAIN_DISEASES for p in sorted(set(PROFILES[d]))]
    forward += [('G0', 'has_function', 'GO:0'), ('G2', 'has_function', 'GO:1'), ('GO:1', 'subclassof', 'GO:0'),
                ('HP:1', 'subclassof', 'HP:0'), ('HP:5', 'subclassof', 'HP:4')]
    triples = [(entities[h], relations[r], entities[t]) for h, r, t in forward]
    triples += [(entities[t], relations['inverse:' + r], entities[h]) for h, r, t in forward]
    np.savez(directory / 'graph.npz', triples=np.asarray(triples, dtype=np.int64))
    (directory / 'vocabulary.json').write_text(json.dumps({'entities': entities, 'relations': relations}))
    (directory / 'profiles.json').write_text(json.dumps(PROFILES))
    (directory / 'candidates.txt').write_text('\n'.join(GENES) + '\n')
    for name, rows in (('training', TRAINING), ('validation', VALIDATION), ('test', TEST)):
        (directory / f'{name}.tsv').write_text('Gene\tDisease\n' + ''.join(f'{g}\t{d}\n' for g, d in rows))
    outputs = {name: {'sha256': digest(directory / name)} for name in
               ('graph.npz', 'vocabulary.json', 'profiles.json', 'candidates.txt', 'training.tsv', 'validation.tsv',
                'test.tsv')}
    (directory / 'identity.json').write_text(json.dumps({'outputs': outputs}))
    return directory


def head_commit(repository):
    git = repository / '.git'
    if not git.is_dir():
        return None
    head = (git / 'HEAD').read_text().strip()
    if not head.startswith('ref: '):
        return head
    reference = head[5:]
    if (git / reference).exists():
        return (git / reference).read_text().strip()
    packed = git / 'packed-refs'
    for line in packed.read_text().splitlines() if packed.exists() else []:
        if line.endswith(' ' + reference):
            return line.split()[0]
    return None


def load_source(directory, name, label):
    """Import a pinned source module with absent plotting/stat packages stubbed and module names isolated."""
    saved = {module: sys.modules.pop(module) for module in SOURCE_MODULES if module in sys.modules}
    stubbed = []
    for module in STUBS:
        if module not in sys.modules:
            stub = types.ModuleType(module)
            stub.tqdm = lambda iterable, *a, **k: iterable
            stub.FancyArrowPatch = stub.FancyArrow = object
            stub.spearmanr = stub.rankdata = None
            sys.modules[module] = stub
            stubbed.append(module)
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(label, directory / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
        for loaded in SOURCE_MODULES:
            sys.modules.pop(loaded, None)
        sys.modules.update(saved)
        for module_name in stubbed:
            sys.modules.pop(module_name, None)
    return module


def require_source(repository, key):
    if not repository.exists():
        pytest.skip(f'pinned source {repository} not present')
    assert head_commit(repository) == models.SOURCES[key]['commit'], 'source checkout is not at the pinned commit'


@pytest.fixture(scope='module')
def kgreasoning():
    require_source(KGREASONING, 'kgreasoning')
    return load_source(KGREASONING, 'models', 'kgreasoning_models')


@pytest.fixture(scope='module')
def geometre():
    require_source(GEOMETRE, 'geometre')
    return load_source(GEOMETRE / 'src', 'models', 'geometre_models')


NAMES = {('e', ('r',)): '1p', ('e', ('r', 'r')): '2p'}
ENTITIES, RELATIONS, DIM, BATCH, NEGATIVES = 40, 6, 8, 5, 7


def batch(seed=3):
    generator = torch.Generator().manual_seed(seed)
    heads = torch.randint(ENTITIES, (BATCH,), generator=generator)
    rels = torch.randint(RELATIONS, (BATCH, 2), generator=generator)
    positives = torch.randint(ENTITIES, (BATCH,), generator=generator)
    negatives = torch.randint(ENTITIES, (BATCH, NEGATIVES), generator=generator)
    weights = torch.rand(BATCH, generator=generator) + 0.1
    return heads, rels, positives, negatives, weights


def mine_for(method, **options):
    config = {**models.METHOD_DEFAULTS[method], 'method': method, 'dim': DIM, **options}
    torch.manual_seed(11)
    return models.build_model(config, ENTITIES, RELATIONS)


def kgr_for(source, geo, gamma, **options):
    torch.manual_seed(11)
    return source.KGReasoning(ENTITIES, RELATIONS, DIM, gamma, geo, query_name_dict=NAMES, **options)


class FakeOptimizer:
    def zero_grad(self):
        pass

    def step(self):
        pass


def source_train_loss(model, train_step, heads, rels, positives, negatives, weights, **arguments):
    queries = [[int(h), int(r)] for h, r in zip(heads, rels[:, 0])]
    sample = (positives.clone(), negatives.clone(), weights.clone(), queries, [('e', ('r',))] * BATCH)
    args = types.SimpleNamespace(cuda=False, **arguments)
    return train_step(model, FakeOptimizer(), iter([sample]), args, 0)


def my_train_loss(model, heads, rels, positives, negatives, weights):
    query = model.encode(heads, rels[:, :1])
    positive = model.score(query, positives.unsqueeze(1)).squeeze(1)
    negative = model.score(query, negatives, negative=True)
    loss, positive_loss, negative_loss = query_loss(positive, negative, weights)
    auxiliary = model.auxiliary_losses()
    return loss + sum(auxiliary.values()) if auxiliary else loss, positive_loss, negative_loss, auxiliary


KGR_CASES = [('transe', 'vec', {}, {}), ('q2b', 'box', {'box_mode': ('none', 0.02)}, {}),
             ('betae', 'beta', {'beta_mode': (32, 2)}, {'beta_hidden': 32, 'beta_layers': 2})]


@pytest.mark.parametrize('method,geo,source_options,my_options', KGR_CASES)
def test_kgreasoning_initialisation_and_scores_match_source(kgreasoning, method, geo, source_options, my_options):
    gamma = models.METHOD_DEFAULTS[method]['gamma']
    source = kgr_for(kgreasoning, geo, gamma, **source_options)
    mine = mine_for(method, **my_options)
    source_state = source.state_dict()
    for name, value in mine.state_dict().items():
        if name != 'gamma':
            assert torch.equal(value, source_state[name]), name
    heads, rels, positives, negatives, weights = batch()
    embed = getattr(source, f'embed_query_{geo}')
    for structure, hops in ((('e', ('r',)), 1), (('e', ('r', 'r')), 2)):
        queries = torch.cat([heads.unsqueeze(1), rels[:, :hops]], dim=1)
        theirs = embed(queries, structure, 0)[:-1]
        ours = mine.encode(heads, rels[:, :hops])
        for left, right in zip(theirs, ours):
            assert torch.equal(left, right)
        entity = source.entity_embedding[negatives]
        if geo == 'vec':
            expected = source.cal_logit_vec(entity, theirs[0].unsqueeze(1))
        elif geo == 'box':
            expected = source.cal_logit_box(entity, theirs[0].unsqueeze(1), theirs[1].unsqueeze(1))
        else:
            distribution = torch.distributions.beta.Beta(theirs[0].unsqueeze(1), theirs[1].unsqueeze(1))
            expected = source.cal_logit_beta(source.entity_regularizer(entity), distribution)
        assert torch.equal(mine.score(ours, negatives, negative=True), expected)
    log = source_train_loss(source, kgreasoning.KGReasoning.train_step, heads, rels, positives, negatives, weights)
    loss, positive_loss, negative_loss, _ = my_train_loss(mine, heads, rels, positives, negatives, weights)
    assert log['loss'] == float(loss)
    assert log['positive_sample_loss'] == float(positive_loss)
    assert log['negative_sample_loss'] == float(negative_loss)


def geometre_source(source, answer_embedding):
    torch.manual_seed(11)
    return source.KGReasoning(ENTITIES, RELATIONS, DIM, 20.0, 0.2, 'pure_box', query_name_dict=NAMES,
                              transitive_ids=[0], inverse_ids=[1], with_answer_embedding=answer_embedding)


@pytest.mark.parametrize('answer_embedding', [False, True])
@pytest.mark.parametrize('mode', ['original', 'sign_corrected'])
def test_geometre_matches_released_code(geometre, answer_embedding, mode):
    source = geometre_source(geometre, answer_embedding)
    mine = mine_for('geometre', answer_embedding=answer_embedding, membership_mode=mode)
    source_state = source.state_dict()
    for name, value in mine.state_dict().items():
        if name not in ('gamma', 'alpha'):
            assert torch.equal(value, source_state[name]), name
    heads, rels, positives, negatives, weights = batch()
    with torch.no_grad():
        source.offset_embedding.weight[:3] *= -1
        mine.offset_embedding.weight[:3] *= -1
    for hops in (1, 2):
        queries = torch.cat([heads.unsqueeze(1), rels[:, :hops]], dim=1)
        box, *_ = source.embed_query_box(queries, f'{hops}p', False)
        ours = mine.encode(heads, rels[:, :hops])
        assert torch.equal(box.center, ours[0]) and torch.equal(box.offset, ours[1])
        box.center, box.offset = box.center.unsqueeze(1), box.offset.unsqueeze(1)
        table = source.answer_embedding if answer_embedding else source.center_embedding
        point = geometre.Box(table(negatives), as_point=True)
        expected = source.cal_logit_box(point, box, None, None, None, transitive=False, negative=True)
        assert torch.equal(mine.score(ours, negatives, negative=True), expected)
    positive_logit, negative_logit, membership_logit, *_ = source.forward_box(
        positives, negatives, weights, {('e', ('r',)): torch.cat([heads.unsqueeze(1), rels[:, :1]], 1)},
        {('e', ('r',)): list(range(BATCH))}, transitive=False)
    assert torch.equal(mine.membership_distance(), membership_logit)
    log = source_train_loss(source, geometre.KGReasoning.train_step, heads, rels, positives, negatives, weights,
                            transitive=False)
    loss, positive_loss, negative_loss, auxiliary = my_train_loss(mine, heads, rels, positives, negatives, weights)
    assert log['positive_sample_loss'] == float(positive_loss)
    assert log['negative_sample_loss'] == float(negative_loss)
    expected_membership = torch.nn.functional.softplus(membership_logit if mode == 'sign_corrected'
                                                       else -membership_logit).mean()
    assert torch.allclose(auxiliary['membership'], expected_membership, rtol=1e-6, atol=0)
    if mode == 'original':
        assert log['membership_loss'] == float(auxiliary['membership'])
        assert log['loss'] == float(loss)
    assert log['transitive_rel_loss'] == 0.0


def test_geometre_corner_penalty_uses_raw_offsets_only_in_membership():
    model = mine_for('geometre', membership_mode='original')
    with torch.no_grad():
        model.offset_embedding.weight.fill_(-0.5)
    raw = model.membership_distance()
    assert torch.allclose(raw, torch.full((ENTITIES,), (1.0 + 0.2 * 0.5 + 1.0) * DIM), rtol=1e-6, atol=1e-5)
    query = model.encode(torch.arange(3), torch.zeros(3, 1, dtype=torch.long))
    assert bool((query[1] >= 0).all())
    point = torch.arange(3).unsqueeze(1)
    assert torch.equal(model.score(query, point), model.score(query, point, negative=True))


def test_geometre_membership_sign_variants_have_opposite_gradient_direction():
    original = mine_for('geometre', membership_mode='original')
    corrected = mine_for('geometre', membership_mode='sign_corrected')
    original.auxiliary_losses()['membership'].backward()
    corrected.auxiliary_losses()['membership'].backward()
    distance = original.membership_distance().detach()
    weight_original = torch.sigmoid(-distance)
    weight_corrected = torch.sigmoid(distance)
    left = original.offset_embedding.weight.grad
    right = corrected.offset_embedding.weight.grad
    ratio = (weight_corrected / weight_original).unsqueeze(1)
    assert torch.allclose(right, -left * ratio, rtol=1e-4, atol=1e-7)


def test_build_refuses_unknown_membership_mode():
    with pytest.raises(ValueError):
        mine_for('geometre', membership_mode='paper')


def test_q2b_offsets_are_unclamped_upstream_default():
    model = mine_for('q2b')
    assert model.offset_activation == 'none'
    with torch.no_grad():
        model.offset_embedding.fill_(-1.0)
    _, offset = model.encode(torch.tensor([0]), torch.tensor([[0, 1]]))
    assert torch.equal(offset, torch.full((1, DIM), -2.0))


def test_query_index_weights_and_sampler_filtering_follow_source():
    triples = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 2], [3, 1, 0], [1, 0, 3]], dtype=np.int64)
    index = QueryIndex(triples, 5)
    assert len(index) == 3
    assert index.answer_set(0).tolist() == [1, 2]
    for query in range(len(index)):
        count = 4 + len(index.answer_set(query))
        assert index.weights[query] == torch.sqrt(1 / torch.Tensor([count])).item()
    sampler = Sampler(index, 2, 16, seed=0)
    for _ in range(6):
        heads, rels, positives, negatives, weights = sampler.next_batch()
        assert negatives.shape[1] == 16 and rels.shape[1] == 1
        for row, (head, relation) in enumerate(zip(heads, rels[:, 0])):
            query = int(np.flatnonzero((index.heads == head) & (index.relations == relation))[0])
            known = index.answer_set(query)
            assert positives[row] in known
            assert not np.isin(negatives[row], known).any()


def test_sampler_state_round_trip_reproduces_draws():
    index = QueryIndex(np.array([[i, i % 2, (i + 1) % 9] for i in range(9)], dtype=np.int64), 9)
    first = Sampler(index, 4, 3, seed=5)
    for _ in range(2):
        first.next_batch()
    state = json.loads(json.dumps(first.state_dict()))
    expected = [first.next_batch() for _ in range(4)]
    second = Sampler(index, 4, 3, seed=5)
    second.load_state_dict(state)
    for left, right in zip(expected, [second.next_batch() for _ in range(4)]):
        for a, b in zip(left, right):
            assert np.array_equal(a, b)
    assert first.epoch >= 1


def test_calibration_and_ranks_match_phenoquery_reference():
    from phenoquery import common
    generator = np.random.default_rng(0)
    scores = np.round(generator.normal(size=(5, 12)), 1)
    genes = [f'g{i}' for i in range(12)]
    diseases = [f'd{i}' for i in range(5)]
    rows = [(genes[i % 12], diseases[i % 5]) for i in range(9)]
    assert np.array_equal(evaluate.calibrate(scores), common.cal(scores))
    assert np.array_equal(evaluate.average_ranks(scores, diseases, genes, rows), common.ranks(scores, diseases, genes, rows))
    ranks = evaluate.average_ranks(scores, diseases, genes, rows)
    reference = common.metrics(ranks)
    for name in reference:
        assert evaluate.metrics(ranks)[name] == reference[name]
    assert evaluate.selection_key(reference) == common.key(reference)


def test_aggregation_keeps_profile_duplicates():
    matrix = np.array([[1.0, 0.0], [3.0, 2.0]])
    profiles = {'d': ['a', 'b', 'b']}
    result = evaluate.aggregate(matrix, ['a', 'b'], profiles, ['d'], 0.5)
    assert np.allclose(result, [[0.5 * 3 + 0.5 * (7 / 3), 0.5 * 2 + 0.5 * (4 / 3)]])


def test_selection_rule_fallback_gate_and_ties():
    def m(h10, h1, mrr=0.1, mr=5.0):
        return {'calibrated': {'hits@10': h10, 'hits@1': h1, 'mrr': mrr, 'mr': mr}}
    curve = [(0, {'forward': m(0.5, 0.2), 'inverse': m(0.5, 0.2)}),
             (100, {'forward': m(0.9, 0.1), 'inverse': m(0.6, 0.2)}),
             (200, {'forward': m(0.5, 0.2), 'inverse': m(0.6, 0.2)})]
    assert evaluate.replay_selection(curve) == {'forward': 0, 'inverse': 100}


@pytest.mark.parametrize('steps,message', [
    ((), 'empty'),
    ((100, 200), 'start at step 0'),
    ((200, 100, 0), 'start at step 0'),
    ((0, 200, 100), 'strictly increase'),
    ((0, 100, 100), 'strictly increase'),
    ((0, 0), 'strictly increase'),
    ((0, 100.0), 'integers'),
    ((False, 100), 'integers'),
])
def test_selection_replay_refuses_malformed_curves(steps, message):
    entry = {'forward': {'calibrated': {'hits@10': 0.5, 'hits@1': 0.2, 'mrr': 0.1, 'mr': 5.0}}}
    entry['inverse'] = entry['forward']
    with pytest.raises(ValueError, match=message):
        evaluate.replay_selection([(step, entry) for step in steps])


def test_selection_replay_accepts_numpy_integer_steps():
    def m(h10):
        return {'calibrated': {'hits@10': h10, 'hits@1': 0.2, 'mrr': 0.1, 'mr': 5.0}}
    curve = [(np.int64(0), {'forward': m(0.5), 'inverse': m(0.5)}),
             (np.int64(100), {'forward': m(0.6), 'inverse': m(0.4)})]
    assert evaluate.replay_selection(curve) == {'forward': 100, 'inverse': 0}


@pytest.mark.parametrize('method', ['transe', 'q2b', 'betae', 'geometre'])
def test_two_hop_scores_are_chunk_invariant_and_forward_inverse_differ(tmp_path, method):
    data = PreparedData(make_prepared(tmp_path / 'fold0'))
    options = {'beta_hidden': 16} if method == 'betae' else {}
    if method == 'geometre':
        options['membership_mode'] = 'original'
    config = {**models.METHOD_DEFAULTS[method], 'method': method, 'dim': 6, **options}
    torch.manual_seed(0)
    model = models.build_model(config, data.num_entities, data.num_relations)
    genes = [data.entities[g] for g in data.candidates]
    phenotypes = [data.entities[p] for p in PHENOTYPES]
    full = evaluate.path_scores(model, genes, data.relation_ids('forward'), phenotypes, 'cpu', 1000, 10 ** 9, 6)
    small = evaluate.path_scores(model, genes, data.relation_ids('forward'), phenotypes, 'cpu', 2, 13, 6)
    assert np.allclose(full, small, rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        query = model.encode(torch.tensor(genes), torch.tensor([data.relation_ids('forward')] * len(genes)))
        direct = model.score(query, torch.tensor([phenotypes]), negative=True).numpy()
    assert np.array_equal(full, direct)
    inverse = evaluate.path_scores(model, phenotypes, data.relation_ids('inverse'), genes, 'cpu', 1000, 10 ** 9, 6)
    assert inverse.shape == full.T.shape and not np.allclose(inverse, full.T)


def test_validation_never_looks_up_disease_entities(tmp_path):
    data = PreparedData(make_prepared(tmp_path / 'fold0'))
    diseases, rows = data.validation_panel(3)
    assert diseases == ['V0', 'V1', 'V2'] and all(d not in data.entities for d in diseases)
    config = {**models.METHOD_DEFAULTS['transe'], 'method': 'transe', 'dim': 4, 'query_chunk': 3,
              'score_chunk_elements': 50, 'aggregation_weight': 0.5}
    torch.manual_seed(0)
    model = models.build_model(config, data.num_entities, data.num_relations)
    looked_up = []
    original = model.encode

    def recording(anchors, relations):
        looked_up.extend(anchors.tolist())
        return original(anchors, relations)
    model.encode = recording
    arrays, readouts = evaluate.validate(model, data, diseases, rows, config, 'cpu')
    disease_ids = {data.entities[d] for d in TRAIN_DISEASES}
    assert not disease_ids & set(looked_up)
    assert arrays['forward_phenotype_scores'].shape == (len(arrays['phenotypes']), len(GENES))
    recomputed = evaluate.recompute_from_arrays(arrays, PROFILES, 0.5)
    assert recomputed == readouts
    assert arrays['scoring_query_chunk'].item() == 3 and arrays['scoring_score_chunk_elements'].item() == 50
    assert arrays['scoring_device'].item() == 'cpu' and arrays['scoring_device_name'].item() == 'cpu'


def test_scoring_metadata_survives_saved_archive(tmp_path):
    config = {'query_chunk': 7, 'score_chunk_elements': 2 ** 25}
    path = tmp_path / 'step.npz'
    np.savez_compressed(path, **evaluate.scoring_metadata(config, torch.device('cpu')))
    with np.load(path, allow_pickle=False) as archive:
        assert archive['scoring_query_chunk'].item() == 7
        assert archive['scoring_score_chunk_elements'].item() == 2 ** 25
        assert archive['scoring_device'].item() == 'cpu' and archive['scoring_device_name'].item() == 'cpu'


def test_prepared_data_refuses_leakage_and_changed_files(tmp_path):
    directory = make_prepared(tmp_path / 'ok')
    PreparedData(directory)
    (directory / 'candidates.txt').write_text('G0\n')
    with pytest.raises(ValueError, match='changed'):
        PreparedData(directory)
    shortcut = make_prepared(tmp_path / 'shortcut', extra_relations=('causes_phenotype',))
    with pytest.raises(ValueError, match='causes_phenotype'):
        PreparedData(shortcut)
