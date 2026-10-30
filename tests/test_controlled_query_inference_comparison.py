import importlib.util
from pathlib import Path
import shutil

import numpy as np
import pytest

from phenoquery.common import cal, metrics, pairs, ranks, read, sha, write
from test_controlled_query_report import make_campaign

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inference_comparison',
    REPO/'reproducibility/compare_controlled_query_inference.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
BUNDLE = 'ultra_4g-seed1/test/fold0/bridge_pool'


@pytest.fixture
def collections(tmp_path):
    reference = make_campaign(tmp_path)
    replay = tmp_path/'replay'
    shutil.copytree(reference, replay)
    return reference, replay


def replace_predictions(root, values):
    folder = root/BUNDLE
    with np.load(folder/'scores.npz', allow_pickle=False) as source:
        scores = {name: source[name].copy() for name in source.files}
    with np.load(folder/'ranks.npz', allow_pickle=False) as source:
        saved = {name: source[name].copy() for name in source.files}
    scores['scores'] = values
    rows = pairs(root/'ultra_4g-seed1/data/fold0/test.tsv')
    recalculated = {mode: ranks(matrix, scores['diseases'], scores['genes'], rows)
                    for mode, matrix in [('raw', values), ('calibrated', cal(values))]}
    saved.update(recalculated)
    np.savez_compressed(folder/'scores.npz', **scores)
    np.savez_compressed(folder/'ranks.npz', **saved)
    complete = read(folder/'COMPLETE.json')
    complete['metrics'] = {mode: metrics(values) for mode, values in recalculated.items()}
    complete['files'] = {name: sha(folder/name) for name in ('scores.npz', 'ranks.npz')}
    write(folder/'COMPLETE.json', complete)


def test_complete_collections_compare_every_bundle(collections):
    result = comparison.compare(*collections, expected_candidates=17)
    assert result['status'] == 'passed' and len(result['records']) == result['score_bundles'] == 480
    assert result['score_cells'] == 480*3*17
    assert result['association_rank_pairs_per_mode'] == 480*4
    assert result['changed_ranks'] == {'raw': 0, 'calibrated': 0}
    assert result['maximum_absolute_score_difference'] == 0
    assert len(result['input_sha256']) == 10


def test_small_score_difference_passes_with_identical_ranks(collections):
    reference, replay = collections
    values = np.random.default_rng(31).normal(size=(3, 17))
    replace_predictions(reference, values)
    replace_predictions(replay, values+1e-8)
    result = comparison.compare(reference, replay, expected_candidates=17)
    assert result['status'] == 'passed'
    record = next(r for r in result['records'] if r['bundle'] == BUNDLE)
    assert record['reference_evidence']['scores.npz'] != record['replay_evidence']['scores.npz']
    assert 0 < record['maximum_absolute_score_difference'] < comparison.ATOL


def test_score_tolerance_does_not_excuse_changed_rank(collections):
    reference, replay = collections
    values = np.random.default_rng(31).normal(size=(3, 17))
    values[0, 0] = 0
    values[0, 1] = 5e-8
    regenerated = values.copy()
    regenerated[0, 1] = -5e-8
    replace_predictions(reference, values)
    replace_predictions(replay, regenerated)
    result = comparison.compare(reference, replay, expected_candidates=17)
    assert result['status'] == 'requires_review'
    assert result['score_values_outside_tolerance'] == 0
    assert result['changed_ranks']['raw'] > 0


def test_identical_ranks_do_not_excuse_large_score_error(collections):
    reference, replay = collections
    values = np.random.default_rng(31).normal(size=(3, 17))
    replace_predictions(reference, values)
    replace_predictions(replay, values+1)
    result = comparison.compare(reference, replay, expected_candidates=17)
    assert result['status'] == 'requires_review'
    assert result['score_values_outside_tolerance'] == 3*17
    assert result['changed_ranks'] == {'raw': 0, 'calibrated': 0}


def test_calibrated_rank_change_requires_review_when_raw_ranks_agree(collections):
    reference, replay = collections
    values = np.random.default_rng(31).normal(size=(3, 17))
    values[:, 0] = [0, 1, 3]
    values[:, 1] = [10, 12, 16]
    regenerated = values.copy()
    regenerated[0, 1] += 1e-7
    replace_predictions(reference, values)
    replace_predictions(replay, regenerated)
    result = comparison.compare(reference, replay, expected_candidates=17)
    assert result['status'] == 'requires_review'
    assert result['score_values_outside_tolerance'] == 0
    assert result['changed_ranks']['raw'] == 0
    assert result['changed_ranks']['calibrated'] > 0


@pytest.mark.parametrize('failure', ['last-bundle', 'global-selection', 'dishonest-metrics'])
def test_rejects_incomplete_or_inconsistent_evidence(collections, failure):
    reference, replay = collections
    if failure == 'last-bundle':
        (replay/'ultraquery-seed2/test/fold9/typed_and_frozen/scores.npz').unlink()
    elif failure == 'global-selection':
        path = replay/'GLOBAL-SELECTION.json'
        changed = read(path); changed['selection_data'] = 'different freeze'; write(path, changed)
    else:
        path = replay/BUNDLE/'COMPLETE.json'
        changed = read(path); changed['metrics']['raw']['mr'] += 1; write(path, changed)
    with pytest.raises((ValueError, FileNotFoundError)):
        comparison.compare(reference, replay, expected_candidates=17)


def test_refuses_comparison_with_same_collection(collections):
    with pytest.raises(ValueError, match='separate collections'):
        comparison.compare(collections[0], collections[0], expected_candidates=17)
