import importlib.util
from pathlib import Path
import shutil

import numpy as np
import pytest

from phenoquery.common import pairs, read, sha, write
from test_matched_kge_report import evaluated_fixture
from test_matched_kge_scores import complete_campaign

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('matched_inference_comparison',
    REPO/'reproducibility/compare_matched_kge_inference.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
SCORE = 'campaign/fold0/original/U1/scores-raw_head.npz'


@pytest.fixture(scope='module')
def production_layout(complete_campaign, tmp_path_factory):
    root = tmp_path_factory.mktemp('matched-inference-fixture')/'complete'
    shutil.copytree(complete_campaign, root)
    for path in (root/'campaign').glob('fold*/*/*/scores-*.npz'):
        fold = path.relative_to(root/'campaign').parts[0]
        training = sorted({d for _,d in pairs(root/'data'/fold/'training.tsv')})
        with np.load(path, allow_pickle=False) as source:
            arrays = {name:source[name].copy() for name in source.files}
        arrays.update(train_diseases=np.asarray(training), train_scores=arrays['scores'][:len(training)].copy())
        np.savez_compressed(path, **arrays)
        marker = path.parent/'SCORED.json'
        complete = read(marker); complete['files'][path.name] = sha(path); write(marker, complete)
    return root


@pytest.fixture
def collections(production_layout, tmp_path):
    roots = [tmp_path/'reference', tmp_path/'replay']
    for root in roots:
        shutil.copytree(production_layout, root)
    return roots


def run(roots):
    a, b = roots
    return comparison.compare(a/'campaign', b/'campaign', a/'data', b/'data',
        a/'gene_phenotypes.csv', b/'gene_phenotypes.csv', expected_candidates=17)


def replace(root, values=None, transform=None):
    path = root/SCORE
    with np.load(path, allow_pickle=False) as source:
        arrays = {name:source[name].copy() for name in source.files}
    if values is not None:
        arrays['scores'] = values
    if transform is not None:
        transform(arrays)
    np.savez_compressed(path, **arrays)
    marker = path.parent/'SCORED.json'
    complete = read(marker)
    complete['files'][path.name] = sha(path)
    write(marker, complete)


def test_compares_all_choices_folds_and_graphs(collections):
    result = run(collections)
    assert result['status'] == 'passed'
    assert len(result['records']) == result['score_files'] == 400
    assert result['variants'] == 40 and result['training_runs'] == 100
    assert result['score_cells'] == 400*5*17
    assert result['association_rank_pairs_per_mode'] == 400*4
    assert result['changed_ranks'] == {'raw':0, 'calibrated':0}
    assert result['maximum_absolute_score_difference'] == 0


@pytest.mark.parametrize('change', ['small', 'large', 'raw-rank', 'calibrated-rank'])
def test_score_tolerance_and_both_rank_modes_are_independent_requirements(collections, change):
    a, b = collections
    values = np.random.default_rng(31).normal(size=(3,17))
    if change == 'raw-rank':
        values[0,0], values[0,1] = 0, 5e-8
        regenerated = values.copy(); regenerated[0,1] = -5e-8
    elif change == 'calibrated-rank':
        values[:,0], values[:,1] = [0,1,3], [10,12,16]
        regenerated = values.copy(); regenerated[0,1] += 1e-7
    else:
        regenerated = values + (1e-8 if change == 'small' else 1)
    replace(a, values); replace(b, regenerated)
    result = run(collections)
    assert result['status'] == ('passed' if change == 'small' else 'requires_review')
    assert result['score_values_outside_tolerance'] == (51 if change == 'large' else 0)
    if change in ('small', 'large'):
        assert result['changed_ranks'] == {'raw':0, 'calibrated':0}
    elif change == 'raw-rank':
        assert result['changed_ranks']['raw'] > 0
    else:
        assert result['changed_ranks']['raw'] == 0
        assert result['changed_ranks']['calibrated'] > 0


@pytest.mark.parametrize('problem', ['last-score', 'freeze', 'checksum', 'benchmark',
                                    'dtype', 'nan', 'extra-disease', 'order'])
def test_rejects_incomplete_or_inconsistent_collections(collections, problem):
    a, b = collections
    if problem == 'last-score':
        (b/'campaign/fold9/nobridge/PPOOL/scores-calibrated_head.npz').unlink()
    elif problem == 'freeze':
        path = b/'campaign/FROZEN.json'
        frozen = read(path); frozen['training'].pop('fold9/nobridge/PPOOL'); write(path, frozen)
    elif problem == 'checksum':
        (b/SCORE).write_bytes(b'changed')
    elif problem == 'benchmark':
        (b/'data/fold0/test.tsv').write_text('changed')
    else:
        def transform(arrays):
            if problem == 'dtype': arrays['scores'] = arrays['scores'].astype(np.float32)
            elif problem == 'nan': arrays['scores'][0,0] = np.nan
            elif problem == 'order': arrays['genes'] = arrays['genes'][::-1]
            else:
                arrays['diseases'] = np.append(arrays['diseases'], 'extra')
                arrays['scores'] = np.vstack([arrays['scores'], arrays['scores'][0]])
        replace(b, transform=transform)
    with pytest.raises((ValueError, FileNotFoundError)):
        run(collections)


def test_rejects_comparing_collection_with_itself(collections):
    with pytest.raises(ValueError, match='separate collections'):
        run([collections[0], collections[0]])


def test_rejects_scores_shared_through_symlinks(collections):
    a, b = collections
    (b/SCORE).unlink()
    (b/SCORE).symlink_to(a/SCORE)
    with pytest.raises(ValueError, match='same file'):
        run(collections)


def test_auxiliary_training_prediction_drift_is_detected(collections):
    def transform(arrays):
        arrays['train_scores'] += 1
    replace(collections[1], transform=transform)
    result = run(collections)
    assert result['status'] == 'requires_review'
    assert result['score_values_outside_tolerance'] == 2*17
    assert result['changed_ranks'] == {'raw':0, 'calibrated':0}
