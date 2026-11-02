"""A run directory with history but no ``resume.pt`` is refused before anything is overwritten or trained."""
import json
import shutil

import pytest
import torch

from compositional_gda import train
from test_compositional_models import make_prepared


def write_config(directory, method='transe'):
    config = {'data_dir': 'fold0', 'method': method, 'dim': 4, 'steps': 6, 'eval_every': 2, 'batch_size': 4,
              'negative_sample_size': 3, 'learning_rate': 0.05, 'validation_diseases': 3, 'query_chunk': 3,
              'score_chunk_elements': 64, 'checkpoint_every_steps': 1}
    path = directory / f'{method}.json'
    path.write_text(json.dumps(config))
    return path


def run(config, run_dir, *extra):
    return train.main(['--config', str(config), '--run-dir', str(run_dir), '--device', 'cpu', *extra])


def snapshot(run_dir):
    return {str(path.relative_to(run_dir)): path.read_bytes() for path in sorted(run_dir.rglob('*'))
            if path.is_file() and path.name != '.lock'}


def train_records(run_dir):
    records = [json.loads(line) for line in (run_dir / 'training.jsonl').read_text().splitlines()]
    for record in records:
        record.pop('seconds')
    return records


@pytest.fixture
def workspace(tmp_path):
    make_prepared(tmp_path / 'fold0')
    return tmp_path


def test_paused_run_without_resume_checkpoint_is_refused_unchanged(workspace):
    config = write_config(workspace)
    run_dir = workspace / 'run'
    assert run(config, run_dir, '--stop-after', '3') == train.EXIT_PAUSE
    (run_dir / 'resume.pt').unlink()
    before = snapshot(run_dir)
    assert 'checkpoints/step-000002.pt' in before and 'progress.json' in before
    with pytest.raises(train.OrphanedHistory, match='no resume.pt'):
        run(config, run_dir)
    assert snapshot(run_dir) == before


@pytest.mark.parametrize('artifact', ['progress.json', 'training.jsonl', 'checkpoints/step-000000.pt',
                                      'validation/step-000000.npz', 'selected-forward.pt',
                                      'selected-inverse-validation.npz'])
def test_each_history_artifact_alone_is_refused(workspace, artifact):
    config = write_config(workspace)
    source = workspace / 'source'
    assert run(config, source, '--stop-after', '1') == train.EXIT_PAUSE
    run_dir = workspace / 'run'
    (run_dir / artifact).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source / 'identity.json', run_dir / 'identity.json')
    shutil.copy(source / artifact, run_dir / artifact)
    before = snapshot(run_dir)
    with pytest.raises(train.OrphanedHistory, match=artifact):
        run(config, run_dir)
    assert snapshot(run_dir) == before


def test_history_without_identity_is_refused_before_identity_is_written(workspace):
    config = write_config(workspace)
    run_dir = workspace / 'run'
    (run_dir / 'checkpoints').mkdir(parents=True)
    (run_dir / 'checkpoints' / 'step-000000.pt.tmp').write_bytes(b'partial')
    with pytest.raises(train.OrphanedHistory):
        run(config, run_dir)
    assert not (run_dir / 'identity.json').exists()


def test_interrupted_initialisation_with_identity_only_recovers(workspace, monkeypatch):
    config = write_config(workspace)
    reference = workspace / 'reference'
    assert run(config, reference) == train.EXIT_COMPLETE
    run_dir = workspace / 'run'

    def killed(*args, **kwargs):
        raise KeyboardInterrupt
    original = train.validate
    monkeypatch.setattr(train, 'validate', killed)
    with pytest.raises(KeyboardInterrupt):
        run(config, run_dir)
    monkeypatch.setattr(train, 'validate', original)
    (run_dir / 'training.jsonl').write_bytes(b'')
    (run_dir / 'resume.pt.tmp').write_bytes(b'partial')
    assert set(snapshot(run_dir)) == {'identity.json', 'segments.jsonl', 'training.jsonl', 'resume.pt.tmp'}
    assert run(config, run_dir) == train.EXIT_COMPLETE
    expected, actual = snapshot(reference), snapshot(run_dir)
    for name in expected:
        if name == 'identity.json' or name.endswith('.npz'):
            assert actual[name] == expected[name], name
        elif name.endswith('.pt') and name != 'resume.pt':
            left = torch.load(reference / name, map_location='cpu', weights_only=False)['model']
            right = torch.load(run_dir / name, map_location='cpu', weights_only=False)['model']
            assert left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left), name
    assert train_records(run_dir) == train_records(reference)


def test_completed_run_shortcut_survives_missing_resume_checkpoint(workspace):
    config = write_config(workspace)
    run_dir = workspace / 'run'
    assert run(config, run_dir) == train.EXIT_COMPLETE
    (run_dir / 'resume.pt').unlink()
    before = snapshot(run_dir)
    assert run(config, run_dir) == train.EXIT_COMPLETE
    assert snapshot(run_dir) == before
