"""Interrupted versus uninterrupted runs (CPU, or CUDA via COMPOSITIONAL_TEST_DEVICE), pause/exit protocol, identity refusal and recomputation."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from compositional_gda import recompute, train
from test_compositional_models import PROFILES, make_prepared

VARIANTS = [('transe', {}), ('q2b', {}), ('betae', {'beta_hidden': 16}),
            ('geometre', {'membership_mode': 'original'}), ('geometre', {'membership_mode': 'sign_corrected'})]
IDS = ['transe', 'q2b', 'betae', 'geometre-original', 'geometre-sign_corrected']


def write_config(directory, method, options, **overrides):
    config = {'data_dir': 'fold0', 'method': method, 'dim': 4, 'steps': 6, 'eval_every': 2, 'batch_size': 4,
              'negative_sample_size': 3, 'learning_rate': 0.05, 'validation_diseases': 3, 'query_chunk': 3,
              'score_chunk_elements': 64, 'checkpoint_every_steps': 1, **options, **overrides}
    path = directory / f'{method}-{options.get("membership_mode", "")}-{len(overrides)}.json'
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def workspace(tmp_path):
    make_prepared(tmp_path / 'fold0')
    return tmp_path


def selected_device():
    """Device for ``run``: ``COMPOSITIONAL_TEST_DEVICE`` is ``cpu`` (default) or ``cuda``; CUDA never falls back."""
    device = os.environ.get('COMPOSITIONAL_TEST_DEVICE', 'cpu')
    if device not in ('cpu', 'cuda'):
        pytest.fail(f'COMPOSITIONAL_TEST_DEVICE must be cpu or cuda, got {device!r}')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.fail('COMPOSITIONAL_TEST_DEVICE=cuda but CUDA is unavailable')
    return device


def run(config, run_dir, *extra):
    return train.main(['--config', str(config), '--run-dir', str(run_dir), '--device', selected_device(), *extra])


def load_state(run_dir):
    return torch.load(run_dir / 'resume.pt', map_location='cpu', weights_only=False)


def assert_same(left, right, path=''):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_same(left[key], right[key], f'{path}/{key}')
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, f'{path}/{index}')
    else:
        assert left == right, path


def train_records(run_dir):
    records = [json.loads(line) for line in (run_dir / 'training.jsonl').read_text().splitlines()]
    for record in records:
        record.pop('seconds')
    return records


def assert_runs_equal(reference, candidate):
    left, right = load_state(reference), load_state(candidate)
    for key in ('step', 'model', 'optimizer', 'scheduler', 'sampler', 'curve', 'best', 'baseline', 'identity_sha256'):
        assert_same(left[key], right[key], key)
    assert_same(left['rng'], right['rng'], 'rng')
    assert train_records(reference) == train_records(candidate)
    for entry in left['curve']:
        with np.load(reference / entry['validation']) as a, np.load(candidate / entry['validation']) as b:
            assert a.files == b.files
            for name in a.files:
                assert np.array_equal(a[name], b[name]), (entry['step'], name)
        assert_same(torch.load(reference / entry['checkpoint'], weights_only=False),
                    torch.load(candidate / entry['checkpoint'], weights_only=False), entry['checkpoint'])
    for readout in ('forward', 'inverse'):
        assert_same(torch.load(reference / f'selected-{readout}.pt', weights_only=False),
                    torch.load(candidate / f'selected-{readout}.pt', weights_only=False), readout)
    completion = json.loads((candidate / 'COMPLETE.json').read_text())
    assert completion['best'] == json.loads((reference / 'COMPLETE.json').read_text())['best']


@pytest.mark.parametrize('method,options', VARIANTS, ids=IDS)
def test_interrupted_runs_reproduce_uninterrupted_run(workspace, method, options):
    config = write_config(workspace, method, options)
    reference = workspace / 'reference'
    assert run(config, reference) == train.EXIT_COMPLETE
    progress = json.loads((reference / 'progress.json').read_text())
    assert [entry['step'] for entry in progress['curve']] == [0, 2, 4, 6]
    assert set(progress['best']) == {'forward', 'inverse'}
    identity = json.loads((reference / 'identity.json').read_text())
    assert identity['config'].get('membership_mode') == options.get('membership_mode')
    assert identity['device_type'] == selected_device()
    for entry in progress['curve']:
        with np.load(reference / entry['validation'], allow_pickle=False) as archive:
            assert archive['scoring_device'].item().split(':')[0] == selected_device()
            assert archive['scoring_query_chunk'].item() == 3 and archive['scoring_score_chunk_elements'].item() == 64
            if selected_device() == 'cuda':
                assert archive['scoring_device_name'].item() == torch.cuda.get_device_name()

    paused = workspace / 'paused'
    assert run(config, paused, '--stop-after', '3') == train.EXIT_PAUSE
    assert json.loads((paused / 'progress.json').read_text())['step'] == 3
    with (paused / 'training.jsonl').open('a') as handle:
        handle.write(json.dumps({'event': 'train', 'step': 99}) + '\n')
    assert run(config, paused, '--stop-before-validation', '4') == train.EXIT_PAUSE
    pending = json.loads((paused / 'progress.json').read_text())
    assert pending['step'] == 4 and pending['pending_validation'] and pending['curve'][-1]['step'] == 2
    before = load_state(paused)
    assert run(config, paused, '--stop-after', '5') == train.EXIT_PAUSE
    after = load_state(paused)
    assert [entry['step'] for entry in after['curve']] == [0, 2, 4]
    assert run(config, paused) == train.EXIT_COMPLETE
    assert before['curve'] == after['curve'][:-1]
    assert_runs_equal(reference, paused)
    assert run(config, paused) == train.EXIT_COMPLETE
    summary = recompute.recompute_run(paused, workspace / 'fold0')
    assert summary['selected_step'] == {readout: progress['best'][readout]['step'] for readout in ('forward', 'inverse')}


@pytest.mark.parametrize('value,message', [('cuda', 'CUDA is unavailable'), ('gpu', 'must be cpu or cuda')])
def test_device_option_fails_instead_of_falling_back(monkeypatch, value, message):
    monkeypatch.setenv('COMPOSITIONAL_TEST_DEVICE', value)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(pytest.fail.Exception, match=message):
        selected_device()


def test_device_option_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv('COMPOSITIONAL_TEST_DEVICE', raising=False)
    assert selected_device() == 'cpu'


def test_pending_replay_leaves_training_state_bitwise_unchanged(workspace):
    config = write_config(workspace, 'q2b', {})
    run_dir = workspace / 'boundary'
    assert run(config, run_dir, '--stop-before-validation', '2') == train.EXIT_PAUSE
    before = load_state(run_dir)
    assert run(config, run_dir, '--stop-after', '2') == train.EXIT_PAUSE
    after = load_state(run_dir)
    for key in ('step', 'model', 'optimizer', 'sampler', 'rng', 'baseline'):
        assert_same(before[key], after[key], key)
    assert after['curve'][:-1] == before['curve'] and after['curve'][-1]['step'] == 2


def test_signal_during_validation_pauses_with_pending_boundary(workspace, monkeypatch):
    config = write_config(workspace, 'transe', {})
    reference = workspace / 'reference'
    assert run(config, reference) == train.EXIT_COMPLETE
    calls = {'count': 0}
    original = train.validate

    def interrupted(*args, **kwargs):
        calls['count'] += 1
        if calls['count'] == 2:
            os.kill(os.getpid(), signal.SIGUSR1)
        return original(*args, **kwargs)
    monkeypatch.setattr(train, 'validate', interrupted)
    run_dir = workspace / 'signalled'
    assert run(config, run_dir) == train.EXIT_PAUSE
    progress = json.loads((run_dir / 'progress.json').read_text())
    assert progress['step'] == 2 and progress['pending_validation']
    segments = [json.loads(line) for line in (run_dir / 'segments.jsonl').read_text().splitlines()]
    assert segments[-1]['reason'] == 'signal' and segments[-1]['signal'] == 'SIGUSR1'
    monkeypatch.setattr(train, 'validate', original)
    assert run(config, run_dir) == train.EXIT_COMPLETE
    assert_runs_equal(reference, run_dir)


def test_deadline_pauses_before_initial_validation_and_resumes(workspace):
    config = write_config(workspace, 'geometre', {'membership_mode': 'original'})
    reference = workspace / 'reference'
    assert run(config, reference) == train.EXIT_COMPLETE
    run_dir = workspace / 'deadline'
    assert run(config, run_dir, '--deadline-seconds', '0') == train.EXIT_PAUSE
    progress = json.loads((run_dir / 'progress.json').read_text())
    assert progress['step'] == 0 and progress['curve'] == [] and progress['pending_validation']
    assert run(config, run_dir) == train.EXIT_COMPLETE
    assert_runs_equal(reference, run_dir)


def test_changed_identity_is_refused(workspace):
    config = write_config(workspace, 'transe', {})
    run_dir = workspace / 'run'
    assert run(config, run_dir, '--stop-after', '1') == train.EXIT_PAUSE
    changed = write_config(workspace, 'transe', {}, learning_rate=0.01)
    with pytest.raises(train.IdentityMismatch):
        run(changed, run_dir)
    assert run(config, run_dir) == train.EXIT_COMPLETE
    with pytest.raises(train.IdentityMismatch):
        run(changed, run_dir)


def test_changed_committed_artifact_is_refused(workspace):
    config = write_config(workspace, 'transe', {})
    run_dir = workspace / 'run'
    assert run(config, run_dir, '--stop-after', '3') == train.EXIT_PAUSE
    (run_dir / 'validation' / 'step-000002.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Committed artifact changed'):
        run(config, run_dir)


def test_uncommitted_selection_is_rolled_back_on_resume(workspace):
    config = write_config(workspace, 'transe', {})
    run_dir = workspace / 'run'
    assert run(config, run_dir, '--stop-after', '3') == train.EXIT_PAUSE
    (run_dir / 'selected-forward.pt').write_bytes(b'uncommitted')
    state = load_state(run_dir)
    assert run(config, run_dir, '--stop-after', '4') == train.EXIT_PAUSE
    step = state['best']['forward']['step']
    committed = state['artifacts'][str(step)]['checkpoint']
    assert_same(torch.load(run_dir / 'selected-forward.pt', weights_only=False),
                torch.load(run_dir / committed['file'], weights_only=False))


def test_locked_run_directory_exits_without_training(workspace):
    config = write_config(workspace, 'transe', {})
    run_dir = workspace / 'run'
    run_dir.mkdir()
    with (run_dir / '.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert run(config, run_dir) == train.EXIT_LOCKED
    assert not (run_dir / 'resume.pt').exists()


def test_configuration_validation(workspace):
    with pytest.raises(ValueError, match='membership_mode'):
        train.resolve_config({'data_dir': 'x', 'method': 'geometre'}, workspace / 'c.json')
    with pytest.raises(ValueError, match='inapplicable'):
        train.resolve_config({'data_dir': 'x', 'method': 'transe', 'alpha': 0.2}, workspace / 'c.json')
    with pytest.raises(ValueError, match='inapplicable'):
        train.resolve_config({'data_dir': 'x', 'method': 'transe', 'score_test': True}, workspace / 'c.json')
    config, data_dir = train.resolve_config({'data_dir': 'prepared/fold0', 'method': 'q2b'}, workspace / 'c.json')
    assert data_dir == (workspace / 'prepared' / 'fold0').resolve()
    assert (config['dim'], config['seed'], config['steps'], config['eval_every'], config['batch_size'],
            config['negative_sample_size'], config['learning_rate'], config['validation_diseases']) == (
               100, 0, 500, 100, 128, 32, 1e-4, 32)
    assert (config['gamma'], config['cen'], config['offset_activation']) == (24.0, 0.02, 'none')


def test_pending_validation_schedule():
    curve = [{'step': step} for step in (0, 100, 200)]
    assert not train.pending_validation(curve, 250, 500, 100)
    assert train.pending_validation(curve, 300, 500, 100)
    with pytest.raises(ValueError):
        train.pending_validation(curve, 400, 500, 100)
    assert train.pending_validation(curve[:1] + [{'step': step} for step in (100, 200, 300, 400)], 450, 450, 100)


def test_profiles_fixture_keeps_duplicates():
    assert PROFILES['V0'].count('HP:1') == 2


def test_script_entry_points_and_sigterm_in_subprocess(workspace):
    code = Path(train.__file__).resolve().parent
    config = write_config(workspace, 'transe', {}, steps=400, eval_every=200)
    run_dir = workspace / 'script'
    command = [sys.executable, str(code / 'train.py'), '--config', str(config), '--run-dir', str(run_dir),
               '--device', 'cpu']
    assert subprocess.run(command + ['--stop-after', '1'], capture_output=True).returncode == train.EXIT_PAUSE
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log = run_dir / 'training.jsonl'
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and child.poll() is None:
        if log.exists() and len(log.read_text().splitlines()) > 5:
            break
        time.sleep(0.05)
    child.send_signal(signal.SIGTERM)
    assert child.wait(60) == train.EXIT_PAUSE
    segments = [json.loads(line) for line in (run_dir / 'segments.jsonl').read_text().splitlines()]
    assert segments[-1]['signal'] == 'SIGTERM'
    assert subprocess.run(command, capture_output=True).returncode == train.EXIT_COMPLETE
    result = subprocess.run([sys.executable, str(code / 'recompute.py'), '--run-dir', str(run_dir), '--data-dir',
                             str(workspace / 'fold0')], capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)['steps'] == [0, 200, 400]
