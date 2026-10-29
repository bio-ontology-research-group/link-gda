"""Operational scheduling must preserve single-writer training ownership."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('shared_scheduler', Path(__file__).resolve().parents[1] / 'scripts/run_matched_kge_shared.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


def test_exclusive_leases_and_inherited_child_ownership(tmp_path, monkeypatch):
    task = Path('fold0/original/U1')
    monkeypatch.setattr(scheduler, 'tasks', lambda: iter([task]))
    acquired, lease = scheduler.claim(tmp_path)
    assert acquired == task and scheduler.claim(tmp_path) is None
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                             stdin=subprocess.PIPE, pass_fds=(lease.fileno(),))
    lease.close()
    try:
        assert scheduler.claim(tmp_path) is None
    finally:
        child.communicate(timeout=10)
    acquired, lease = scheduler.claim(tmp_path)
    assert acquired == task
    lease.close()


def test_warm_start_dependency_completion_and_failure(tmp_path, monkeypatch):
    task = Path('fold0/original/CONVKB')
    monkeypatch.setattr(scheduler, 'tasks', lambda: iter([task]))
    assert scheduler.claim(tmp_path) is None
    parent = tmp_path / 'runs/fold0/original/U1/TRAINED.json'
    parent.parent.mkdir(parents=True)
    parent.write_text('{}')
    acquired, lease = scheduler.claim(tmp_path)
    assert acquired == task
    lease.close()
    failure = tmp_path / 'scheduling-leases/fold0-original-CONVKB.failure.json'
    failure.write_text('{}')
    with pytest.raises(RuntimeError, match='Unresolved training failure'):
        scheduler.claim(tmp_path)
    failure.unlink()
    marker = tmp_path / 'runs' / task / 'TRAINED.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{}')
    assert scheduler.claim(tmp_path) is None


@pytest.mark.parametrize('produce_marker', [True, False])
def test_real_children_require_completion_marker(tmp_path, monkeypatch, produce_marker):
    task_list = [Path('fold0/original/U1'), Path('fold1/original/U1')]
    monkeypatch.setattr(scheduler, 'tasks', lambda: iter(task_list))
    monkeypatch.setenv('SLURM_JOB_ID', 'integration')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.delenv('SLURM_RESTART_COUNT', raising=False)
    monkeypatch.setattr(scheduler.subprocess, 'check_output', lambda *a, **kw: 'GPU-test, 90, 1234, 140000\n')
    real_sleep = scheduler.time.sleep
    monkeypatch.setattr(scheduler.time, 'sleep', lambda _: real_sleep(.01))
    source = tmp_path / 'source'
    source.mkdir()
    args = SimpleNamespace(campaign_root=tmp_path, source_root=source, data_root=tmp_path, slots=2)

    def child_command(args, task):
        marker = args.campaign_root / 'runs' / task / 'TRAINED.json'
        code = 'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); '
        code += "p.write_text('{}')" if produce_marker else 'pass'
        return [sys.executable, '-c', code, str(marker)]

    monkeypatch.setattr(scheduler, 'command', child_command)
    out = tmp_path / 'shared-workers/integration-restart0'
    if produce_marker:
        scheduler.run(args)
        completion = json.loads((out / 'completion.json').read_text())
        assert set(completion['finished']) == set(map(str, task_list))
        assert completion['status'] == 'no_unclaimed_work'
        assert scheduler.claim(tmp_path) is None
    else:
        with pytest.raises(RuntimeError, match='without TRAINED'):
            scheduler.run(args)
        assert not (out / 'completion.json').exists()
        assert list((tmp_path / 'scheduling-leases').glob('*.failure.json'))


def test_command_preserves_scientific_budget_and_train_only(tmp_path):
    args = SimpleNamespace(campaign_root=tmp_path, source_root=tmp_path / 'frozen', data_root=tmp_path / 'data')
    command = scheduler.command(args, Path('fold9/nobridge/PPOOL'))
    options = dict(zip(command[3::2], command[4::2]))
    assert options['--phase'] == 'train'
    assert options['--epochs'] == '1000' and options['--profile-epochs'] == '20'
    assert options['--fold'] == '9' and options['--graph'] == 'nobridge'
    assert '--stop-after' not in command and '--device' not in command
