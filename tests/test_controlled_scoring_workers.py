import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from phenoquery.common import read, sha, write
from test_controlled_query_artifacts import campaign

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('scoring_workers',ROOT/'scripts/run_controlled_query_scoring.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    root = campaign(tmp_path_factory.mktemp('scoring-worker-fixture'))
    prepared = root/'prepared'; prepared.mkdir()
    (prepared/'data').symlink_to(root/'data',target_is_directory=True)
    (prepared/'projection').mkdir()
    for child in root.glob('*-seed*'):
        (child/'projection').symlink_to(prepared/'projection',target_is_directory=True)
        shutil.rmtree(child/'test')
    coordinator=m.load_coordinator(ROOT)
    coordinator.plan(arguments(root))
    for initialization in coordinator.INITS:
        for seed in range(3):
            args=arguments(root,initialization,seed)
            child=coordinator.child_args(args,f'{initialization}-seed{seed}',initialization,seed)
            (child.root/'SELECTION.json').unlink()
            coordinator.eq_freeze(child.root,coordinator.production_configs(child))
    (root/'GLOBAL-SELECTION.json').unlink()
    coordinator.freeze_global(root)
    return root


def arguments(root, initialization='ultra_4g', seed=0):
    return SimpleNamespace(source_root=ROOT,root=root,prepared_root=root/'prepared',
        base=root,ultra_root=root,initialization=initialization,seed=seed,device='cuda',
        protocol=ROOT/'reproducibility/protocols/controlled-query-20261028.md',phase='score')


def copy_fixture(fixture,tmp_path):
    # Dereference links, then restore the operational shared-input layout.
    root=tmp_path/'campaign'; shutil.copytree(fixture,root)
    for child in root.glob('*-seed*'):
        for part in ('data','projection'):
            shutil.rmtree(child/part)
            (child/part).symlink_to(root/'prepared'/part,target_is_directory=True)
    return root


def test_all_worker_configs_equal_original_serial_coordinator(fixture,monkeypatch):
    args=arguments(fixture); c=m.load_coordinator(ROOT)
    digest=sha(fixture/'GLOBAL-SELECTION.json'); original=[]
    monkeypatch.setattr(c,'plan',lambda a:None)
    monkeypatch.setattr(c,'require_gates',lambda a:None)
    monkeypatch.setattr(c,'eq_freeze',lambda *a:None)
    monkeypatch.setattr(c,'freeze_global',lambda p:digest)
    monkeypatch.setattr(c,'invoke',lambda child,config:original.append((child.root,config)))
    c.run(args)
    workers=[]
    for initialization in c.INITS:
        for seed in range(3):
            for child,config in m.configs(arguments(fixture,initialization,seed),c,digest):
                workers.append((child.root,config))
    assert workers==original and len(workers)==60
    assert len({(str(root),config['fold']) for root,config in workers})==60
    assert all(config['score_initial'] and len(config['arms'])==4 for root,config in workers)


@pytest.mark.parametrize('worker_count',[1,2])
def test_first_worker_uses_real_coordinator_freeze_and_restart_reuses_it(fixture,tmp_path,monkeypatch,worker_count):
    root=copy_fixture(fixture,tmp_path); args=arguments(root)
    original=(root/'GLOBAL-SELECTION.json').read_bytes(); (root/'GLOBAL-SELECTION.json').unlink()
    real_run=subprocess.run; calls=[]
    def run(command,**kwargs):
        if command[1].endswith('run_controlled_query.py'):
            assert command[-1]=='freeze'; calls.append('freeze')
            return real_run(command,**kwargs)
        assert (root/'GLOBAL-SELECTION.json').read_bytes()==original
        config=read(Path(command[command.index('--config')+1]));calls.append(config['fold'])
        child=Path(command[command.index('--root')+1])
        records={arm+suffix:{} for arm in config['arms'] for suffix in ('','_frozen')}
        write(child/f"test/fold{config['fold']}/COMPLETE.json",dict(fold=config['fold'],
            arms=config['arms'],records=records,selection_sha256=config['selection_sha256']))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess,'run',run)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        list(pool.map(m.run,[arguments(root,seed=s) for s in range(worker_count)]))
    m.run(args)
    assert calls[0]=='freeze' and calls.count('freeze')==1
    assert Counter(calls[1:])==Counter({fold:worker_count+1 for fold in range(10)})
    assert (root/'GLOBAL-SELECTION.json').read_bytes()==original
    assert read(root/'scoring-workers/ultra_4g-seed0.json')['score_bundles']==80


@pytest.mark.parametrize('problem',['incomplete-training','changed-other-selection','changed-source',
    'changed-plan','missing-freeze-with-test','serial-lock','worker-lock'])
def test_invalid_or_concurrent_campaign_never_scores(fixture,tmp_path,monkeypatch,problem):
    root=copy_fixture(fixture,tmp_path); held=None
    if problem=='incomplete-training':
        (root/'GLOBAL-SELECTION.json').unlink()
        (root/'ultraquery-seed2/training/typed_and-fold9/COMPLETE.json').unlink()
    elif problem=='changed-other-selection':
        path=root/'ultraquery-seed2/SELECTION.json';path.write_text('{}')
    elif problem=='changed-source': (root/'ultraquery-seed2/source.sha256').write_text('changed')
    elif problem=='changed-plan':
        path=root/'CONTROLLED.json';value=read(path);value['expected_runs']=239;write(path,value)
    elif problem=='missing-freeze-with-test':
        (root/'GLOBAL-SELECTION.json').unlink()
        write(root/'ultraquery-seed2/test/fold0/typed_and/COMPLETE.json',{})
    else:
        path=root.parent/'scoring.lock' if problem=='serial-lock' else root/'scoring-workers/ultra_4g-seed0.lock'
        path.parent.mkdir(exist_ok=True);held=path.open('a');fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    real_run=subprocess.run
    def run(command,**kwargs):
        assert command[1].endswith('run_controlled_query.py'),'Inference started before its gates passed'
        return real_run(command,**kwargs)
    monkeypatch.setattr(m.subprocess,'run',run)
    try:
        with pytest.raises((ValueError,FileNotFoundError,BlockingIOError,subprocess.CalledProcessError)):
            m.run(arguments(root))
    finally:
        if held:held.close()
    assert not (root/'scoring-workers/ultra_4g-seed0.json').exists()


def test_score_child_inherits_campaign_config_and_serial_exclusion_locks(tmp_path):
    child=SimpleNamespace(root=tmp_path/'child');(child.root/'configs').mkdir(parents=True)
    worker_path,shared_path=tmp_path/'worker.lock',tmp_path/'shared.lock'
    config_path=child.root/'configs/score-fold0.lock'
    check="""import fcntl,os,sys
held=[os.path.realpath('/proc/self/fd/'+fd) for fd in os.listdir('/proc/self/fd')]
for path in sys.argv[1:]:
 assert os.path.realpath(path) in held, 'Lease descriptor was not inherited: '+path
 with open(path,'a') as stream:
  try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError: continue
  raise AssertionError('Parent lease was not held in the child: '+path)
"""
    coordinator=SimpleNamespace(immutable=write,command=lambda child,path:
        [sys.executable,'-c',check,str(worker_path),str(shared_path),str(config_path)])
    with worker_path.open('a') as worker,shared_path.open('a') as shared:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(shared,fcntl.LOCK_SH|fcntl.LOCK_NB)
        m.invoke_score(coordinator,child,dict(fold=0),worker,shared)


def test_change_to_another_campaign_freeze_stops_current_worker(fixture,tmp_path,monkeypatch):
    root=copy_fixture(fixture,tmp_path); calls=[]
    def score(coordinator,child,config,worker,shared):
        calls.append(config['fold'])
        (root/'ultraquery-seed2/SELECTION.json').write_text('{}')
    monkeypatch.setattr(m,'invoke_score',score)
    with pytest.raises(ValueError,match='selection changed'):
        m.run(arguments(root))
    assert calls==[0]
    assert not (root/'scoring-workers/ultra_4g-seed0.json').exists()


def test_successful_child_without_fold_completion_stops_worker(fixture,tmp_path,monkeypatch):
    root=copy_fixture(fixture,tmp_path)
    monkeypatch.setattr(m,'invoke_score',lambda *a:None)
    with pytest.raises(FileNotFoundError):m.run(arguments(root))
    assert not (root/'scoring-workers/ultra_4g-seed0.json').exists()
