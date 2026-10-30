import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from inventory import RUNS, CHOICES, ROOT_METADATA, groups, inventory, relative, sha


def record(payload):
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), text=payload.decode())


def public_files():
    m=json.loads((HERE.parent/'inference/checkpoint-release-manifest.json').read_text())
    files={n:r for c in m['components'] for n,r in c['files'].items()}
    files.update(json.loads((HERE.parent/'inference/INPUTS_REFERENCE.json').read_text())['files'])
    return {n:dict(bytes=0,sha256=hashlib.sha256(b'').hexdigest()) for n in files}


@pytest.fixture
def capture():
    metadata={n:record(b'{}\n') for n in ROOT_METADATA}
    metadata['STAGED_INPUTS.json']=record(json.dumps(dict(files=public_files())).encode())
    metadata['inference-plan.json']=record(json.dumps(dict(capture_script_sha256=metadata['capture-native.py']['sha256'])).encode())
    binaries,scores={},{}
    for fold in range(10):
        for name in ('STARTED.json','COMPLETE.json',f'attempt-{355336+fold%3}-0.json'):
            metadata[f'gpu-runs/fold{fold}/{name}']=record(b'{}\n')
    for run in RUNS:
        prefix='workspace/matched-kge/'+run
        name=prefix+'/SCORED.json';metadata[name]=record(b'{}\n');scores[name]=metadata[name]['sha256']
        metadata[prefix+'/run.lock']=record(b'')
        for choice in CHOICES:
            name=prefix+'/scores-'+choice+'.npz'
            binaries[name]={k:v for k,v in record(b'array').items() if k!='text'}
            scores[name]=binaries[name]['sha256']
    return dict(status='native_completed',cluster='orix',inference_array=355335,native_allocations=3,folds=10,
        inference_plan_sha256=metadata['inference-plan.json']['sha256'],
        capture_script_sha256=metadata['capture-native.py']['sha256'],
        native_fold_job_ids={str(i):355336+i%3 for i in range(10)},metadata_files=metadata,
        binary_files=binaries,score_files_sha256=scores)


def test_complete_inventory(capture):
    assert len(inventory(capture)) == 600


@pytest.mark.parametrize('change', ['binary', 'metadata', 'score-hash', 'metadata-bytes',
                                  'extra-metadata', 'runtime', 'status', 'job', 'size'])
def test_rejects_incomplete_or_changed_capture(capture, change):
    if change == 'binary': capture['binary_files'].pop(next(iter(capture['binary_files'])))
    elif change == 'metadata': capture['metadata_files'].pop('SOURCE_READY.json')
    elif change == 'score-hash': capture['score_files_sha256'][next(iter(capture['score_files_sha256']))] = '0'*64
    elif change == 'metadata-bytes': capture['metadata_files']['SOURCE_READY.json']['text'] = 'changed'
    elif change == 'extra-metadata': capture['metadata_files']['../../escape'] = record(b'{}\n')
    elif change == 'runtime':
        capture['metadata_files'].pop('gpu-runs/fold0/attempt-355336-0.json')
    elif change == 'status': capture['status'] = 'running'
    elif change == 'job': capture['native_fold_job_ids']['0'] = capture['native_fold_job_ids']['1']
    elif change == 'size': capture['binary_files'][next(iter(capture['binary_files']))]['bytes'] = 0
    with pytest.raises(ValueError): inventory(capture)


@pytest.mark.parametrize('name', ['/absolute', '../up', 'a/../up', 'a//b', 'a/./b', 'x;false', 'a b'])
def test_rejects_unsafe_paths(name):
    with pytest.raises(ValueError): relative(name)


@pytest.fixture
def staged(tmp_path, monkeypatch, capture):
    operation = tmp_path/'operation'; operation.mkdir()
    destination = tmp_path/'destination'; destination.mkdir()
    for name in ('inventory.py', 'stage-metadata.py', 'relay.py'):
        shutil.copy2(HERE/name, operation/name)
    native = ''.join(f'355335_{i}|{355336+i}|COMPLETED|0:0\n' for i in range(3)).encode()
    capture['native_completion_sha256'] = hashlib.sha256(native).hexdigest()
    (operation/'native-completion.log').write_bytes(native)
    (operation/'NATIVE_CAPTURE.json').write_text(json.dumps(capture))
    p = dict(scripts={n: sha(operation/n) for n in ('inventory.py', 'stage-metadata.py', 'relay.py')},
        source='/source', destination=str(destination), operation_destination=str(operation),
        capture_sha256=sha(operation/'NATIVE_CAPTURE.json'), inference_array=capture['inference_array'],
        inference_plan_sha256=capture['inference_plan_sha256'])
    proof=dict(status='passed',files=843,model_checkpoints=400,no_test_predictions=True,
        staging_proof_sha256=capture['metadata_files']['STAGED_INPUTS.json']['sha256'])
    (destination/'RETAINED_INPUTS_STAGED.json').write_text(json.dumps(proof))
    p['checkpoint_stage_proof_sha256']=sha(destination/'RETAINED_INPUTS_STAGED.json')
    (operation/'plan.json').write_text(json.dumps(p))
    for name in public_files():
        target=destination/'workspace'/name;target.parent.mkdir(parents=True,exist_ok=True);target.touch()
    spec = importlib.util.spec_from_file_location('stage_metadata', HERE/'stage-metadata.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT = operation
    monkeypatch.setattr(sys, 'argv', ['stage-metadata.py'])
    return module, operation, destination, capture


def fill(staged):
    module, operation, destination, capture = staged
    module.main()
    for name in capture['binary_files']:
        (destination/name).write_bytes(b'array')


def test_stage_and_exact_transfer_verification_are_restartable(staged, monkeypatch):
    fill(staged)
    module, operation, destination, capture = staged
    monkeypatch.setattr(sys, 'argv', ['stage-metadata.py', '--verify-transfer'])
    module.main()
    before = (operation/'DESTINATION_READY.json').read_bytes()
    module.main()
    assert (operation/'DESTINATION_READY.json').read_bytes() == before


@pytest.mark.parametrize('change', ['missing', 'extra', 'metadata', 'size', 'symlink'])
def test_destination_rejects_corruption(staged, monkeypatch, change):
    fill(staged)
    module, operation, destination, capture = staged
    target = destination/next(iter(capture['binary_files']))
    if change == 'missing': target.unlink()
    elif change == 'extra': target.with_name('extra.bin').write_bytes(b'array')
    elif change == 'metadata': target.with_name('SCORED.json').write_bytes(b'changed')
    elif change == 'size': target.write_bytes(b'shorter')
    elif change == 'symlink':
        target.unlink(); target.symlink_to(destination/'RETAINED_INPUTS_STAGED.json')
    monkeypatch.setattr(sys, 'argv', ['stage-metadata.py', '--verify-transfer'])
    with pytest.raises((AssertionError, ValueError)): module.main()


def test_stage_rejects_preexisting_predictions(staged):
    module, operation, destination, capture = staged
    (destination/next(iter(capture['binary_files']))).touch()
    with pytest.raises(AssertionError): module.main()
    assert not (operation/'METADATA_STAGED.json').exists()


def test_stage_preserves_conflicting_metadata(staged):
    module, operation, destination, capture = staged
    (destination/'SOURCE_READY.json').write_bytes(b'previous')
    with pytest.raises(AssertionError): module.main()
    assert (destination/'SOURCE_READY.json').read_bytes() == b'previous'
    assert not (operation/'METADATA_STAGED.json').exists()


@pytest.mark.parametrize('change', ['missing', 'running', 'duplicate', 'identity'])
def test_stage_rejects_inconsistent_native_accounting(staged, change):
    module, operation, destination, capture = staged
    lines = (operation/'native-completion.log').read_text().splitlines()
    if change == 'missing': lines.pop()
    elif change == 'running': lines[0] = lines[0].replace('COMPLETED', 'RUNNING')
    elif change == 'duplicate': lines[1] = lines[0]
    elif change == 'identity': lines[0] = lines[0].replace('|355336|', '|999999|')
    (operation/'native-completion.log').write_text('\n'.join(lines)+'\n')
    capture['native_completion_sha256'] = sha(operation/'native-completion.log')
    (operation/'NATIVE_CAPTURE.json').write_text(json.dumps(capture))
    p = json.loads((operation/'plan.json').read_text())
    p['capture_sha256'] = sha(operation/'NATIVE_CAPTURE.json')
    (operation/'plan.json').write_text(json.dumps(p))
    with pytest.raises(ValueError): module.main()


def test_grouping_covers_only_new_arrays(capture):
    grouped=groups(capture)
    assert len(grouped)==100 and all(len(v)==4 for v in grouped.values())
    assert {folder+'/'+n for folder,names in grouped.items() for n in names}==set(capture['binary_files'])
