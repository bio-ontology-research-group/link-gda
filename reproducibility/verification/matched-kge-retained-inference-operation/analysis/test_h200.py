import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
spec = importlib.util.spec_from_file_location('h200_analysis', ROOT/'verify-and-analyze.py')
operation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(operation)


def public_files():
    m = json.loads((BASE/'inference/checkpoint-release-manifest.json').read_text())
    files = {n:r for c in m['components'] for n,r in c['files'].items()}
    files.update(json.loads((BASE/'inference/INPUTS_REFERENCE.json').read_text())['files'])
    return files


def test_real_public_members_cover_all_inputs_and_weights():
    files = public_files()
    sets = [operation.members_for_fold(files, fold) for fold in range(10)]
    assert set().union(*sets) == files.keys()
    assert sum(len(s) for s in sets) == 870


def started():
    plan = dict(inference_plan_sha256='a'*64, staging_proof_sha256='b'*64,
                inference_script_sha256='c'*64)
    execution = {}
    members = operation.members_for_fold(public_files(), 0)
    start = dict(fold=0, inference_plan_sha256='a'*64, staging_proof_sha256='b'*64,
        freeze_sha256=operation.FREEZE, operation_script_sha256='c'*64,
        training_source_commit=operation.TRAINING, no_test_predictions=True,
        member_hashes=members)
    return start, plan, execution, members


def test_bound_fresh_fold_start():
    s,p,e,m = started(); operation.verify_started(s,0,p,e,m)


@pytest.mark.parametrize('key', ['fold', 'inference_plan_sha256', 'staging_proof_sha256',
    'freeze_sha256', 'operation_script_sha256', 'training_source_commit',
    'no_test_predictions', 'member_hashes'])
def test_changed_start_identity_is_rejected(key):
    s,p,e,m = started(); s[key] = None
    with pytest.raises(ValueError): operation.verify_started(s,0,p,e,m)


@pytest.fixture
def captured_workspace(tmp_path):
    public = {n:dict(bytes=0,sha256=hashlib.sha256(b'').hexdigest()) for n in public_files()}
    retained = json.loads((BASE/'fixtures/native-score-metadata.json').read_text())['files']
    captured = dict(score_files_sha256={}, binary_files={}, metadata_files={})
    for n in public:
        p=tmp_path/'workspace'/n; p.parent.mkdir(parents=True,exist_ok=True); p.touch()
    for n,r in retained.items():
        if not (n.endswith('/SCORED.json') or '/scores-' in n): continue
        name='workspace/matched-kge/'+n; p=tmp_path/name
        payload=r['contents'].encode() if n.endswith('.json') else b'test prediction bytes'
        p.write_bytes(payload); digest=hashlib.sha256(payload).hexdigest()
        captured['score_files_sha256'][name]=digest
        record=dict(bytes=len(payload),sha256=digest)
        if n.endswith('.json'): captured['metadata_files'][name]=dict(record,text=payload.decode())
        else: captured['binary_files'][name]=record
    for fold in range(10):
        for graph in ('original','nobridge'):
            for family in operation.FAMILIES:
                name=f'workspace/matched-kge/fold{fold}/{graph}/{family}/run.lock'
                (tmp_path/name).touch()
                captured['metadata_files'][name]=dict(bytes=0,sha256=hashlib.sha256(b'').hexdigest(),text='')
    return tmp_path,captured,public


def test_closed_capture_and_all_bytes(captured_workspace):
    operation.verify_capture_files(*captured_workspace)


@pytest.mark.parametrize('kind', ['score_bytes','input_bytes','extra','missing','symlink',
    'scorer_hash','metadata_text','binary_size','missing_hash','public_path'])
def test_capture_corruption_is_rejected(captured_workspace,kind):
    root,capture,public = captured_workspace
    name=next(iter(capture['binary_files'])); score=root/name
    if kind=='score_bytes': score.write_bytes(b'wrong')
    if kind=='input_bytes': (root/'workspace'/next(iter(public))).write_bytes(b'wrong')
    if kind=='extra': (root/'workspace/extra').touch()
    if kind=='missing': score.unlink()
    if kind=='symlink': score.unlink(); score.symlink_to(root/'workspace/matched-kge/FROZEN.json')
    if kind=='scorer_hash': capture['score_files_sha256'][name]='a'*64
    if kind=='metadata_text': next(iter(capture['metadata_files'].values()))['text']='wrong'
    if kind=='binary_size': capture['binary_files'][name]['bytes']+=1
    if kind=='missing_hash': capture['score_files_sha256'].pop(name)
    if kind=='public_path': public['../unsafe']=public.pop(next(iter(public)))
    with pytest.raises(ValueError): operation.verify_capture_files(root,capture,public)


def test_original_scientific_analysis_body_unchanged():
    old=(BASE/'fixtures/prior-analysis.py').read_text()
    new=(ROOT/'verify-and-analyze.py').read_text()
    marker="    production = retain(REFERENCE/'ANALYSIS_COMPLETE.json'"
    old = old.replace('All ten public-checkpoint inference tasks completed;', 'Three retained-checkpoint allocations completed all ten inference folds;')
    assert old[old.index(marker):] == new[new.index(marker):]
