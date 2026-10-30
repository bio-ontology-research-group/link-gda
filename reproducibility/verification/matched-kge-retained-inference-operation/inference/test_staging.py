"""Test native assembly boundaries and binding to the real frozen inventory."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


stage = module('retained_stage', 'stage-inputs.py')
inference = module('retained_inference', 'run-inference.py')


def test_exact_real_inventory_preserves_every_public_member():
    members = stage.expected_members(stage.read(ROOT/'checkpoint-release-manifest.json'),
                                     stage.read(ROOT/'INPUTS_REFERENCE.json'))
    assert members == stage.read(ROOT/'test-member-inventory.json')
    groups = [inference.members_for_fold(members, fold) for fold in range(10)]
    assert set().union(*(g.keys() for g in groups)) == set(members)


@pytest.mark.parametrize('name', ['matched-kge/../bad', '/matched-kge/file',
                                'benchmark/data/fold0/../../outside',
                                'matched-kge/fold0/original/U1/SCORED.json',
                                'matched-kge/fold0/original/U1/scores-raw_head.npz',
                                'unmapped/file'])
def test_refuses_unsafe_or_prediction_members(name):
    with pytest.raises(ValueError):
        stage.source_path(name, {})


@pytest.fixture
def tiny(tmp_path):
    source = tmp_path/'native'; source.mkdir()
    plan = dict(training_root=str(source/'runs'), data_root=str(source/'data'),
                annotation_source=str(source/'gene.csv'), annotation_provenance_source=str(source/'provenance.json'))
    names = ['matched-kge/fold0/original/U1/selected_raw_head.pt',
             'benchmark/data/fold0/candidates.txt', 'matched-kge/inputs/PROVENANCE.json']
    members = {}
    for index, name in enumerate(names):
        p = stage.source_path(name, plan); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(('payload-'+str(index)).encode())
        members[name] = dict(bytes=p.stat().st_size, sha256=stage.sha(p))
    return tmp_path/'workspace', members, plan


def test_assembly_links_only_expected_inputs_and_preserves_sources(tiny):
    workspace, members, plan = tiny
    native_scores = Path(plan['training_root'])/'fold0/original/U1/SCORED.json'
    native_scores.write_text('never copy native predictions')
    records = stage.assemble(workspace, members, plan)
    assert set(records) == set(members)
    assert not (workspace/'matched-kge/fold0/original/U1/SCORED.json').exists()
    for name, record in members.items():
        source = stage.source_path(name, plan)
        assert (workspace/name).stat().st_ino == source.stat().st_ino
        assert stage.sha(workspace/name) == stage.sha(source) == record['sha256']


def test_orix_link_timestamp_change_does_not_change_content(tiny, monkeypatch):
    workspace, members, plan = tiny
    actual_link = stage.os.link
    def link_with_orix_timestamp(source, destination, **kwargs):
        actual_link(source, destination, **kwargs)
        stage.os.utime(source, ns=(1600000000000000000, 1600000000000000000))
    monkeypatch.setattr(stage.os, 'link', link_with_orix_timestamp)
    stage.assemble(workspace, members, plan)
    for name, record in members.items():
        assert stage.sha(workspace/name) == record['sha256']
        assert stage.source_path(name, plan).samefile(workspace/name)


@pytest.mark.parametrize('change', ['wrong_size', 'symlink', 'missing', 'existing_workspace'])
def test_preflight_rejects_invalid_source_or_destination(tiny, change):
    workspace, members, plan = tiny
    name = next(iter(members)); source = stage.source_path(name, plan)
    if change == 'wrong_size': source.write_bytes(b'bad')
    elif change == 'symlink':
        actual = source.with_suffix('.retained'); source.rename(actual); source.symlink_to(actual)
    elif change == 'missing': source.unlink()
    else: workspace.mkdir(); (workspace/'keep').write_text('preserved')
    with pytest.raises((ValueError, AssertionError, FileNotFoundError)):
        stage.assemble(workspace, members, plan)
    if change != 'existing_workspace': assert not workspace.exists()
    else: assert (workspace/'keep').read_text() == 'preserved'


@pytest.fixture
def staged(tmp_path):
    for name in ['stage-inputs.py', 'checkpoint-release-manifest.json', 'INPUTS_REFERENCE.json']:
        shutil.copyfile(ROOT/name, tmp_path/name)
    (tmp_path/'staging-plan.json').write_text('{}')
    reference = stage.read(ROOT/'INPUTS_REFERENCE.json')
    manifest = stage.read(ROOT/'checkpoint-release-manifest.json')
    plan = dict(source_proof_sha256=reference['source_proof_sha256'])
    proof = dict(status='passed', transport='native_hardlinks', verification='metadata_only_before_task_hashes',
                 staging_plan_sha256=stage.sha(tmp_path/'staging-plan.json'), stage_script_sha256=stage.sha(tmp_path/'stage-inputs.py'),
                 manifest_sha256=stage.sha(tmp_path/'checkpoint-release-manifest.json'),
                 input_reference_sha256=stage.sha(tmp_path/'INPUTS_REFERENCE.json'), checkpoint_files=400,
                 checkpoint_collection_files=701, file_count=843,
                 files=stage.expected_members(manifest, reference), no_predictions_before_inference=True,
                 source_proof_sha256=plan['source_proof_sha256'], freeze_sha256=manifest['provenance']['freeze_sha256'])
    return proof, plan, tmp_path


def test_accepts_bound_staging_evidence(staged):
    inference.validate_staging(*staged)


@pytest.mark.parametrize('change', ['member_hash', 'member_missing', 'source', 'freeze', 'predictions', 'transport', 'script'])
def test_inference_rejects_changed_staging_evidence(staged, change):
    proof, plan, root = staged
    if change == 'member_hash': proof['files']['matched-kge/FROZEN.json']['sha256'] = 'a'*64
    elif change == 'member_missing': proof['files'].pop('matched-kge/FROZEN.json')
    elif change == 'source': proof['source_proof_sha256'] = 'a'*64
    elif change == 'freeze': proof['freeze_sha256'] = 'a'*64
    elif change == 'predictions': proof['no_predictions_before_inference'] = False
    elif change == 'transport': proof['transport'] = 'unverified_copy'
    else: (root/'stage-inputs.py').write_text('changed')
    with pytest.raises(AssertionError):
        inference.validate_staging(proof, plan, root)
