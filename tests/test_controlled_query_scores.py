import importlib.util
from pathlib import Path
import shutil

import pytest

from analysis.summarize_controlled_query import summarize
from phenoquery.common import read, sha, write
from reproduce.artifacts import fetch, pack
from test_controlled_query_report import make_campaign

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'reproducibility'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('build_controlled_query_scores')
tables = load('generate_controlled_query_tables')


def campaign(tmp_path):
    root = make_campaign(tmp_path)
    report = tmp_path/'controlled-query-report.json'
    write(report, summarize(root, expected_candidates=17))
    return root, report


def test_full_score_archive_replays_without_models_or_original_data(tmp_path):
    root, report = campaign(tmp_path)
    original_report = read(report)
    # Production campaigns link to a prepared data tree outside the run root.
    external = tmp_path/'prepared'
    shutil.move(root/'data', external)
    for name in read(root/'CONTROLLED.json')['campaigns']:
        link = root/name/'data'
        link.unlink(); link.symlink_to(external, target_is_directory=True)
    for weight in root.glob('*/training/*/*.pt'):
        weight.unlink()
    (root/'unlisted-private-file').write_text('must not enter an archive')
    plan = m.build(root, report, max_part_bytes=1_000_000, expected_candidates=17)
    expected = plan['provenance']['expected_files']
    assert len(expected) == 2961
    assert sum(p.endswith('/scores.npz') for p in expected) == 480
    assert sum(p.endswith('/ranks.npz') for p in expected) == 480
    assert sum(p.endswith('/validation.npz') for p in expected) == 240
    assert sum(p.endswith('/initial-validation.npz') for p in expected) == 240
    assert sum('/data/fold' in p for p in expected) == 300
    assert not any(p.endswith('.pt') or 'unlisted-private-file' in p for p in expected)
    assert len(plan['components']) > 8
    plan_path, archives = tmp_path/'plan.json', tmp_path/'archives'
    write(plan_path, plan)
    pack(plan_path, archives, archives.as_uri())
    assert m.verify_packed(plan_path, archives/'release-manifest.json')['files'] == 2961
    restored = tmp_path/'restored'
    fetch(archives/'release-manifest.json', restored, tmp_path/'downloads')
    # Remove every original input to make accidental fallback impossible.
    shutil.rmtree(root); shutil.rmtree(external); report.unlink()
    assert not any(p.is_symlink() for p in restored.rglob('*'))
    reproduced = tmp_path/'reproduced.json'
    write(reproduced, summarize(restored/'controlled-query', expected_candidates=17))
    assert read(reproduced) == original_report
    assert sha(reproduced) == plan['provenance']['report_sha256']
    reference = restored/plan['provenance']['reference_report']
    assert reference.read_bytes() == reproduced.read_bytes()
    tables.generate(read(reference), tmp_path/'reference-tables')
    tables.generate(read(reproduced), tmp_path/'reproduced-tables')
    assert len(list((tmp_path/'reference-tables').glob('*.tex'))) == 3
    for file in (tmp_path/'reference-tables').glob('*.tex'):
        assert file.read_bytes() == (tmp_path/'reproduced-tables'/file.name).read_bytes()


@pytest.mark.parametrize('failure', ['report', 'missing-score', 'changed-rank', 'changed-benchmark'])
def test_rejects_incomplete_or_changed_score_evidence(tmp_path, failure):
    root, report = campaign(tmp_path)
    folder = root/'ultra_4g-seed0/test/fold0/bridge_pool'
    if failure == 'report':
        result = read(report); result['associations'] += 1; write(report, result)
    elif failure == 'missing-score':
        (folder/'scores.npz').unlink()
    elif failure == 'changed-rank':
        (folder/'ranks.npz').write_bytes(b'changed')
    else:
        file = root/'data/fold0/profiles.json'
        profiles = read(file); profiles['test0a'] = ['changed']; write(file, profiles)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.build(root, report, expected_candidates=17)


def test_packed_verification_rejects_change_after_planning(tmp_path):
    root, report = campaign(tmp_path)
    plan = m.build(root, report, expected_candidates=17)
    plan_path, archives = tmp_path/'plan.json', tmp_path/'archives'
    write(plan_path, plan)
    (root/'ultra_4g-seed0/training/bridge_pool-fold0/progress.json').write_text('{}\n')
    pack(plan_path, archives, archives.as_uri())
    with pytest.raises(ValueError, match='Packed files differ'):
        m.verify_packed(plan_path, archives/'release-manifest.json')
