import importlib.util
from pathlib import Path

import pytest

from phenoquery.common import read, sha, write
from reproduce.artifacts import fetch, pack
from training.run_matched_kge import freeze, validate_completion
from test_matched_kge_manifest import fixture, m as manifest, rebind

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('matched_artifacts', ROOT / 'reproducibility/build_matched_kge_artifacts.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def campaign(tmp_path):
    root, data = fixture(tmp_path)
    # The metadata-only manifest fixture omits the native trainer's schema field.
    # Keep this archive/inference fixture identical to the trainer's freeze format.
    frozen = read(root / 'FROZEN.json'); frozen['schema'] = 1; write(root / 'FROZEN.json', frozen)
    for run in frozen['training']:
        p = root / run / 'SCORED.json'; scored = read(p)
        scored['freeze_sha256'] = sha(root / 'FROZEN.json'); write(p, scored)
    annotations = data.parent / 'gene_phenotypes.csv'
    score_manifest = tmp_path / 'scores.json'
    write(score_manifest, manifest.build(root, data, score_manifest, annotations))
    return root, data, score_manifest, annotations


def test_all_choices_roundtrip_and_relocated_native_freeze(tmp_path):
    args = campaign(tmp_path)
    root = args[0]
    extra = root / 'fold0/original/U1/resume'
    extra.mkdir(); (extra / 'optimizer.pt').write_bytes(b'not an inference dependency')
    plan = m.build(*args, max_part_bytes=200_000)
    expected = plan['provenance']['expected_files']
    assert len(expected) == 701
    assert sum(name.endswith('.pt') for name in expected) == 400
    assert not any('SCORED.json' in name or 'scores-' in name or '/resume/' in name for name in expected)
    assert len(plan['components']) > 1
    plan_file = tmp_path / 'plan.json'; write(plan_file, plan)
    archives = tmp_path / 'archives'
    pack(plan_file, archives, archives.as_uri())
    assert m.verify_packed(plan_file, archives / 'release-manifest.json')['files'] == 701
    installed = tmp_path / 'installed'
    fetch(archives / 'release-manifest.json', installed, tmp_path / 'downloads')
    restored = installed / 'matched-kge'
    frozen_hash = sha(restored / 'FROZEN.json')
    freeze(restored, list(range(10)))
    assert sha(restored / 'FROZEN.json') == frozen_hash
    for run in read(restored / 'FROZEN.json')['training']:
        trained = validate_completion(restored / run)
        manifest.verify_selections(restored / run, trained, read(restored / run / 'identity.json'))


@pytest.mark.parametrize('failure', ['changed-checkpoint', 'missing-choice', 'score-manifest', 'warm-start', 'freeze-schema'])
def test_rejects_inconsistent_checkpoint_collection(tmp_path, failure):
    args = campaign(tmp_path)
    root, _, score_manifest, _ = args
    folder = root / 'fold0/original/U1'
    if failure == 'changed-checkpoint': (folder / 'selected_raw_head.pt').write_bytes(b'changed')
    elif failure == 'missing-choice': (folder / 'selected_calibrated_mr.pt').unlink()
    elif failure == 'score-manifest':
        report = read(score_manifest); report['training_runs'] = 99; write(score_manifest, report)
    elif failure == 'freeze-schema':
        frozen = read(root / 'FROZEN.json'); frozen.pop('schema'); write(root / 'FROZEN.json', frozen)
    else:
        folder = root / 'fold0/original/CONVKB'
        identity = read(folder / 'identity.json'); identity['warm_start_sha256'] = 'wrong-parent'
        write(folder / 'identity.json', identity); rebind(root, folder)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.build(*args)


def test_archive_part_limits_reject_an_oversized_file(tmp_path):
    args = campaign(tmp_path)
    with pytest.raises(ValueError, match='Individual file exceeds'):
        m.build(*args, max_part_bytes=1)
