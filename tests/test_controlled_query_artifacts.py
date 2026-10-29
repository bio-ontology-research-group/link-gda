import importlib.util
from pathlib import Path

import pytest

from phenoquery.campaign import freeze_global, source_text
from phenoquery.common import read, sha, write
from reproduce.artifacts import fetch, pack
from test_controlled_query_report import make_campaign

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('query_artifacts', ROOT / 'reproducibility/build_controlled_query_artifacts.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def campaign(tmp_path):
    root = make_campaign(tmp_path)
    plan = read(root / 'CONTROLLED.json')
    for name in plan['campaigns']:
        child = root / name
        initialization, seed = name.rsplit('-seed', 1)
        (child / 'source.sha256').write_text(source_text(ROOT / 'code'))
        plan['source_sha256'] = sha(child / 'source.sha256')
        write(child / 'campaign.json', dict(seed=int(seed), initialization=initialization,
            kind='eq', steps=500, cohort='both', arms=plan['arms'],
            protocol_sha256=plan['protocol_sha256'], source_sha256=plan['source_sha256']))
        selected = read(child / 'SELECTION.json')
        for run in selected['runs']:
            folder = child / 'training' / run
            identity = read(folder / 'identity.json')
            identity['source_manifest_sha256'] = plan['source_sha256']
            identity['initial_checkpoint_sha256'] = 'synthetic-' + initialization
            identity['config'].update(phase='train', protocol_sha256=plan['protocol_sha256'])
            write(folder / 'identity.json', identity)
            write(child / 'configs' / (run + '.json'), identity['config'])
            (folder / 'training.jsonl').write_text(''.join('{"step": %d}\n' % step for step in range(1, 501)))
            trained = read(folder / 'COMPLETE.json')
            trained.update(source_manifest_sha256=plan['source_sha256'], identity_sha256=sha(folder / 'identity.json'))
            write(folder / 'COMPLETE.json', trained)
            selected['runs'][run]['completion_sha256'] = sha(folder / 'COMPLETE.json')
        write(child / 'SELECTION.json', selected)
    write(root / 'CONTROLLED.json', plan)
    (root / 'GLOBAL-SELECTION.json').unlink()
    freeze_global(root)
    for initialization in plan['initializations']:
        path = root / f'pilot-{initialization}/resume.pt'
        path.parent.mkdir()
        path.write_bytes(b'synthetic gate bytes; no actual GPU training')
        write(root / f'GATE-{initialization}.json', dict(steps=4,
            source_sha256=plan['source_sha256'],
            validation_boundary=dict(steps=4, restored_state_bitwise_equal=True,
                pending_validation_completed=True),
            artifacts={str(path.relative_to(root)): sha(path)}))
    return root


def test_pack_extract_and_relocated_checkpoint_inventory(tmp_path):
    root = campaign(tmp_path)
    (root / 'ultra_4g-seed0/training/bridge_pool-fold0/resume.pt').write_bytes(b'not an inference input')
    plan = m.build(root, max_part_bytes=400_000)
    expected = plan['provenance']['expected_files']
    assert sum(p.endswith('/best.pt') for p in expected) == 240
    assert sum(p.endswith('/initial.pt') for p in expected) == 240
    assert sum(p.endswith('/training.jsonl') for p in expected) == 240
    assert not any('/test/' in p or '/training/' in p and p.endswith('/resume.pt') for p in expected)
    assert len(plan['components']) > 1
    plan_file = tmp_path / 'plan.json'
    write(plan_file, plan)
    archive = tmp_path / 'archive'
    pack(plan_file, archive, archive.as_uri())
    checked = m.verify_packed(plan_file, archive / 'release-manifest.json')
    assert checked['status'] == 'passed' and checked['files'] == len(expected)
    installed = tmp_path / 'installed'
    fetch(archive / 'release-manifest.json', installed, tmp_path / 'cache')
    restored = m.build(installed / 'controlled-query', max_part_bytes=400_000)
    assert restored['provenance'] == plan['provenance']
    assert [c['name'] for c in restored['components']] == [c['name'] for c in plan['components']]


@pytest.mark.parametrize('failure', ['no-freeze', 'changed-checkpoint', 'missing-initial',
    'incomplete-validation', 'incomplete-updates', 'config', 'source', 'gate'])
def test_rejects_incomplete_or_changed_checkpoint_evidence(tmp_path, failure):
    root = campaign(tmp_path)
    child = root / 'ultra_4g-seed0'
    run = child / 'training/bridge_pool-fold0'
    if failure == 'no-freeze': (root / 'GLOBAL-SELECTION.json').unlink()
    elif failure == 'changed-checkpoint': (run / 'best.pt').write_bytes(b'corrupt')
    elif failure == 'missing-initial': (run / 'initial.pt').unlink()
    elif failure == 'incomplete-validation':
        progress = read(run / 'progress.json'); progress['curve'].pop(); write(run / 'progress.json', progress)
    elif failure == 'incomplete-updates': (run / 'training.jsonl').write_text('{"step": 500}\n')
    elif failure == 'config': write(child / 'configs/bridge_pool-fold0.json', {'arm': 'typed_and'})
    elif failure == 'source': (child / 'source.sha256').write_text('different source')
    elif failure == 'gate': (root / 'pilot-ultra_4g/resume.pt').write_bytes(b'changed gate')
    with pytest.raises((ValueError, FileNotFoundError)):
        m.build(root)


def test_detects_source_change_between_plan_and_pack(tmp_path):
    root = campaign(tmp_path)
    plan_file = tmp_path / 'plan.json'
    write(plan_file, m.build(root))
    (root / 'ultra_4g-seed0/training/bridge_pool-fold0/best.pt').write_bytes(b'changed after planning')
    archive = tmp_path / 'archive'
    pack(plan_file, archive, archive.as_uri())
    with pytest.raises(ValueError, match='Packed files differ'):
        m.verify_packed(plan_file, archive / 'release-manifest.json')
