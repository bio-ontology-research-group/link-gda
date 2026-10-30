"""Exact frozen views survive deduplicated packaging, including executable aliases."""
import copy
import hashlib
import json
import os
import io
from pathlib import Path

import pytest

from reproduce import artifacts, curated_archive as curated


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def archive(tmp_path):
    manifests, retention = {}, []
    specifications = [
        ('matched-kge-scores-demo-v1', 'ibex', [
            ('first', {'matched-kge/fold0/original/T1/a': (b'A'*600, 0o644),
                       'matched-kge/fold0/original/U1/b': (b'B'*600, 0o644)}),
            ('second', {'matched-kge/fold0/nobridge/U1/c': (b'C'*600, 0o644),
                        'matched-kge/fold1/original/T1/d': (b'D'*600, 0o644),
                        'matched-kge/fold2/original/U1/plain': (b'print("ok")\n', 0o644),
                        'matched-kge/fold2/original/U1/script': (b'print("ok")\n', 0o755)})]),
        ('controlled-query-scores-demo-v1', 'ws', [
            ('third', {'query/alias': (b'A'*600, 0o644),
                       'query/data': (b'query-data', 0o644)})])]
    for tag, host, components in specifications:
        base = tmp_path/tag; base.mkdir()
        entries = []
        for name, files in components:
            sources = []
            for i, (destination, (data, mode)) in enumerate(files.items()):
                path = base/(name+str(i)); path.write_bytes(data); path.chmod(mode)
                sources.append(dict(source=str(path), destination=destination))
            entries.append(dict(name=name, sources=sources))
        plan = base/'plan.json'; plan.write_text(json.dumps(dict(description='Exact small frozen test data', components=entries)))
        packed = base/'packed'
        artifacts.pack(plan, packed, 'https://example.invalid/'+tag)
        raw = (packed/'release-manifest.json').read_bytes(); m = json.loads(raw)
        manifests[tag] = raw
        retention.append(dict(host=host, archives=[dict(release_tag=tag, name=c['archive'],
            retained_path=str(packed/c['archive']), size=c['bytes'], sha256=c['sha256'],
            manifest_sha256=digest(raw)) for c in m['components']]))
    plan = curated.build_plan(manifests, retention)
    plan['small_threshold'], plan['part_payload_bytes'] = 64, 1024
    outputs = {}
    for host in ('ibex', 'ws', 'local'):
        outputs[host] = tmp_path/('curated-'+host)
    return tmp_path, manifests, retention, plan, outputs


def build(archive, workers=4):
    root, manifests, retention, plan, outputs = archive
    receipts = [curated.pack_host(plan, host, path, workers) for host, path in outputs.items()]
    catalog = curated.seal(plan, receipts)
    bundle = root/'bundle'; bundle.mkdir()
    for part in catalog['parts']:
        source = next(p/part['relative_path'] for p in outputs.values() if (p/part['relative_path']).exists())
        target = bundle/part['relative_path']; target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    return catalog, bundle


def test_all_logical_members_preserved_with_fewer_physical_objects(archive):
    root, manifests, _, plan, _ = archive
    assert plan['logical_members'] == 8 and len(plan['objects']) == 6
    catalog, bundle = build(archive)
    cache = root/'cache'
    for i, tag in enumerate(manifests):
        target = root/('view'+str(i))
        result = curated.restore(catalog, [tag], target, cache, bundle)
        original = json.loads(manifests[tag])
        names = set()
        for component in original['components']:
            for name, record in component['files'].items():
                path = target/name
                assert artifacts.sha256(path) == record['sha256']
                assert path.stat().st_mode & 0o777 == record['mode']
                names.add(name)
            name = 'MANIFESTS/'+component['name']+'.json'; names.add(name)
            assert (target/name).read_bytes() == artifacts.json_bytes(dict(schema=1, component=component['name'], files=component['files']))
        names.add('.artifacts.lock')
        assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()} == names
        assert result['status'] == 'passed'
        assert curated.restore(catalog, [tag], target, cache, bundle) == result
    script = root/'view0/matched-kge/fold2/original/U1/script'
    plain = root/'view0/matched-kge/fold2/original/U1/plain'
    assert script.read_bytes() == plain.read_bytes() and script.stat().st_ino != plain.stat().st_ino
    assert (cache/'verified/objects'/digest(script.read_bytes())).stat().st_mode & 0o777 == 0o644


def test_parallel_packing_is_byte_deterministic(archive):
    root, _, _, plan, outputs = archive
    first = curated.pack_host(plan, 'ibex', outputs['ibex'], 1)
    second = curated.pack_host(plan, 'ibex', root/'second-build', 4)
    assert first == second


def test_combined_inference_view_shares_identical_paths_and_preserves_modes(archive):
    root, manifests, _, _, _ = archive
    catalog, bundle = build(archive)
    # A benchmark input can be included in both the input and checkpoint
    # collections. Combining them must retain one exact path, including mode.
    first, second = list(manifests)
    left = json.loads(catalog['original_manifests'][first]['text'])
    right = json.loads(catalog['original_manifests'][second]['text'])
    name, record = next(iter(left['components'][0]['files'].items()))
    right['components'][0]['files'][name] = record
    raw = artifacts.json_bytes(right)
    catalog['original_manifests'][second] = dict(text=raw.decode(), sha256=digest(raw))
    target = root/'combined'
    result = curated.restore(catalog, [first, second], target, root/'combined-cache', bundle)
    assert result['files'] == 8
    assert (target/name).read_bytes() == b'A'*600
    assert (target/name).stat().st_mode & 0o777 == 0o644
    assert (target/'query/alias').read_bytes() == (target/name).read_bytes()
    script = target/'matched-kge/fold2/original/U1/script'
    assert script.stat().st_mode & 0o777 == 0o755
    assert script.read_bytes() == b'print("ok")\n'


def test_staging_backpressure_waits_before_reading_payload(tmp_path, monkeypatch):
    body = io.BytesIO(b'payload')
    free = iter([0, 10000])
    monkeypatch.setattr(curated.shutil, 'disk_usage', lambda path: type('Usage', (), dict(free=next(free)))())
    waited = []
    def wait(seconds):
        assert body.tell() == 0
        waited.append(seconds)
    monkeypatch.setattr(curated.time, 'sleep', wait)
    writer = curated.PartWriter(tmp_path, 'test', 'part.tar.gz', reserve_bytes=50)
    writer.add(digest(b'payload'), 7, body)
    part = writer.close()
    assert waited == [5] and part['files']['objects/'+digest(b'payload')]['bytes'] == 7


def test_corrupt_source_archive_does_not_get_a_completion_receipt(archive):
    _, _, _, plan, outputs = archive
    source = Path(next(s['path'] for s in plan['sources'] if s['host'] == 'ibex'))
    data = bytearray(source.read_bytes()); data[4] ^= 1; source.write_bytes(data)
    with pytest.raises(AssertionError):
        curated.pack_host(plan, 'ibex', outputs['ibex'], 4)
    assert not (outputs['ibex']/'PACKED.json').exists()


def test_cannot_reuse_an_existing_build_directory(archive):
    _, _, _, plan, outputs = archive
    outputs['ibex'].mkdir(); (outputs['ibex']/'keep').write_text('original')
    with pytest.raises(AssertionError):
        curated.pack_host(plan, 'ibex', outputs['ibex'])
    assert (outputs['ibex']/'keep').read_text() == 'original'


@pytest.mark.parametrize('kind', ['traversal', 'mode', 'digest', 'size'])
def test_invalid_original_member_identity_rejected_before_pack(archive, kind):
    _, manifests, retention, _, _ = archive
    tag = next(iter(manifests)); m = json.loads(manifests[tag]); files = m['components'][0]['files']; name = next(iter(files))
    if kind == 'traversal': files['../escape'] = files.pop(name)
    if kind == 'mode': files[name]['mode'] = 0o4755
    if kind == 'digest': files[name]['sha256'] = '../../unsafe'
    if kind == 'size': files[name]['bytes'] = -1
    raw = artifacts.json_bytes(m); manifests[tag] = raw
    for p in retention:
        for r in p['archives']:
            if r['release_tag'] == tag: r['manifest_sha256'] = digest(raw)
    with pytest.raises((ValueError, AssertionError)):
        curated.build_plan(manifests, retention)


def test_seal_rejects_missing_or_duplicated_object_receipts(archive):
    root, _, _, plan, outputs = archive
    ibex = curated.pack_host(plan, 'ibex', outputs['ibex'])
    with pytest.raises(AssertionError): curated.seal(plan, [ibex])
    with pytest.raises(AssertionError): curated.seal(plan, [ibex, ibex])


@pytest.mark.parametrize('kind', ['unsafe_component', 'unsafe_digest', 'negative_part', 'size', 'mode'])
def test_malformed_public_catalog_cannot_write_workspace(archive, kind):
    root, manifests, _, _, _ = archive
    catalog, bundle = build(archive); tag = next(iter(manifests))
    m = json.loads(catalog['original_manifests'][tag]['text']); c = m['components'][0]; record = next(iter(c['files'].values()))
    if kind == 'unsafe_component': c['name'] = '../../outside'
    if kind == 'unsafe_digest': record['sha256'] = '../../outside'
    if kind == 'negative_part': catalog['objects'][record['sha256']]['part'] = -1
    if kind == 'size': record['bytes'] += 1
    if kind == 'mode': record['mode'] = 0o7777
    raw = artifacts.json_bytes(m); catalog['original_manifests'][tag] = dict(text=raw.decode(),sha256=digest(raw))
    with pytest.raises((AssertionError, ValueError, KeyError)):
        curated.restore(catalog, [tag], root/'view', root/'cache', bundle)
    assert not (root/'view').exists() and not (root/'outside').exists()


def test_bad_download_bytes_rejected_before_creating_view(archive):
    root, manifests, _, _, _ = archive
    catalog, bundle = build(archive)
    path = bundle/catalog['parts'][0]['relative_path']; path.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        curated.restore(catalog, list(manifests), root/'view', root/'cache', bundle)
    assert not (root/'view').exists()


@pytest.mark.parametrize('kind', ['existing_bytes', 'inner_symlink', 'existing_mode'])
def test_existing_user_files_are_preserved_on_rejection(archive, kind):
    root, manifests, _, _, _ = archive
    catalog, bundle = build(archive); tag = next(iter(manifests)); m = json.loads(manifests[tag]); name = next(iter(m['components'][0]['files']))
    target = root/'view'; p = target/name; p.parent.mkdir(parents=True)
    if kind == 'existing_bytes': p.write_bytes(b'keep original')
    elif kind == 'existing_mode': p.write_bytes(b'A'*600); p.chmod(0o600)
    else:
        outside = root/'outside'; outside.write_bytes(b'keep outside'); p.symlink_to(outside)
    before = p.read_bytes()
    with pytest.raises(AssertionError):
        curated.restore(catalog, [tag], target, root/'cache', bundle)
    assert p.read_bytes() == before
