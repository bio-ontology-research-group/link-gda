import io
import json
from pathlib import Path
import tarfile
import pytest
from reproduce.artifacts import pack, fetch, sha256, safe_name


def fixture(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'candidates.txt').write_text('g1\ng2\n')
    (source/'nested').mkdir()
    (source/'nested/bytes.bin').write_bytes(bytes(range(256)))
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'description': 'test fixture', 'components': [
        {'name': 'fold0', 'sources': [{'source': str(source), 'destination': 'benchmark/data/fold0'}]}]}))
    archive = tmp_path/'archive'
    report = pack(plan, archive, archive.as_uri())
    return plan, archive, report


def test_roundtrip_idempotence_and_deterministic_archive(tmp_path):
    plan, archive, report = fixture(tmp_path)
    second = tmp_path/'second'
    repeated = pack(plan, second, second.as_uri())
    assert report['components'][0]['sha256'] == repeated['components'][0]['sha256']
    assert pack(plan, archive, archive.as_uri()) == report
    root, cache = tmp_path/'installed', tmp_path/'cache'
    fetch(archive/'release-manifest.json', root, cache)
    assert (root/'benchmark/data/fold0/nested/bytes.bin').read_bytes() == bytes(range(256))
    fetch(archive/'release-manifest.json', root, cache)
    assert (root/'MANIFESTS/fold0.json').is_file()


def test_refuses_changed_inputs_and_existing_data(tmp_path):
    plan, archive, _ = fixture(tmp_path)
    root, cache = tmp_path/'installed', tmp_path/'cache'
    fetch(archive/'release-manifest.json', root, cache)
    target = root/'benchmark/data/fold0/candidates.txt'
    target.write_text('different user data')
    with pytest.raises(FileExistsError, match='different existing data'):
        fetch(archive/'release-manifest.json', root, cache)
    assert target.read_text() == 'different user data'
    (tmp_path/'source/candidates.txt').write_text('changed input')
    with pytest.raises(ValueError, match='source changed'):
        pack(plan, archive, archive.as_uri())


def test_rejects_corrupt_download_before_install(tmp_path):
    _, archive, report = fixture(tmp_path)
    (archive/report['components'][0]['archive']).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Download checksum'):
        fetch(archive/'release-manifest.json', tmp_path/'installed', tmp_path/'cache')
    assert not (tmp_path/'installed/benchmark').exists()


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../../escape', './a', 'a\\b'])
def test_unsafe_member_names(name):
    with pytest.raises(ValueError): safe_name(name)


def test_rejects_tar_links_even_with_matching_archive_hash(tmp_path):
    _, archive, report = fixture(tmp_path)
    component = report['components'][0]
    path = archive/component['archive']
    with tarfile.open(path, 'w:gz') as tar:
        member = tarfile.TarInfo('benchmark/data/fold0/candidates.txt')
        member.type, member.linkname = tarfile.SYMTYPE, '../../../../escape'
        tar.addfile(member)
    component.update(sha256=sha256(path), bytes=path.stat().st_size)
    (archive/'release-manifest.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='link, special file'):
        fetch(archive/'release-manifest.json', tmp_path/'installed', tmp_path/'cache')
    assert not (tmp_path/'installed/benchmark').exists()


def test_destination_symlink_cannot_escape_workspace(tmp_path):
    _, archive, _ = fixture(tmp_path)
    root, outside = tmp_path/'installed', tmp_path/'outside'
    root.mkdir(); outside.mkdir()
    (root/'benchmark').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        fetch(archive/'release-manifest.json', root, tmp_path/'cache')
    assert not list(outside.iterdir())


def test_duplicate_destinations_rejected(tmp_path):
    plan, archive, _ = fixture(tmp_path)
    contents = json.loads(plan.read_text())
    contents['components'].append(dict(contents['components'][0], name='other'))
    plan.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match='must not overwrite'):
        pack(plan, archive, archive.as_uri())
