"""Deduplicate frozen release bytes, then restore exact original collection views.

The build plan is private and identifies retained source archives. The sealed
catalog contains only original manifests, logical file identities and portable
part locations. No model arrays are re-encoded or statistical rows removed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce.artifacts import HashingReader, atomic_json, fetch, json_bytes, safe_name, sha256

SMALL = 1024 * 1024
PART_PAYLOAD = 4_000_000_000


def group_for(occurrences):
    tags = {x['collection'] for x in occurrences}
    names = sorted({x['member'] for x in occurrences})
    if any(t.startswith(('inputs-', 'matched-inputs-')) for t in tags):
        return 'prepared-inputs'
    if any(t.startswith('controlled-query-') for t in tags):
        return 'controlled-query'
    for name in names:
        parts = name.split('/')
        if name.startswith('matched-kge/'):
            for family in ('CONVKB', 'INDIGENA', 'PPOOL', 'T1', 'U1'):
                if family in parts:
                    suffix = '-' + ('nobridge' if 'nobridge' in parts else 'original') if family == 'INDIGENA' else ''
                    return 'matched-' + family.lower() + suffix
    if any(t.startswith('matched-kge-') for t in tags):
        return 'prepared-inputs'
    if any(t.startswith('baseline-outputs-') for t in tags):
        return 'native-baselines'
    if any(t.startswith('embedding-outputs-') for t in tags):
        return 'historical-embeddings'
    return 'historical-query-controls'


def build_plan(manifests, retention):
    """All inputs are metadata; validate exact retained/public archive identities."""
    sources, objects, paths = [], {}, {}
    retained = {(x['release_tag'], x['name']): dict(x, host=p['host'])
                for p in retention for x in p['archives']}
    original = {}
    for tag, raw in sorted(manifests.items()):
        safe_name(tag)
        m = json.loads(raw)
        assert m['schema'] == 1
        digest = hashlib.sha256(raw).hexdigest()
        original[tag] = dict(sha256=digest, text=raw.decode())
        seen = set()
        for c in m['components']:
            r = retained[(tag, c['archive'])]
            assert r['manifest_sha256'] == digest and r['size'] == c['bytes'] and r['sha256'] == c['sha256']
            assert not seen & set(c['files'])
            seen.update(c['files'])
            number = len(sources)
            sources.append(dict(number=number, collection=tag, host=r['host'], path=r['retained_path'], component=c))
            for name, record in c['files'].items():
                safe_name(name)
                assert record['mode'] in (0o644, 0o755)
                assert type(record['bytes']) is int and record['bytes'] >= 0
                assert re.fullmatch('[0-9a-f]{64}', record['sha256'])
                identity = {k: record[k] for k in ('sha256', 'bytes')}
                if name in paths:
                    assert paths[name] == identity, 'Conflicting logical file: ' + name
                paths[name] = identity
                obj = objects.setdefault(record['sha256'], dict(bytes=record['bytes'], occurrences=[]))
                assert obj['bytes'] == record['bytes']
                obj['occurrences'].append(dict(source=number, member=name, collection=tag, host=r['host']))
    assert len(sources) == len(retained)
    priority = {'ibex': 0, 'ws': 1, 'local': 2}
    for obj in objects.values():
        obj['group'] = group_for(obj['occurrences'])
        obj['owner'] = min(obj['occurrences'], key=lambda x: (priority[x['host']], x['source'], x['member']))
    small_bytes = sum(x['bytes'] for x in objects.values() if x['bytes'] <= SMALL)
    assert small_bytes < 512 * 1024**2, 'Keep small-object buffering bounded'
    estimates = {}
    for obj in objects.values():
        group = estimates.setdefault(obj['group'], dict(objects=0, raw_bytes=0, large_by_host={}, small_hosts=set(), largest=0))
        group['objects'] += 1
        group['raw_bytes'] += obj['bytes']
        owner = obj['owner']
        if obj['bytes'] <= SMALL:
            group['small_hosts'].add(owner['host'])
        else:
            group['large_by_host'][owner['host']] = group['large_by_host'].get(owner['host'], 0) + obj['bytes']
            group['largest'] = max(group['largest'], obj['bytes'])
    for group in estimates.values():
        room = PART_PAYLOAD - group['largest']
        assert room > 0
        group['parts'] = len(group['small_hosts']) + sum((n + room - 1)//room for n in group['large_by_host'].values())
        group['small_hosts'] = sorted(group['small_hosts'])
        assert group['parts'] <= 95, 'Leave room for Zenodo record metadata'
        assert group['raw_bytes'] < 49_800_000_000, 'Keep each meaningful record below the default quota'
    return dict(schema=1, kind='phenoquery-curation-plan', original_manifests=original,
                sources=sources, objects=objects, logical_paths=paths, small_threshold=SMALL,
                part_payload_bytes=PART_PAYLOAD, record_estimates=estimates,
                logical_members=sum(len(c['files']) for r in original.values()
                                    for c in json.loads(r['text'])['components']))


class PartWriter:
    def __init__(self, root, group, name, reserve_bytes=0):
        self.root, self.group, self.name = root, group, name
        self.path = root/group/name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        assert not self.path.exists() and not self.path.with_suffix('.partial').exists()
        self.raw = self.path.with_suffix('.partial').open('xb')
        self.compressed = gzip.GzipFile(fileobj=self.raw, filename='', mode='wb', mtime=0, compresslevel=6)
        self.tar = tarfile.open(fileobj=self.compressed, mode='w', format=tarfile.PAX_FORMAT)
        self.files = {}
        self.component = name.removesuffix('.tar.gz')
        self.reserve_bytes = reserve_bytes

    def add(self, digest, size, stream):
        waits = 0
        while self.reserve_bytes and shutil.disk_usage(self.root).free < self.reserve_bytes + size:
            if waits % 12 == 0:
                print('WAITING_FOR_VERIFIED_PART_RELAY', self.name, flush=True)
            time.sleep(5)
            waits += 1
        name = 'objects/' + digest
        assert name not in self.files
        info = tarfile.TarInfo(name)
        info.size, info.mode = size, 0o644
        reader = HashingReader(stream)
        self.tar.addfile(info, reader)
        assert reader.count == size and reader.hash.hexdigest() == digest
        self.files[name] = dict(bytes=size, sha256=digest, mode=0o644)

    def close(self):
        payload = json_bytes(dict(schema=1, component=self.component, files=self.files))
        info = tarfile.TarInfo('MANIFESTS/' + self.component + '.json')
        info.size, info.mode = len(payload), 0o644
        self.tar.addfile(info, io.BytesIO(payload))
        self.tar.close()
        self.compressed.close()
        self.raw.close()
        self.path.with_suffix('.partial').replace(self.path)
        result = dict(name=self.component, archive=self.name, group=self.group,
                      relative_path=self.group + '/' + self.name, bytes=self.path.stat().st_size,
                      sha256=sha256(self.path), files=self.files)
        atomic_json(self.path.with_suffix('.json'), result)
        return result


def pack_host(plan, host, output, workers=4, reserve_bytes=0):
    output = Path(output)
    assert not output.exists(), 'Use a fresh output directory; preserve failed builds'
    output.mkdir(parents=True)
    selected = {}
    for digest, obj in plan['objects'].items():
        owner = obj['owner']
        if owner['host'] == host:
            selected.setdefault(owner['source'], {})[owner['member']] = digest
    small, mutex = {}, threading.Lock()
    aborted = threading.Event()

    class GroupWriter:
        def __init__(self, group):
            self.group = group
            self.remaining = {}
            for number, members in selected.items():
                count = sum(plan['objects'][h]['group'] == group and plan['objects'][h]['bytes'] > plan['small_threshold']
                            for h in members.values())
                if count:
                    self.remaining[number] = count
            self.order = sorted(self.remaining)
            self.position, self.writer, self.payload = 0, None, 0
            self.parts = []
            self.condition = threading.Condition()

        def add(self, number, digest, size, stream):
            with self.condition:
                if aborted.is_set():
                    raise RuntimeError('Another source failed; preserve this incomplete build')
                while self.order[self.position] != number:
                    if aborted.is_set():
                        raise RuntimeError('Another source failed; preserve this incomplete build')
                    self.condition.wait(timeout=1)
                assert size <= plan['part_payload_bytes']
                if self.writer is not None and self.payload + size > plan['part_payload_bytes']:
                    self.parts.append(self.writer.close())
                    self.writer, self.payload = None, 0
                if self.writer is None:
                    self.writer = PartWriter(output, self.group, f'{self.group}-{host}-part{len(self.parts):03d}.tar.gz', reserve_bytes)
                self.writer.add(digest, size, stream)
                self.payload += size
                self.remaining[number] -= 1
                if self.remaining[number] == 0:
                    self.position += 1
                    self.condition.notify_all()

        def finish(self):
            assert self.position == len(self.order) and not any(self.remaining.values())
            if self.writer is not None:
                self.parts.append(self.writer.close())
            return self.parts

    large_groups = {plan['objects'][h]['group'] for members in selected.values() for h in members.values()
                    if plan['objects'][h]['bytes'] > plan['small_threshold']}
    writers = {group: GroupWriter(group) for group in large_groups}

    def process_source(number):
        source = plan['sources'][number]
        component = source['component']
        origin = Path(source['path']).resolve(strict=True)
        assert origin.is_file() and origin.stat().st_size == component['bytes']
        expected = component['files']
        embedded = 'MANIFESTS/' + component['name'] + '.json'
        found = set()
        with origin.open('rb') as raw:
            archive_reader = HashingReader(raw)
            with tarfile.open(fileobj=archive_reader, mode='r|gz') as tar:
                for member in tar:
                    safe_name(member.name)
                    assert member.isfile() and member.name not in found
                    found.add(member.name)
                    assert member.name in expected or member.name == embedded
                    if member.name == embedded:
                        assert member.size <= 16 * 1024**2
                        with tar.extractfile(member) as handle:
                            assert json.load(handle) == dict(schema=1, component=component['name'], files=expected)
                        continue
                    record = expected[member.name]
                    assert member.size == record['bytes']
                    if member.name not in selected[number]:
                        continue
                    digest = selected[number][member.name]
                    obj = plan['objects'][digest]
                    with tar.extractfile(member) as handle:
                        if obj['bytes'] <= plan['small_threshold']:
                            payload = handle.read()
                            assert len(payload) == obj['bytes'] and hashlib.sha256(payload).hexdigest() == digest
                            with mutex:
                                assert digest not in small
                                small[digest] = payload
                        else:
                            group = obj['group']
                            writers[group].add(number, digest, obj['bytes'], handle)
            for _ in iter(lambda: archive_reader.read(8 * 1024**2), b''):
                pass
            assert archive_reader.count == component['bytes'] and archive_reader.hash.hexdigest() == component['sha256']
        assert found == set(expected) | {embedded}
        print('CURATED_SOURCE', host, number, len(selected[number]), flush=True)

    def process(number):
        try:
            if aborted.is_set():
                raise RuntimeError('Another source failed; preserve this incomplete build')
            process_source(number)
        except BaseException:
            aborted.set()
            for writer in writers.values():
                with writer.condition:
                    writer.condition.notify_all()
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(process, sorted(selected)))
    parts = [part for group in sorted(writers) for part in writers[group].finish()]
    groups = sorted({plan['objects'][h]['group'] for h in small})
    for group in groups:
        writer = PartWriter(output, group, f'{group}-{host}-metadata.tar.gz', reserve_bytes)
        for digest in sorted(small):
            if plan['objects'][digest]['group'] == group:
                writer.add(digest, len(small[digest]), io.BytesIO(small[digest]))
        parts.append(writer.close())
    wanted = {h for h, obj in plan['objects'].items() if obj['owner']['host'] == host}
    actual = [n.removeprefix('objects/') for part in parts for n in part['files']]
    assert len(actual) == len(set(actual)) and set(actual) == wanted
    result = dict(schema=1, status='passed', host=host, parts=parts,
                  objects=len(wanted), bytes=sum(plan['objects'][h]['bytes'] for h in wanted))
    atomic_json(output/'PACKED.json', result)
    return result


def seal(plan, receipts):
    parts, objects = [], {}
    assert len({r['host'] for r in receipts}) == len(receipts)
    for receipt in receipts:
        assert receipt['status'] == 'passed'
        for part in receipt['parts']:
            index = len(parts)
            parts.append(part)
            for name, record in part['files'].items():
                digest = name.removeprefix('objects/')
                assert re.fullmatch('[0-9a-f]{64}', digest) and digest not in objects
                assert record == dict(sha256=digest, bytes=plan['objects'][digest]['bytes'], mode=0o644)
                objects[digest] = dict(bytes=record['bytes'], part=index, group=part['group'])
    assert set(objects) == set(plan['objects'])
    for group in {p['group'] for p in parts}:
        selected = [p for p in parts if p['group'] == group]
        assert len(selected) <= 95, 'Leave room for record metadata within Zenodo file limits'
        assert sum(p['bytes'] for p in selected) < 49_900_000_000, 'Keep records below default quota'
    return dict(schema=1, kind='phenoquery-frozen-archive', parts=parts, objects=objects,
                original_manifests=plan['original_manifests'], logical_members=plan['logical_members'])


def restore(catalog, collections, root, cache, bundle_root=None):
    assert catalog['kind'] == 'phenoquery-frozen-archive' and catalog['schema'] == 1
    root, cache = Path(root).resolve(), Path(cache).resolve()
    assert root != cache and root not in cache.parents and cache not in root.parents
    expected, embedded = {}, {}
    for tag in collections:
        original = catalog['original_manifests'][tag]
        assert hashlib.sha256(original['text'].encode()).hexdigest() == original['sha256']
        for component in json.loads(original['text'])['components']:
            assert safe_name(component['name']).name == component['name']
            for name, record in component['files'].items():
                safe_name(name)
                assert re.fullmatch('[0-9a-f]{64}', record['sha256'])
                assert record['mode'] in (0o644, 0o755)
                assert type(record['bytes']) is int and record['bytes'] >= 0
                assert name not in expected or expected[name] == record
                expected[name] = record
            name = 'MANIFESTS/' + component['name'] + '.json'
            safe_name(name)
            payload = json_bytes(dict(schema=1, component=component['name'], files=component['files']))
            assert name not in embedded or embedded[name] == payload
            embedded[name] = payload
    assert expected
    indices = set()
    for record in expected.values():
        obj = catalog['objects'][record['sha256']]
        assert obj['bytes'] == record['bytes']
        assert type(obj['part']) is int and obj['part'] in range(len(catalog['parts']))
        indices.add(obj['part'])
    components = []
    for index in sorted(indices):
        component = dict(catalog['parts'][index])
        if bundle_root is not None:
            safe_name(component['relative_path'])
            component['url'] = (Path(bundle_root).resolve()/component['relative_path']).as_uri()
        assert component.get('url'), 'Supply a local bundle root or finalized Zenodo URLs'
        components.append(component)
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache/'requested-parts.json'
    atomic_json(manifest_path, dict(schema=1, components=components))
    fetch(manifest_path, cache/'verified', cache/'downloads')
    verified = cache/'verified'
    # Verify every requested object before creating any workspace files.
    for record in expected.values():
        digest = record['sha256']
        path = verified/'objects'/digest
        assert path.stat().st_size == record['bytes'] and sha256(path) == digest
    root.mkdir(parents=True, exist_ok=True)
    allowed = set(expected) | set(embedded) | {'.artifacts.lock'}
    current = list(root.rglob('*'))
    assert not any(p.is_symlink() for p in current)
    assert {p.relative_to(root).as_posix() for p in current if p.is_file()} <= allowed
    for name, record in expected.items():
        target = root/name
        if target.exists():
            assert target.is_file() and target.stat().st_size == record['bytes'] and sha256(target) == record['sha256']
            assert target.stat().st_mode & 0o777 == record['mode']
    for name, payload in embedded.items():
        target = root/name
        if target.exists():
            assert target.read_bytes() == payload
    for name, record in expected.items():
        target = root/name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            source = verified/'objects'/record['sha256']
            if source.stat().st_dev == target.parent.stat().st_dev and record['mode'] == 0o644:
                os.link(source, target)
            else:
                shutil.copyfile(source, target)
                os.chmod(target, record['mode'])
        assert target.stat().st_size == record['bytes'] and sha256(target) == record['sha256']
        assert target.stat().st_mode & 0o777 == record['mode']
    for name, payload in embedded.items():
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (root/'.artifacts.lock').touch()
    assert (root/'.artifacts.lock').read_bytes() == b''
    assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == allowed
    return dict(status='passed', collections=collections, files=len(expected), embedded_manifests=len(embedded),
                logical_bytes=sum(r['bytes'] for r in expected.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build-plan')
    b.add_argument('--manifests-dir', type=Path, required=True)
    b.add_argument('--retention-root', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('pack')
    b.add_argument('--plan', type=Path, required=True)
    b.add_argument('--host', choices=['ibex', 'ws', 'local'], required=True)
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--workers', type=int, default=4)
    b.add_argument('--staging-reserve-gib', type=int, default=0)
    b = sub.add_parser('seal')
    b.add_argument('--plan', type=Path, required=True)
    b.add_argument('--receipts', type=Path, nargs='+', required=True)
    b.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('restore')
    b.add_argument('--catalog', type=Path, required=True)
    b.add_argument('--collections', nargs='+', required=True)
    b.add_argument('--root', type=Path, required=True)
    b.add_argument('--cache', type=Path, required=True)
    b.add_argument('--bundle-root', type=Path)
    b.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'build-plan':
        manifests = {f.parent.name: f.read_bytes() for f in args.manifests_dir.glob('*/*.json')}
        retention = [json.loads(f.read_text()) for f in args.retention_root.glob('*/retention-plan.json')]
        atomic_json(args.output, build_plan(manifests, retention))
    elif args.command == 'pack':
        assert args.workers in range(1, 9)
        assert args.staging_reserve_gib >= 0
        pack_host(json.loads(args.plan.read_text()), args.host, args.output, args.workers, args.staging_reserve_gib*1024**3)
    elif args.command == 'seal':
        atomic_json(args.output, seal(json.loads(args.plan.read_text()), [json.loads(f.read_text()) for f in args.receipts]))
    else:
        result = restore(json.loads(args.catalog.read_text()), args.collections, args.root, args.cache, args.bundle_root)
        atomic_json(args.receipt, result)


if __name__ == '__main__':
    main()
