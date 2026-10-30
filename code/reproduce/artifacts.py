"""Build and fetch checksum-bound scientific input/output release archives.

Packing uses an explicit local allowlist; no repository or home directory is
implicitly included. Fetching verifies both archive bytes and every member,
rejects links/path traversal, and never replaces a different existing file.
"""
import argparse
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''): h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name or name == '.':
        raise ValueError(f'Unsafe artifact path: {name!r}')
    return path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


class HashingReader:
    def __init__(self, handle):
        self.handle, self.hash, self.count = handle, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.handle.read(size)
        self.hash.update(data)
        self.count += len(data)
        return data


def inventory(component):
    files = {}
    for mapping in component['sources']:
        source = Path(mapping['source'])
        destination = safe_name(mapping['destination'])
        if source.is_dir():
            members = sorted(source.rglob('*'))
        elif source.is_file():
            members = [source]
        else:
            raise FileNotFoundError(source)
        for member in members:
            if member.is_dir(): continue
            if not member.is_file(): raise ValueError(f'Not a regular file: {member}')
            if member.is_symlink(): raise ValueError(f'Explicitly resolve source symlinks before packaging: {member}')
            relative = member.relative_to(source) if source.is_dir() else None
            name = str(destination / relative) if relative is not None else str(destination)
            safe_name(name)
            if name.startswith('MANIFESTS/') or name in files:
                raise ValueError(f'Duplicate/reserved archive destination: {name}')
            files[name] = member
    if not files: raise ValueError('An archive must contain at least one file')
    return files


def pack_component(component, output):
    name = component['name']
    if safe_name(name).name != name: raise ValueError('Component name must be a basename')
    archive = output/(name+'.tar.gz')
    sidecar = output/(name+'.json')
    if archive.exists() or sidecar.exists():
        raise FileExistsError(f'Refusing to replace an existing component: {name}')
    files = inventory(component)
    records = {}
    temporary = archive.with_suffix(archive.suffix+'.partial')
    try:
        with temporary.open('wb') as raw, gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0, compresslevel=6) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as tar:
                for relative, source in sorted(files.items()):
                    before = source.stat()
                    info = tarfile.TarInfo(relative)
                    info.size = before.st_size
                    info.mode = 0o755 if before.st_mode & 0o111 else 0o644
                    with source.open('rb') as handle:
                        reader = HashingReader(handle)
                        tar.addfile(info, reader)
                    after = source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or reader.count != info.size:
                        raise ValueError(f'Source changed while packing: {source}')
                    records[relative] = {'sha256': reader.hash.hexdigest(), 'bytes': info.size, 'mode': info.mode}
                embedded = json_bytes({'schema': 1, 'component': name, 'files': records})
                info = tarfile.TarInfo('MANIFESTS/'+name+'.json')
                info.size, info.mode = len(embedded), 0o644
                tar.addfile(info, io.BytesIO(embedded))
        if temporary.stat().st_size >= 2*1024**3:
            raise ValueError('Archive exceeds the 2 GiB release-asset limit; split the allowlist')
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    record = {'name': name, 'archive': archive.name, 'bytes': archive.stat().st_size,
              'sha256': sha256(archive), 'files': records,
              'description': component.get('description', '')}
    atomic_json(sidecar, record)
    return record


def pack(plan_path, output, url_base):
    plan = json.loads(Path(plan_path).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    destinations = set()
    names = [c['name'] for c in plan['components']]
    if len(names) != len(set(names)): raise ValueError('Duplicate component names')
    for component in plan['components']:
        paths = set(inventory(component))
        if paths & destinations: raise ValueError('Components must not overwrite one another')
        destinations.update(paths)
        sidecar = output/(component['name']+'.json')
        # Interrupted packing may resume only from a verified component matching
        # the current allowlist and current source bytes.
        if sidecar.exists():
            record = json.loads(sidecar.read_text())
            current = inventory(component)
            if set(record['files']) != set(current): raise ValueError('Packing allowlist changed')
            if any(sha256(source) != record['files'][name]['sha256'] for name,source in current.items()):
                raise ValueError('Packing source changed')
            if sha256(output/record['archive']) != record['sha256']: raise ValueError('Packed archive checksum mismatch')
        else:
            record = pack_component(component, output)
        record['url'] = url_base.rstrip('/')+'/'+record['archive']
        records.append(record)
        print('PACKED', component['name'], record['bytes'], record['sha256'], flush=True)
    result = {'schema': 1, 'description': plan['description'], 'provenance': plan.get('provenance', {}),
              'packer_sha256': sha256(__file__), 'components': records}
    atomic_json(output/'release-manifest.json', result)
    return result


def download(component, cache):
    name = component['archive']
    if safe_name(name).name != name: raise ValueError('Archive name must be a basename')
    destination = cache/name
    if destination.exists():
        if destination.stat().st_size != component['bytes'] or sha256(destination) != component['sha256']:
            raise ValueError(f'Cached archive checksum mismatch: {name}')
        return destination
    url = component['url']
    if urllib.parse.urlsplit(url).scheme not in ('https', 'file'):
        raise ValueError('Artifact downloads require HTTPS (or a local file URL)')
    temporary = destination.with_suffix(destination.suffix+'.partial')
    try:
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as out:
            shutil.copyfileobj(source, out, length=1048576)
        if temporary.stat().st_size != component['bytes'] or sha256(temporary) != component['sha256']:
            raise ValueError(f'Download checksum mismatch: {name}')
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def install(component, archive, root):
    if safe_name(component['name']).name != component['name']:
        raise ValueError('Component name must be a basename')
    expected = component['files']
    for name in expected: safe_name(name)
    embedded_name = 'MANIFESTS/'+component['name']+'.json'
    safe_name(embedded_name)
    with tempfile.TemporaryDirectory(prefix='.artifact-', dir=root) as staging:
        staging = Path(staging)
        found = set()
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                safe_name(member.name)
                if not member.isfile() or member.name in found:
                    raise ValueError('Archive contains a link, special file or duplicate member')
                if member.name not in expected and member.name != embedded_name:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                found.add(member.name)
                destination = staging/member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, destination.open('wb') as out:
                    shutil.copyfileobj(source, out, length=1048576)
                if member.name in expected:
                    record = expected[member.name]
                    if member.size != record['bytes'] or sha256(destination) != record['sha256']:
                        raise ValueError(f'Archive member checksum mismatch: {member.name}')
                    os.chmod(destination, record['mode'])
        if found != set(expected) | {embedded_name}: raise ValueError('Archive members are missing')
        embedded = json.loads((staging/embedded_name).read_text())
        if embedded != {'schema': 1, 'component': component['name'], 'files': expected}:
            raise ValueError('Embedded manifest differs from the published manifest')
        # Validate all destinations before adding any file to the workspace.
        for name,record in expected.items():
            target = root/name
            target.resolve().relative_to(root.resolve())
            if target.exists() and (not target.is_file() or sha256(target) != record['sha256']):
                raise FileExistsError(f'Refusing to replace different existing data: {name}')
        (root/embedded_name).resolve().relative_to(root.resolve())
        for name in expected:
            target = root/name
            if target.exists(): continue
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging/name).replace(target)
        (root/'MANIFESTS').mkdir(exist_ok=True)
        atomic_json(root/embedded_name, embedded)
    print('VERIFIED', component['name'], len(expected), 'files', flush=True)


def fetch(manifest_path, root, cache, components=None):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['schema'] != 1: raise ValueError('Unknown release manifest schema')
    root, cache = Path(root).resolve(), Path(cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    names = [c['name'] for c in manifest['components']]
    if len(names) != len(set(names)): raise ValueError('Duplicate components in manifest')
    requested = set(components or names)
    if requested - set(names): raise ValueError('Unknown component requested')
    with (root/'.artifacts.lock').open('w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for component in manifest['components']:
            if component['name'] in requested:
                archive = download(component, cache)
                install(component, archive, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack', help='Pack an explicit local source allowlist')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url-base', required=True)
    p = commands.add_parser('fetch', help='Download, verify and install immutable release components')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--components', nargs='+')
    args = parser.parse_args()
    if args.command == 'pack': pack(args.plan, args.output, args.url_base)
    else: fetch(args.manifest, args.root, args.cache, args.components)


if __name__ == '__main__': main()
