"""Explicit checkpoint inventories shared by the experiment-specific exporters."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from phenoquery.common import read, sha
from reproduce.artifacts import safe_name

MAX_PART_BYTES = 1400 * 1024**2


class CheckpointInventory:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}

    def include(self, name, expected=None):
        safe_name(name)
        path = self.root / name
        path.resolve(strict=True).relative_to(self.root)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Expected a regular archive source: {name}')
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Checkpoint/evidence checksum mismatch: {name}')
        record = {'sha256': digest, 'bytes': path.stat().st_size}
        if name in self.files and self.files[name] != record:
            raise ValueError(f'Archive source changed during inventory: {name}')
        self.files[name] = record
        return path

    def components(self, archive_prefix, destination_prefix, max_part_bytes=MAX_PART_BYTES):
        if max_part_bytes <= 0 or max_part_bytes > MAX_PART_BYTES:
            raise ValueError('Archive part budget must be positive and at most 1,400 MiB')
        components, batch, size = [], [], 0

        def flush():
            if batch:
                components.append(dict(name=f'{archive_prefix}-part{len(components):03}',
                    description='Frozen model weights and validation selection evidence',
                    sources=list(batch)))
                batch.clear()

        for name, record in sorted(self.files.items()):
            if record['bytes'] > max_part_bytes:
                raise ValueError(f'Individual file exceeds archive part budget: {name}')
            if batch and size + record['bytes'] > max_part_bytes:
                flush()
                size = 0
            batch.append(dict(source=str(self.root / name), destination=f'{destination_prefix}/{name}'))
            size += record['bytes']
        flush()
        return components

    def expected_files(self, destination_prefix):
        return {f'{destination_prefix}/{name}': value for name, value in sorted(self.files.items())}


def verify_packed(plan_path, manifest_path):
    plan, manifest = read(plan_path), read(manifest_path)
    if manifest['provenance'] != plan['provenance'] or manifest['description'] != plan['description']:
        raise ValueError('Packed release differs from the verified plan')
    if [c['name'] for c in manifest['components']] != [c['name'] for c in plan['components']]:
        raise ValueError('Packed release has missing or unexpected components')
    found = {}
    for component in manifest['components']:
        for name, record in component['files'].items():
            if name in found:
                raise ValueError('Duplicate member across archive components')
            found[name] = {key: record[key] for key in ('sha256', 'bytes')}
    if found != plan['provenance']['expected_files']:
        raise ValueError('Packed files differ from the verified checkpoint inventory')
    return {'status': 'passed', 'files': len(found), 'archives': len(manifest['components']),
            'plan_sha256': sha(plan_path), 'release_manifest_sha256': sha(manifest_path),
            'scope': 'Packed member inventory matches the previously verified frozen checkpoint files; archive extraction is a separate verification'}
