"""Regenerate legacy per-association metrics from explicit checksum-bound inputs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_metrics import analyze, sha


REQUIRED = {'predictions', 'candidates', 'training'}
OPTIONAL = {'restricted_pool', 'reference_stats', 'excluded_diseases',
            'reference_candidates', 'reference_metadata'}


def relative_path(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name or name == '.':
        raise ValueError(f'Unsafe relative path: {name}')
    return path


def validate_manifest(manifest):
    if manifest.get('schema') != 1 or not manifest.get('entries'):
        raise ValueError('Expected a nonempty schema-1 metric manifest')
    outputs = set()
    for entry in manifest['entries']:
        target = relative_path(entry['output'])
        if len(target.parts) != 2 or target.suffix != '.json' or entry['output'] in outputs:
            raise ValueError('Invalid or duplicate metric output')
        outputs.add(entry['output'])
        inputs = entry['inputs']
        if not REQUIRED <= inputs.keys() or inputs.keys() - REQUIRED - OPTIONAL:
            raise ValueError('Missing or unknown metric inputs')
        reference = {'reference_stats', 'reference_candidates', 'reference_metadata'}
        if inputs.keys() & reference and not reference <= inputs.keys():
            raise ValueError('Fixed-reference inputs require both sidecars')
        for record in inputs.values():
            relative_path(record['path'])
            digest = record['sha256']
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('Invalid input checksum')
        if reference <= inputs.keys():
            stem = inputs['reference_stats']['path']
            if inputs['reference_candidates']['path'] != stem + '.candidates.txt' or inputs['reference_metadata']['path'] != stem + '.metadata.json':
                raise ValueError('Fixed-reference sidecar paths differ')


def replay(manifest_path, input_root, workspace, shard=0, shards=1, replace=False):
    if not 0 <= shard < shards:
        raise ValueError('Invalid shard')
    manifest_path, input_root, workspace = map(Path, (manifest_path, input_root, workspace))
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    entries = manifest['entries'][shard::shards]
    if not entries:
        raise ValueError('Shard contains no entries')
    outputs = [workspace / 'revised-metrics' / e['output'] for e in entries]
    marker = workspace / f'embedding-replay-shard-{shard}-of-{shards}.json'
    if not replace and (marker.exists() or any(p.exists() for p in outputs)):
        raise FileExistsError('Metric output exists; use --replace for deliberate replay')
    marker.unlink(missing_ok=True)
    checked = {}
    for entry, output in zip(entries, outputs):
        paths = {}
        for kind, record in entry['inputs'].items():
            path = (input_root / record['path']).resolve()
            if not path.is_relative_to(input_root.resolve()):
                raise ValueError('Input escaped artifact root')
            if path not in checked:
                checked[path] = sha(path)
            if checked[path] != record['sha256']:
                raise ValueError(f'Input checksum differs: {path}')
            paths[kind] = path
        result = analyze(paths['predictions'], paths['candidates'], paths.get('restricted_pool'),
                         paths.get('reference_stats'), paths.get('excluded_diseases'), paths['training'])
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_suffix('.json.pending')
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        pending.replace(output)
        print('Replayed', entry['output'], flush=True)
    record = dict(schema=1, manifest_sha256=sha(manifest_path), shard=shard, shards=shards,
                  implementation_sha256=sha(Path(__file__).with_name('revision_metrics.py')),
                  outputs={str(p.relative_to(workspace)):sha(p) for p in outputs})
    marker.write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    replay(args.manifest, args.input_root, args.workspace, args.shard, args.shards, args.replace)
