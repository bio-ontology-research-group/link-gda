"""Assemble an empty-prediction workspace using retained native files, without bulk copies.

This login-node preparation checks metadata only. Each inference task must hash
every assigned member against the frozen manifest before and after scoring.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_name(name):
    p = PurePosixPath(name)
    if (not re.fullmatch(r'(matched-kge|benchmark)/[A-Za-z0-9_./-]+', name)
            or '..' in p.parts or str(p) != name or p.is_absolute()
            or '/scores-' in name or name.endswith('/SCORED.json')):
        raise ValueError('Unsafe or prediction member: ' + name)


def expected_members(manifest, reference):
    assert manifest['schema'] == 1 and reference['status'] == 'passed'
    checkpoint = manifest['provenance']['expected_files']
    assert len(checkpoint) == 701 and sum(n.endswith('.pt') for n in checkpoint) == 400
    assert len(reference['files']) == reference['file_count'] == 142
    assert not set(checkpoint) & set(reference['files'])
    component_files = {}
    for component in manifest['components']:
        assert not set(component_files) & set(component['files'])
        component_files.update({n: {k: r[k] for k in ('bytes', 'sha256')}
                                for n, r in component['files'].items()})
    assert component_files == checkpoint
    expected = {**checkpoint, **reference['files']}
    assert len(expected) == 843
    for name, record in expected.items():
        validate_name(name)
        assert re.fullmatch('[0-9a-f]{64}', record['sha256'])
        assert type(record['bytes']) is int and record['bytes'] >= 0
    return expected


def source_path(name, plan):
    validate_name(name)
    if name == 'matched-kge/inputs/PROVENANCE.json':
        return Path(plan['annotation_provenance_source'])
    if name == 'matched-kge/inputs/gene_phenotypes.csv':
        return Path(plan['annotation_source'])
    if name.startswith('benchmark/data/'):
        return Path(plan['data_root']) / name.removeprefix('benchmark/data/')
    if name.startswith('matched-kge/') and '/inputs/' not in name:
        return Path(plan['training_root']) / name.removeprefix('matched-kge/')
    raise ValueError('Unmapped member: ' + name)


def source_metadata(path, expected_bytes):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Symlink in retained source: ' + str(path))
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != expected_bytes:
        raise ValueError('Missing, nonregular or wrong-size retained source: ' + str(path))
    return dict(device=info.st_dev, inode=info.st_ino, bytes=info.st_size,
                mtime_ns=info.st_mtime_ns)


def assemble(workspace, expected, plan):
    workspace = Path(workspace)
    assert not workspace.exists() and not workspace.is_symlink()
    records = {}
    # Complete preflight before creating the destination or any hard links.
    for name, record in expected.items():
        source = source_path(name, plan)
        metadata = source_metadata(source, record['bytes'])
        assert metadata['device'] == workspace.parent.stat().st_dev, 'Refuse a bulk copy across filesystems'
        records[name] = dict(source=str(source), **metadata)
    workspace.mkdir()
    for name, record in expected.items():
        source = Path(records[name]['source'])
        identity = {k: records[name][k] for k in ('device', 'inode', 'bytes')}
        assert {k: v for k, v in source_metadata(source, record['bytes']).items() if k in identity} == identity
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target, follow_symlinks=False)
        # ORIX updates mtime when a hard link is created; it is not a content check.
        assert {k: v for k, v in source_metadata(target, record['bytes']).items() if k in identity} == identity
        assert source.samefile(target)
    actual = {p.relative_to(workspace).as_posix() for p in workspace.rglob('*') if p.is_file()}
    assert actual == set(expected)
    return records


def main():
    plan = read(ROOT/'staging-plan.json')
    assert plan['stage_script_sha256'] == sha(__file__)
    assert plan['source_proof_sha256'] == sha(ROOT/'SOURCE_READY.json')
    source = read(ROOT/'SOURCE_READY.json')
    assert source['status'] == 'passed'
    assert source['commits'] == {'source': '68101ed08e3e271f5f2276724f8a5338b9cf32a2',
                                  'training-source': '063efc263446b4d119d3dbf4b97edb8709859f1e'}
    assert plan['manifest_sha256'] == sha(ROOT/'checkpoint-release-manifest.json')
    assert plan['input_reference_sha256'] == sha(ROOT/'INPUTS_REFERENCE.json')
    assert not (ROOT/'STAGED_INPUTS.json').exists()
    manifest = read(ROOT/'checkpoint-release-manifest.json')
    reference = read(ROOT/'INPUTS_REFERENCE.json')
    assert manifest['provenance']['freeze_sha256'] == plan['freeze_sha256']
    assert reference['source_proof_sha256'] == plan['source_proof_sha256']
    expected = expected_members(manifest, reference)
    assert sha(Path(plan['training_root'])/'FROZEN.json') == plan['freeze_sha256']
    assert sha(Path(plan['annotation_provenance_source'])) == expected['matched-kge/inputs/PROVENANCE.json']['sha256']
    records = assemble(ROOT/'workspace', expected, plan)
    result = dict(status='passed', transport='native_hardlinks',
        verification='metadata_only_before_task_hashes', source_proof_sha256=plan['source_proof_sha256'],
        stage_script_sha256=sha(__file__), staging_plan_sha256=sha(ROOT/'staging-plan.json'),
        manifest_sha256=plan['manifest_sha256'], input_reference_sha256=plan['input_reference_sha256'],
        freeze_sha256=plan['freeze_sha256'], checkpoint_files=400, checkpoint_collection_files=701,
        file_count=843, files=expected, retained_sources=records, no_predictions_before_inference=True,
        scope='Fresh prediction-free workspace linked from retained native inputs and selected weights. This is metadata-only preparation, not a new public download or member-hash verification. Every one of the 843 frozen member hashes must be checked by the inference tasks before and after scoring, followed by independent CPU verification.')
    with (ROOT/'STAGED_INPUTS.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='passed',files=843,weights=400,transport=result['transport'],
                          staging_proof_sha256=sha(ROOT/'STAGED_INPUTS.json'))),flush=True)


if __name__ == '__main__':
    main()
