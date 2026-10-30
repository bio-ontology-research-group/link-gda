"""Bind H200 validation to the complete audited campaign and public frozen source."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

RUNS = ('fold0/original/U1', 'fold0/nobridge/U1', 'fold0/original/T1')
CHOICES = {'raw_mr', 'raw_head', 'calibrated_mr', 'calibrated_head'}
FREEZE = 'c0044268b876ff8d0a7f03daef733c6d4908bf381c152ada5bee91d13779cee5'
AUDIT = '21914e0fbfd8256b85db9ee7b6afe664a9a7a5709ff25db96ed533a4f7eae27e'
SOURCE = '0dde1adf994141dd2c7b1d27e5edeeeeaec80438269dc9007cafa2eae686f1e7'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def validate_evidence(source, audit):
    assert source['status'] == audit['status'] == 'passed'
    assert source['commits'] == {'source': '68101ed08e3e271f5f2276724f8a5338b9cf32a2',
                                'training-source': '063efc263446b4d119d3dbf4b97edb8709859f1e'}
    assert audit['training_runs'] == 100 and audit['checkpoint_files'] == 400
    assert audit['files'] == 700 and audit['freeze_sha256'] == FREEZE
    assert audit['torch'] == '2.10.0+cu128' and audit['pykeen'] == '1.11.1'
    expected_source = {'code/' + n: h for n, h in audit['training_source_files'].items()}
    expected_source['reproducibility/protocols/matched-kge-20261029.md'] = audit['protocol_sha256']
    assert source['training_files'] == expected_source
    records = {r['run']: r for r in audit['records']}
    assert len(records) == len(audit['records']) == 100
    assert set(records) == {f'fold{f}/{g}/{m}' for f in range(10)
                           for g in ('original', 'nobridge')
                           for m in ('U1', 'T1', 'INDIGENA', 'CONVKB', 'PPOOL')}
    groups = {}
    for name in RUNS:
        record = records[name]
        files = record['file_sha256']
        assert set(files) == {'TRAINED.json', 'identity.json', 'validation.json'} | {
            'selected_' + choice + '.pt' for choice in CHOICES}
        assert all(re.fullmatch('[0-9a-f]{64}', h) for h in files.values())
        assert set(record['checkpoints']) == CHOICES
        for choice, checkpoint in record['checkpoints'].items():
            assert checkpoint['finite'] and checkpoint['strict_model_load']
            assert checkpoint['checkpoint_sha256'] == files['selected_' + choice + '.pt']
        groups[name] = {'relative': name, 'files': files}
    return groups


def verify(root):
    plan = read(root / 'plan.json')
    assert set(plan['files_sha256']) == {'context.py', 'replay.py', 'preflight.py', 'replay.sbatch'}
    for name, digest in plan['files_sha256'].items():
        assert sha(root / name) == digest, name
    assert plan['audit_sha256'] == sha(root / 'CHECKPOINTS_VERIFIED.json') == AUDIT
    public = Path(plan['source_root'])
    assert public == Path('/mnt/data/u/hohndor/matched-kge-20261029/public-full-replay-v1')
    assert plan['source_proof_sha256'] == sha(public / 'SOURCE_READY.json') == SOURCE
    assert plan['destination'] == '/mnt/data/u/hohndor/matched-kge-20261029/production-v1/runs'
    assert plan['freeze_sha256'] == sha(Path(plan['destination']) / 'FROZEN.json') == FREEZE
    source = read(public / 'SOURCE_READY.json')
    audit = read(root / 'CHECKPOINTS_VERIFIED.json')
    groups = validate_evidence(source, audit)
    for directory, key in [('source', 'analysis_files'), ('training-source', 'training_files')]:
        folder = public / directory
        assert subprocess.check_output(['git', '-C', str(folder), 'rev-parse', 'HEAD'],
                                       text=True).strip() == source['commits'][directory]
        assert not subprocess.check_output(['git', '-C', str(folder), 'status', '--porcelain'], text=True)
        for name, digest in source[key].items():
            assert sha(folder / name) == digest, name
    return plan, groups
