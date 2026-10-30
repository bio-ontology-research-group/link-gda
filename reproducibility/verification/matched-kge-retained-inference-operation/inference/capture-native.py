"""Capture completed matched H200 inference without reading large arrays on login."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
FAMILIES = ('U1', 'T1', 'INDIGENA', 'CONVKB', 'PPOOL')
CHOICES = ('raw_mr', 'raw_head', 'calibrated_mr', 'calibrated_head')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def native_tasks(text, array):
    if type(array) is not int or array <= 0:
        raise ValueError('Invalid native array')
    result = {}
    for line in text.strip().splitlines():
        name, raw, state, code = line.strip().split('|')
        if not re.fullmatch(str(array)+r'_[0-2]', name):
            raise ValueError('Unexpected native task')
        task = int(name.split('_')[1])
        if task in result or not raw.isdigit() or int(raw) <= 0:
            raise ValueError('Duplicate or invalid native task')
        if state != 'COMPLETED' or code != '0:0':
            raise ValueError('Every task must complete successfully')
        result[task] = int(raw)
    if set(result) != set(range(3)) or len(set(result.values())) != 3:
        raise ValueError('Require three distinct completed allocations for all ten folds')
    return {fold: result[fold % 3] for fold in range(10)}


def run_names(fold):
    return {f'fold{fold}/{g}/{m}' for g in ('original', 'nobridge') for m in FAMILIES}


def score_names():
    return {'scores-'+choice+'.npz' for choice in CHOICES}


def validate_records(complete, fold):
    if (complete['status'] != 'inference_completed' or complete['fold'] != fold
            or complete['training_runs'] != 10 or complete['score_files'] != 40):
        raise ValueError('Incomplete fold')
    records = complete['records']
    if len(records) != 10 or {r['run'] for r in records} != run_names(fold):
        raise ValueError('Incorrect or duplicated run inventory')
    for record in records:
        if set(record['files']) != score_names():
            raise ValueError('Missing or extra score selection')
        for digest in [record['scored_sha256'], *record['files'].values()]:
            if not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('Invalid scorer digest')
    return records


def workspace_inventory(root, public_files):
    expected = set(public_files)
    if len(expected) != 843:
        raise ValueError('Incomplete public input inventory')
    for fold in range(10):
        for run in run_names(fold):
            expected |= {'matched-kge/'+run+'/'+name for name in score_names() | {'SCORED.json', 'run.lock'}}
    paths = list(root.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Symlink in inference workspace')
    if {p.relative_to(root).as_posix() for p in paths if p.is_file()} != expected:
        raise ValueError('Inference workspace inventory differs')
    for fold in range(10):
        for run in run_names(fold):
            if (root/'matched-kge'/run/'run.lock').read_bytes() != b'':
                raise ValueError('Nonempty scorer lock')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--array', type=int, required=True)
    args = parser.parse_args()
    assert not (ROOT/'NATIVE_CAPTURE.json').exists()
    plan = read(ROOT/'inference-plan.json')
    assert plan['capture_script_sha256'] == sha(__file__)
    command = ['sacct', '--array', '-X', '-n', '-P', '-j', str(args.array),
               '--format=JobID%40,JobIDRaw%40,State,ExitCode']
    native = subprocess.check_output(command, text=True)
    jobs = native_tasks(native, args.array)
    metadata, binaries, scores = {}, {}, {}

    def capture(name):
        path = ROOT/name
        assert path.is_file() and not path.is_symlink(), name
        raw = path.read_bytes()
        metadata[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), text=raw.decode())
        return json.loads(raw) if path.suffix == '.json' else None

    for name in ('inference-plan.json', 'SOURCE_READY.json', 'INPUTS_REFERENCE.json',
                 'STAGED_INPUTS.json', 'staging-plan.json', 'checkpoint-release-manifest.json',
                 'stage-inputs.py', 'annotation-provenance.json',
                 'validation-verification.json', 'validation-native-completion.log',
                 'run-inference.py', 'run-inference.sbatch', 'capture-native.py'):
        capture(name)
    for key, name in [('operation_script_sha256', 'run-inference.py'),
                      ('sbatch_sha256', 'run-inference.sbatch'),
                      ('source_proof_sha256', 'SOURCE_READY.json'),
                      ('staging_proof_sha256', 'STAGED_INPUTS.json'),
                      ('validation_sample_proof_sha256', 'validation-verification.json'),
                      ('validation_native_sha256', 'validation-native-completion.log')]:
        assert plan[key] == metadata[name]['sha256'], name
    staged = read(ROOT/'STAGED_INPUTS.json')
    assert staged['status'] == 'passed' and staged['no_predictions_before_inference']
    assert staged['checkpoint_files'] == 400 and staged['checkpoint_collection_files'] == 701
    assert staged['staging_plan_sha256'] == sha(ROOT/'staging-plan.json')
    assert staged['source_proof_sha256'] == plan['source_proof_sha256']
    sample = read(ROOT/'validation-verification.json')
    workspace_inventory(ROOT/'workspace', staged['files'])
    for fold in range(10):
        prefix = f'gpu-runs/fold{fold}'
        complete = capture(prefix+'/COMPLETE.json')
        start = capture(prefix+'/STARTED.json')
        assert start['fold'] == fold and start['no_test_predictions']
        assert start['inference_plan_sha256'] == sha(ROOT/'inference-plan.json')
        assert start['staging_proof_sha256'] == plan['staging_proof_sha256']
        assert start['operation_script_sha256'] == plan['operation_script_sha256']
        assert start['freeze_sha256'] == plan['freeze_sha256'] == staged['freeze_sha256']
        own = {n:r['sha256'] for n,r in staged['files'].items()
               if n.startswith(f'matched-kge/fold{fold}/')
               or n.startswith(f'benchmark/data/fold{fold}/')
               or n in ('matched-kge/FROZEN.json', 'matched-kge/inputs/PROVENANCE.json',
                        'matched-kge/inputs/gene_phenotypes.csv')}
        assert len(own) == 87 and start['member_hashes'] == own
        assert start['training_source_commit'] == '063efc263446b4d119d3dbf4b97edb8709859f1e'
        assert complete['started_proof_sha256'] == sha(ROOT/prefix/'STARTED.json')
        name = complete['environment_file']
        assert re.fullmatch(r'attempt-[0-9]+-[0-9]+\.json', name)
        runtime = capture(prefix+'/'+name)
        assert metadata[prefix+'/'+name]['sha256'] == complete['environment_sha256']
        wanted = dict(job_id=jobs[fold], array_job_id=args.array, fold=fold,
                      allocated_cpus=8, cpu_threads=8, array_task_id=fold % 3, capability=[9, 0],
                      torch='2.10.0+cu128', numpy='2.4.2', pykeen='1.11.1',
                      started_proof_sha256=sha(ROOT/prefix/'STARTED.json'))
        assert all(runtime[k] == v for k, v in wanted.items())
        assert 'H200' in runtime['gpu']
        assert runtime['matmul_tf32'] == sample['runtime']['tf32_matmul']
        assert runtime['cudnn_tf32'] == sample['runtime']['tf32_cudnn']
        for path in (ROOT/prefix).glob('attempt-*.json'):
            attempt = capture(path.relative_to(ROOT).as_posix())
            assert attempt['job_id'] == jobs[fold] and attempt['array_job_id'] == args.array
        for record in validate_records(complete, fold):
            relative = 'workspace/matched-kge/'+record['run']
            scored = capture(relative+'/SCORED.json')
            assert metadata[relative+'/SCORED.json']['sha256'] == record['scored_sha256']
            assert scored['files'] == record['files']
            assert scored['freeze_sha256'] == plan['freeze_sha256']
            assert scored['training_sha256'] == sha(ROOT/relative/'TRAINED.json')
            assert capture(relative+'/run.lock') is None
            scores[relative+'/SCORED.json'] = record['scored_sha256']
            for name, digest in record['files'].items():
                path = ROOT/relative/name
                assert path.is_file() and not path.is_symlink() and path.stat().st_size > 0
                binaries[relative+'/'+name] = dict(sha256=digest, bytes=path.stat().st_size)
                scores[relative+'/'+name] = digest
    assert len(binaries) == 400 and len(scores) == 500
    for name, record in metadata.items():
        assert sha(ROOT/name) == record['sha256'], name
    assert native_tasks(subprocess.check_output(command, text=True), args.array) == jobs
    (ROOT/'native-completion.log').write_text(native)
    result = dict(status='native_completed', cluster='orix', inference_array=args.array,
        inference_plan_sha256=sha(ROOT/'inference-plan.json'), capture_script_sha256=sha(__file__),
        native_completion_sha256=sha(ROOT/'native-completion.log'),
        native_fold_job_ids={str(k):v for k,v in jobs.items()}, native_allocations=3, folds=10, captured_unix=time.time(),
        score_files_sha256=scores, metadata_files=metadata, binary_files=binaries,
        scope='Three completed native allocations cover all ten completed folds and all 400 declared score hashes. Large binary hashes and scientific equivalence require destination CPU verification.')
    temporary = ROOT/'NATIVE_CAPTURE.pending'
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(ROOT/'NATIVE_CAPTURE.json')
    print(json.dumps(dict(status=result['status'], array=args.array, metadata_files=len(metadata),
        binary_files=len(binaries), binary_bytes=sum(r['bytes'] for r in binaries.values()),
        capture_sha256=sha(ROOT/'NATIVE_CAPTURE.json'))), flush=True)


if __name__ == '__main__':
    main()
