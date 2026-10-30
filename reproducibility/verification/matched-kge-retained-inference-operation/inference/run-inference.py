"""Independently score one full matched fold from verified retained checkpoints."""
import argparse
import fcntl
from importlib.metadata import version
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
TRAINING = ROOT/'training-source'
SOURCE = ROOT/'source'
sys.path.insert(0, str(TRAINING/'code'))
from phenoquery.common import read, sha, write
from training.run_matched_kge import MODELS, GRAPHS, PRIMARY, validate_completion

CHOICES = {'raw_mr', 'raw_head', 'calibrated_mr', 'calibrated_head'}


def runs_for_fold(fold):
    assert type(fold) is int and 0 <= fold < 10
    return [f'fold{fold}/{graph}/{model}' for graph in GRAPHS for model in MODELS]


def score_command(identity, root, data, annotations, source=TRAINING):
    settings = identity['settings']
    model = settings['model']
    assert model in MODELS and identity['graph'] in GRAPHS
    assert type(identity['fold']) is int and 0 <= identity['fold'] < 10
    assert settings['device'] == 'cuda' and settings['seed'] == 0
    assert settings['dimension'] == (400 if model == 'INDIGENA' else 200)
    assert settings['early_stopping'] is False and settings['primary'] == PRIMARY[model]
    fixed = dict(epochs=1000, profile_epochs=20, validation_frequency=20,
        checkpoint_frequency=20, batch_size=65536, sub_batch_size=8192, expected_candidates=13625)
    assert all(settings[name] == value for name,value in fixed.items())
    command = [sys.executable, '-u', str(source/'code/training/run_matched_kge.py'),
        '--phase', 'score', '--root', str(root), '--data-root', str(data),
        '--fold', str(identity['fold']), '--graph', identity['graph'], '--model', model,
        '--device', 'cuda', '--gene-phenotypes', str(annotations)]
    for name in ('seed', 'dimension', *fixed):
        command += ['--'+name.replace('_','-'), str(settings[name])]
    return command


def clean_start(root, fold):
    for relative in runs_for_fold(fold):
        folder = root/relative
        assert not (folder/'SCORED.json').exists(), relative
        assert not list(folder.glob('scores-*')), relative


def completed_run(root, relative, frozen_digest):
    folder = root/relative
    trained = validate_completion(folder)
    scored = read(folder/'SCORED.json')
    assert scored['freeze_sha256'] == frozen_digest
    assert scored['training_sha256'] == sha(folder/'TRAINED.json')
    assert scored['primary'] == trained['primary']
    assert set(scored['files']) == {'scores-'+choice+'.npz' for choice in CHOICES}
    for name,digest in scored['files'].items():
        path = folder/name
        assert path.is_file() and not path.is_symlink() and sha(path) == digest, str(path)
    return dict(run=relative, scored_sha256=sha(folder/'SCORED.json'), files=scored['files'])


def members_for_fold(files, fold):
    assert type(fold) is int and fold in range(10)
    assert len(files) == 843
    for name, record in files.items():
        path = PurePosixPath(name)
        assert re.fullmatch(r'(matched-kge|benchmark)/[A-Za-z0-9_./-]+', name)
        assert '..' not in path.parts and str(path) == name
        assert re.fullmatch('[0-9a-f]{64}', record['sha256'])
        assert type(record['bytes']) is int and record['bytes'] >= 0
        assert '/scores-' not in name and not name.endswith('/SCORED.json')
    shared = {'matched-kge/FROZEN.json', 'matched-kge/inputs/PROVENANCE.json',
              'matched-kge/inputs/gene_phenotypes.csv'}
    assert shared <= files.keys()
    selected = {name: record for name, record in files.items()
                if name.startswith(f'matched-kge/fold{fold}/')
                or name.startswith(f'benchmark/data/fold{fold}/') or name in shared}
    assert len(selected) == 87
    assert sum(n.startswith(f'matched-kge/fold{fold}/') and n.endswith('.pt') for n in selected) == 40
    assert sum(n.endswith('/TRAINED.json') for n in selected) == 10
    assert sum(n.startswith(f'benchmark/data/fold{fold}/') for n in selected) == 14
    return selected


def validate_sample(sample, source_digest):
    assert sample['status'] == 'passed' and sample['all_validation_metrics_exact']
    assert sample['checkpoint_files'] == 12 and sample['job_id'] == 355504
    assert sample['cpu_threads'] == sample['allocated_cpus'] == 8
    assert sample['source_proof_sha256'] == source_digest
    runs = ('fold0/original/U1', 'fold0/nobridge/U1', 'fold0/original/T1')
    assert len(sample['records']) == 12
    assert {(r['run'], r['choice']) for r in sample['records']} == {(r,c) for r in runs for c in CHOICES}
    assert all(r['metrics_exact'] and r['actual'] == r['expected'] for r in sample['records'])


def validate_runtime(runtime, sample):
    assert runtime['torch'] == sample['runtime']['torch'] == '2.10.0+cu128'
    assert runtime['numpy'] == sample['runtime']['numpy'] == '2.4.2'
    assert runtime['pykeen'] == sample['runtime']['pykeen'] == '1.11.1'
    assert runtime['capability'] == sample['runtime']['capability'] == [9, 0]
    assert 'H200' in runtime['gpu']
    assert runtime['threads'] == runtime['cpus'] == 8
    assert runtime['matmul_tf32'] == sample['runtime']['tf32_matmul']
    assert runtime['cudnn_tf32'] == sample['runtime']['tf32_cudnn']


def validate_staging(staged, plan, root):
    assert staged['status'] == 'passed' and staged['transport'] == 'native_hardlinks'
    assert staged['verification'] == 'metadata_only_before_task_hashes'
    assert staged['staging_plan_sha256'] == sha(root/'staging-plan.json')
    assert staged['stage_script_sha256'] == sha(root/'stage-inputs.py')
    assert staged['manifest_sha256'] == sha(root/'checkpoint-release-manifest.json')
    assert staged['input_reference_sha256'] == sha(root/'INPUTS_REFERENCE.json')
    assert staged['checkpoint_files'] == 400 and staged['checkpoint_collection_files'] == 701
    assert staged['file_count'] == len(staged['files']) == 843
    assert staged['no_predictions_before_inference'] is True
    assert staged['source_proof_sha256'] == plan['source_proof_sha256']
    manifest, reference = read(root/'checkpoint-release-manifest.json'), read(root/'INPUTS_REFERENCE.json')
    assert not set(manifest['provenance']['expected_files']) & set(reference['files'])
    assert staged['files'] == {**manifest['provenance']['expected_files'], **reference['files']}
    assert staged['freeze_sha256'] == manifest['provenance']['freeze_sha256']
    assert reference['source_proof_sha256'] == plan['source_proof_sha256']
    for fold in range(10):
        members_for_fold(staged['files'], fold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, choices=range(10), required=True)
    fold = parser.parse_args().task
    assert int(os.environ['SLURM_ARRAY_TASK_ID']) == fold % 3
    import numpy as np
    import torch
    plan = read(ROOT/'inference-plan.json')
    assert plan['operation_script_sha256'] == sha(Path(__file__))
    assert plan['sbatch_sha256'] == sha(ROOT/'run-inference.sbatch')
    source, staged = read(ROOT/'SOURCE_READY.json'), read(ROOT/'STAGED_INPUTS.json')
    assert source['status'] == staged['status'] == 'passed'
    assert plan['source_proof_sha256'] == staged['source_proof_sha256'] == sha(ROOT/'SOURCE_READY.json')
    assert plan['staging_proof_sha256'] == sha(ROOT/'STAGED_INPUTS.json')
    assert staged['staging_plan_sha256'] == sha(ROOT/'staging-plan.json')
    assert staged['checkpoint_files'] == 400 and staged['checkpoint_collection_files'] == 701
    assert staged['no_predictions_before_inference'] is True
    assert source['commits'] == dict(source='68101ed08e3e271f5f2276724f8a5338b9cf32a2',
        **{'training-source':'063efc263446b4d119d3dbf4b97edb8709859f1e'})
    for folder in ('source', 'training-source'):
        assert subprocess.check_output(['git','-C',str(ROOT/folder),'rev-parse','HEAD'],text=True).strip() == source['commits'][folder]
        assert not subprocess.check_output(['git','-C',str(ROOT/folder),'status','--porcelain'],text=True)
    for folder,key in [('source','analysis_files'), ('training-source','training_files')]:
        for name,digest in source[key].items():
            assert sha(ROOT/folder/name) == digest, name
    sample = ROOT/'validation-verification.json'
    assert sha(sample) == plan['validation_sample_proof_sha256'] == 'b83fb30e8e378babc448dd609803b40c446a006344c685ef56d665fa68fc2e95'
    sample_proof = read(sample)
    validate_sample(sample_proof, plan['source_proof_sha256'])
    assert sha(ROOT/'validation-native-completion.log') == plan['validation_native_sha256'] == '4389a78e02ad371b2048c3d5fb259f81a5f212b9a973b4f94c52656d1de7f48b'
    assert (ROOT/'validation-native-completion.log').read_text().strip() == '355504|COMPLETED|0:0|00:01:52'
    validate_staging(staged, plan, ROOT)
    assert torch.cuda.is_available()
    assert all(os.environ[name] == '8' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    torch.set_num_threads(8)
    validate_runtime(dict(torch=str(torch.__version__), numpy=np.__version__, pykeen=version('pykeen'),
        gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        threads=torch.get_num_threads(), cpus=int(os.environ['SLURM_CPUS_PER_TASK'])), sample_proof)
    runs = ROOT/'workspace/matched-kge'
    data = ROOT/'workspace/benchmark/data'
    annotations = runs/'inputs/gene_phenotypes.csv'
    frozen_digest = sha(runs/'FROZEN.json')
    assert frozen_digest == plan['freeze_sha256'] == staged['freeze_sha256']
    assert sha(annotations) == '94829ab611a3f02c4c33d3686028d8eded187c256f852d4e2cecb2f548b4d0b6'
    frozen = read(runs/'FROZEN.json')
    assert set(frozen['training']) == {r for f in range(10) for r in runs_for_fold(f)}
    expected = staged['files']
    own_files = members_for_fold(expected, fold)
    for name,record in own_files.items():
        path = ROOT/'workspace'/name
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
    output = ROOT/'gpu-runs'/f'fold{fold}'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'allocation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity = dict(fold=fold, inference_plan_sha256=sha(ROOT/'inference-plan.json'),
        staging_proof_sha256=sha(ROOT/'STAGED_INPUTS.json'), freeze_sha256=frozen_digest,
        operation_script_sha256=sha(__file__), training_source_commit=source['commits']['training-source'],
        no_test_predictions=True, member_hashes={n:r['sha256'] for n,r in own_files.items()})
    if (output/'STARTED.json').exists():
        assert read(output/'STARTED.json') == identity
    else:
        clean_start(runs, fold)
        write(output/'STARTED.json', identity)
    job, attempt = int(os.environ['SLURM_JOB_ID']), int(os.environ.get('SLURM_RESTART_COUNT','0'))
    fingerprint = output/f'attempt-{job}-{attempt}.json'
    assert not fingerprint.exists()
    runtime = dict(job_id=job, array_job_id=int(os.environ['SLURM_ARRAY_JOB_ID']),
        restart_count=attempt, fold=fold, array_task_id=int(os.environ['SLURM_ARRAY_TASK_ID']), started_unix=time.time(),
        allocated_cpus=int(os.environ['SLURM_CPUS_PER_TASK']), cpu_affinity=sorted(os.sched_getaffinity(0)),
        torch=str(torch.__version__), numpy=np.__version__, pykeen=version('pykeen'),
        gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cpu_threads=torch.get_num_threads(), started_proof_sha256=sha(output/'STARTED.json'))
    write(fingerprint, runtime)
    records = []
    for relative in runs_for_fold(fold):
        folder = runs/relative
        assert frozen['training'][relative] == sha(folder/'TRAINED.json')
        run_identity = read(folder/'identity.json')
        assert f"fold{run_identity['fold']}/{run_identity['graph']}/{run_identity['settings']['model']}" == relative
        command = score_command(run_identity, runs, data, annotations)
        subprocess.run(command, check=True)
        records.append(completed_run(runs, relative, frozen_digest))
        write(output/'progress.json', dict(completed_runs=len(records), expected_runs=10, records=records))
        print('INFERENCE_RUN_COMPLETE', relative, flush=True)
    assert sha(runs/'FROZEN.json') == frozen_digest
    for name,record in own_files.items():
        assert sha(ROOT/'workspace'/name) == record['sha256'], name
    for folder,key in [('source', 'analysis_files'), ('training-source', 'training_files')]:
        for name,digest in source[key].items(): assert sha(ROOT/folder/name) == digest, name
    assert len(records) == 10 and sum(len(r['files']) for r in records) == 40
    if (output/'COMPLETE.json').exists():
        previous = read(output/'COMPLETE.json')
        assert previous['status'] == 'inference_completed' and previous['records'] == records
        assert previous['started_proof_sha256'] == sha(output/'STARTED.json')
        assert Path(previous['environment_file']).name == previous['environment_file']
        assert previous['environment_sha256'] == sha(output/previous['environment_file'])
    else:
        write(output/'COMPLETE.json', dict(status='inference_completed', fold=fold,
            training_runs=10, score_files=40, records=records, started_proof_sha256=sha(output/'STARTED.json'),
            environment_file=fingerprint.name, environment_sha256=sha(fingerprint),
            elapsed_seconds=time.time()-runtime['started_unix'],
            scope='Independent original-scorer execution from retained checkpoints matching the frozen release identities; numerical and full bias-condition comparisons remain required.'))
    print('INFERENCE_FOLD_COMPLETE', fold, flush=True)


if __name__ == '__main__':
    main()
