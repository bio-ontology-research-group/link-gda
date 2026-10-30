"""Independently score one full matched fold from verified public checkpoints."""
import argparse
import fcntl
from importlib.metadata import version
import os
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, choices=range(10), required=True)
    fold = parser.parse_args().task
    import numpy as np
    import torch
    plan = read(ROOT/'inference-plan.json')
    assert plan['operation_script_sha256'] == sha(Path(__file__))
    assert plan['sbatch_sha256'] == sha(ROOT/'run-inference.sbatch')
    source, download = read(ROOT/'SOURCE_READY.json'), read(ROOT/'DOWNLOAD_COMPLETE.json')
    assert source['status'] == download['status'] == 'passed'
    assert plan['source_proof_sha256'] == download['source_proof_sha256'] == sha(ROOT/'SOURCE_READY.json')
    assert plan['download_proof_sha256'] == sha(ROOT/'DOWNLOAD_COMPLETE.json')
    assert download['download_plan_sha256'] == sha(ROOT/'download-plan.json')
    assert download['checkpoint_files'] == 400 and download['checkpoint_collection_files'] == 701
    assert download['no_predictions_before_inference'] is True
    assert source['commits'] == dict(source='68101ed08e3e271f5f2276724f8a5338b9cf32a2',
        **{'training-source':'063efc263446b4d119d3dbf4b97edb8709859f1e'})
    for folder in ('source', 'training-source'):
        assert subprocess.check_output(['git','-C',str(ROOT/folder),'rev-parse','HEAD'],text=True).strip() == source['commits'][folder]
        assert not subprocess.check_output(['git','-C',str(ROOT/folder),'status','--porcelain'],text=True)
    for folder,key in [('source','analysis_files'), ('training-source','training_files')]:
        for name,digest in source[key].items():
            assert sha(ROOT/folder/name) == digest, name
    sample = BASE/'release-workspace/interim-matched-artifact-check/validation-inference-v1/verification.json'
    assert sha(sample) == plan['validation_sample_proof_sha256']
    sample_proof = read(sample)
    assert sample_proof['status'] == 'passed' and sample_proof['checkpoint_files'] == 12
    assert sample_proof['all_validation_metrics_exact'] is True
    environment = BASE/'inference-environments-v1/matched-environment-verification.json'
    gpu_check = BASE/'inference-environments-v1/matched-gpu-54758758.json'
    assert sha(environment) == plan['environment_proof_sha256']
    assert sha(gpu_check) == plan['gpu_environment_proof_sha256']
    assert read(environment)['status'] == read(gpu_check)['status'] == 'passed'
    assert str(torch.__version__) == '2.10.0+cu128' and np.__version__ == '2.4.2'
    assert version('pykeen') == '1.11.1'
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (8,0)
    assert all(os.environ[name] == '8' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    torch.set_num_threads(8)
    runs = ROOT/'workspace/matched-kge'
    data = ROOT/'workspace/benchmark/data'
    annotations = runs/'inputs/gene_phenotypes.csv'
    frozen_digest = sha(runs/'FROZEN.json')
    assert frozen_digest == plan['freeze_sha256'] == download['freeze_sha256']
    assert sha(annotations) == '94829ab611a3f02c4c33d3686028d8eded187c256f852d4e2cecb2f548b4d0b6'
    frozen = read(runs/'FROZEN.json')
    assert set(frozen['training']) == {r for f in range(10) for r in runs_for_fold(f)}
    expected = download['files']
    own_files = {name:record for name,record in expected.items()
                 if name.startswith(f'matched-kge/fold{fold}/') or name.startswith(f'benchmark/data/fold{fold}/')}
    assert sum(name.endswith('.pt') for name in own_files) == 40
    assert sum(name.endswith('/TRAINED.json') for name in own_files) == 10
    for name,record in own_files.items():
        path = ROOT/'workspace'/name
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
    output = ROOT/'gpu-runs'/f'fold{fold}'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'allocation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity = dict(fold=fold, inference_plan_sha256=sha(ROOT/'inference-plan.json'),
        download_proof_sha256=sha(ROOT/'DOWNLOAD_COMPLETE.json'), freeze_sha256=frozen_digest,
        operation_script_sha256=sha(__file__), training_source_commit=source['commits']['training-source'])
    if (output/'STARTED.json').exists():
        assert read(output/'STARTED.json') == identity
    else:
        clean_start(runs, fold)
        write(output/'STARTED.json', identity)
    job, attempt = int(os.environ['SLURM_JOB_ID']), int(os.environ.get('SLURM_RESTART_COUNT','0'))
    fingerprint = output/f'attempt-{job}-{attempt}.json'
    assert not fingerprint.exists()
    runtime = dict(job_id=job, restart_count=attempt, fold=fold, started_unix=time.time(),
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
            scope='Independent original-scorer execution from public checkpoints; numerical and full bias-condition comparisons remain required.'))
    print('INFERENCE_FOLD_COMPLETE', fold, flush=True)


if __name__ == '__main__':
    main()
