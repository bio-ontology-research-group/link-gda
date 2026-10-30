"""Rehash the retained A100 validation outputs in an allocated CPU step."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent/'release-workspace/interim-matched-artifact-check'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if not __debug__:
        raise RuntimeError('Verification requires Python assertion checks')
    assert os.environ['SLURM_JOB_ID'] == '54825093'
    assert os.environ['SLURM_CPUS_PER_TASK'] == '1'
    read = lambda p: json.loads(p.read_text())
    plan = read(ROOT/'hash-plan.json')
    assert sha(Path(__file__)) == plan['script_sha256']
    assert sha(WORK/'validation-inference-v1/verification.json') == plan['validation_proof_sha256']
    assert sha(WORK/'verification.json') == plan['artifact_proof_sha256']
    assert sha(WORK/'replay.py') == plan['source_replay_sha256']
    assert sha(WORK/'verify.py') == plan['source_context_sha256']
    assert sha(WORK/'plan.json') == plan['source_plan_sha256']
    runs = ('fold0/original/U1', 'fold0/nobridge/U1', 'fold0/original/T1')
    choices = ('raw_mr', 'raw_head', 'calibrated_mr', 'calibrated_head')
    assert set(plan['records']) == {r+'/'+c+'.npz' for r in runs for c in choices}
    accounting = subprocess.check_output(['sacct','-X','-j','54764788','-n','-P',
        '--format=JobID,State,ExitCode,AllocCPUS'], text=True).strip()
    assert accounting == '54764788|COMPLETED|0:0|8'
    sizes = {}
    for name, digest in plan['records'].items():
        path = WORK/'validation-inference-v1'/name
        assert path.is_file() and not path.is_symlink() and sha(path) == digest, name
        sizes[name] = path.stat().st_size
    proof = dict(status='passed', native_inference_job=54764788,
        native_verifier_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        inference_native_state='COMPLETED|0:0', allocated_inference_cpus=8,
        current_score_files_verified=12, score_bytes=sum(sizes.values()),
        validation_proof_sha256=plan['validation_proof_sha256'],
        h200_proof_sha256=plan['h200_proof_sha256'],
        script_sha256=sha(Path(__file__)), plan_sha256=sha(ROOT/'hash-plan.json'),
        score_sha256=plan['records'], recorded_unix=time.time(),
        scope='Current A100 validation output hashes verified. These hashes equal those recorded by the separately completed H200 replay. Original training-time validation matrices were not retained; full public test inference remains outstanding.')
    with (ROOT/'SCORES_VERIFIED.json').open('x') as target:
        target.write(json.dumps(proof, indent=2)+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k!='score_sha256'}, indent=2))


if __name__ == '__main__':
    main()
