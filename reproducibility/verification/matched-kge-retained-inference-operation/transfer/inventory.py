"""Closed inventory and actual proof bindings for matched H200 output transfer."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

FAMILIES = ('U1','T1','INDIGENA','CONVKB','PPOOL')
CHOICES = ('raw_mr','raw_head','calibrated_mr','calibrated_head')
RUNS = tuple(f'fold{f}/{g}/{m}' for f in range(10) for g in ('original','nobridge') for m in FAMILIES)
ROOT_METADATA = {'inference-plan.json','SOURCE_READY.json','INPUTS_REFERENCE.json',
    'STAGED_INPUTS.json','staging-plan.json','checkpoint-release-manifest.json',
    'stage-inputs.py','annotation-provenance.json',
    'validation-verification.json','validation-native-completion.log',
    'run-inference.py','run-inference.sbatch','capture-native.py'}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def relative(name):
    p = PurePosixPath(name)
    if (not re.fullmatch(r'[A-Za-z0-9_./-]+',name) or p.is_absolute()
            or '..' in p.parts or str(p) != name):
        raise ValueError('Unsafe or noncanonical relative path')
    return p


def inventory(capture):
    if (capture['status'] != 'native_completed' or capture['cluster'] != 'orix'
            or type(capture['inference_array']) is not int or capture['inference_array'] <= 0):
        raise ValueError('Require completed native H200 inference')
    if capture.get('native_allocations') != 3 or capture.get('folds') != 10:
        raise ValueError('Require three native allocations covering ten folds')
    jobs = capture['native_fold_job_ids']
    if (set(jobs) != {str(i) for i in range(10)} or len(set(jobs.values())) != 3
            or any(jobs[str(i)] != jobs[str(i%3)] for i in range(10))
            or any(type(j) is not int or j <= 0 for j in jobs.values())):
        raise ValueError('Require the exact ten-fold mapping onto three distinct native allocations')
    binaries = {'workspace/matched-kge/'+r+'/scores-'+c+'.npz' for r in RUNS for c in CHOICES}
    scorer = {'workspace/matched-kge/'+r+'/SCORED.json' for r in RUNS}
    if set(capture['binary_files']) != binaries:
        raise ValueError('Require all 400 prediction files')
    metadata = capture['metadata_files']
    locks = {'workspace/matched-kge/'+r+'/run.lock' for r in RUNS}
    expected = ROOT_METADATA | scorer | locks | {f'gpu-runs/fold{f}/{n}' for f in range(10) for n in ('STARTED.json','COMPLETE.json')}
    attempts = {n for n in metadata if re.fullmatch(r'gpu-runs/fold[0-9]/attempt-[0-9]+-[0-9]+\.json',n)}
    if set(metadata) != expected | attempts:
        raise ValueError('Incomplete or extra captured metadata')
    for f in range(10):
        if not any(n.startswith(f'gpu-runs/fold{f}/') for n in attempts):
            raise ValueError('Missing native runtime')
    for name,record in metadata.items():
        relative(name); raw=record['text'].encode()
        if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('Captured metadata bytes differ')
    for name,record in capture['binary_files'].items():
        relative(name)
        if (type(record['bytes']) is not int or record['bytes'] <= 0
                or not re.fullmatch('[0-9a-f]{64}',record['sha256'])):
            raise ValueError('Invalid binary identity')
    hashes = {n:r['sha256'] for n,r in capture['binary_files'].items()}
    hashes.update({n:metadata[n]['sha256'] for n in scorer})
    if capture['score_files_sha256'] != hashes:
        raise ValueError('Scorer identities differ')
    return binaries | scorer | locks


def groups(capture):
    inventory(capture)
    return {'workspace/matched-kge/'+run: ['scores-'+c+'.npz' for c in CHOICES] for run in RUNS}


def plan(root):
    p=read(root/'plan.json')
    if set(p['scripts']) != {'inventory.py','stage-metadata.py','relay.py'}:
        raise ValueError('Incomplete transfer source inventory')
    for name,digest in p['scripts'].items():
        if sha(root/name) != digest: raise ValueError('Transfer source changed')
    for key in ('source','destination','operation_destination'):
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+',p[key]) or '..' in PurePosixPath(p[key]).parts:
            raise ValueError('Unsafe operation path')
    if sha(root/'NATIVE_CAPTURE.json') != p['capture_sha256']:
        raise ValueError('Native capture changed')
    c=read(root/'NATIVE_CAPTURE.json'); inventory(c)
    if c['inference_array'] != p['inference_array'] or c['inference_plan_sha256'] != p['inference_plan_sha256']:
        raise ValueError('Inference identity differs')
    metadata=c['metadata_files']
    if metadata['inference-plan.json']['sha256'] != p['inference_plan_sha256']:
        raise ValueError('Captured inference plan differs')
    execution=json.loads(metadata['inference-plan.json']['text'])
    if (execution['capture_script_sha256'] != c['capture_script_sha256']
            or c['capture_script_sha256'] != metadata['capture-native.py']['sha256']):
        raise ValueError('Capture source differs')
    if sha(root/'native-completion.log') != c['native_completion_sha256']:
        raise ValueError('Native accounting changed')
    observed={}
    for line in (root/'native-completion.log').read_text().strip().splitlines():
        name,raw,state,code=line.strip().split('|')
        if not re.fullmatch(str(p['inference_array'])+r'_[0-2]',name) or not raw.isdigit():
            raise ValueError('Unexpected native task')
        task=name.split('_')[1]
        if task in observed or state != 'COMPLETED' or code != '0:0':
            raise ValueError('Duplicated or incomplete native task')
        observed[task]=int(raw)
    if set(observed) != {str(i) for i in range(3)}:
        raise ValueError('Require three completed allocations')
    observed = {str(f):observed[str(f%3)] for f in range(10)}
    if observed != c['native_fold_job_ids']:
        raise ValueError('Native accounting and capture differ')
    return p,c
