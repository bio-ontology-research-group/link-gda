"""Link and hash all retained IBEX inputs before receiving fresh GPU predictions."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.time()
    assert os.environ.get('SLURM_JOB_ID', '').isdigit()
    plan = read(ROOT/'analysis-input-plan.json')
    assert plan['script_sha256'] == sha(__file__)
    assert plan['sbatch_sha256'] == sha(ROOT/'prepare-analysis-inputs.sbatch')
    assert plan['assembly_script_sha256'] == sha(ROOT/'stage-inputs.py')
    assert plan['staging_proof_sha256'] == sha(ROOT/'STAGED_INPUTS.json')
    staged = read(ROOT/'STAGED_INPUTS.json')
    assert staged['manifest_sha256'] == sha(ROOT/'checkpoint-release-manifest.json')
    assert staged['input_reference_sha256'] == sha(ROOT/'INPUTS_REFERENCE.json')
    assert staged['source_proof_sha256'] == sha(ROOT/'SOURCE_READY.json')
    assert not (ROOT/'RETAINED_INPUTS_STAGED.json').exists()
    spec = importlib.util.spec_from_file_location('retained_input_links', ROOT/'stage-inputs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.expected_members(read(ROOT/'checkpoint-release-manifest.json'), read(ROOT/'INPUTS_REFERENCE.json'))
    assert files == staged['files']
    module.assemble(ROOT/'workspace', files, plan['retained_paths'])
    for number, (name, record) in enumerate(files.items(), 1):
        path = ROOT/'workspace'/name
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name
        if number % 100 == 0:
            print('VERIFIED_RETAINED_MEMBERS', number, flush=True)
    assert not any('/scores-' in n or n.endswith('/SCORED.json') for n in files)
    proof = dict(status='passed', files=843, model_checkpoints=400, no_test_predictions=True,
        staging_proof_sha256=plan['staging_proof_sha256'], native_job=int(os.environ['SLURM_JOB_ID']),
        plan_sha256=sha(ROOT/'analysis-input-plan.json'), script_sha256=sha(__file__),
        assembly_script_sha256=plan['assembly_script_sha256'], bytes=sum(r['bytes'] for r in files.values()),
        elapsed_seconds=time.time()-started,
        scope='All 843 frozen input members, including 400 selected weights, hard-linked from retained IBEX copies and independently SHA256-verified on allocated CPUs. No inference scores staged yet. Fresh H200 predictions and complete scientific comparisons remain required.')
    with (ROOT/'RETAINED_INPUTS_STAGED.json').open('x') as stream:
        stream.write(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__ == '__main__':
    main()
