"""Fetch the complete published matched-score collection into an empty workspace."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent/'full-matched-public-replay-v1'
SOURCE = SHARED/'source'
COMMIT = '68101ed08e3e271f5f2276724f8a5338b9cf32a2'
HELPER_SHA = '9ad016c91c17a11f8bacfd684f2711e040eccc6525ef57083db92ff8323986a6'
URL = 'https://github.com/bio-ontology-research-group/link-gda/releases/download/matched-kge-scores-20261029-v1/release-manifest.json'
sys.path[:0] = [str(SOURCE/'code'), str(SOURCE/'reproducibility')]
from phenoquery.common import read, sha, write
from reproduce.artifacts import fetch


def main():
    started = time.time()
    plan, source = read(ROOT/'download-plan-v2.json'), read(SHARED/'SOURCE_READY.json')
    assert source['status'] == 'passed' and source['commits']['source'] == COMMIT
    assert plan['source_proof_sha256'] == sha(SHARED/'SOURCE_READY.json')
    assert plan['script_sha256'] == sha(Path(__file__)) and plan['sbatch_sha256'] == sha(ROOT/'download-v2.sbatch')
    assert plan['manifest_url'] == URL
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() == COMMIT
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True)
    for name,digest in source['analysis_files'].items():
        assert sha(SOURCE/name) == digest, name
    helper = SHARED/'download-checkpoints.py'
    assert sha(helper) == HELPER_SHA
    spec = importlib.util.spec_from_file_location('matched_inventory',helper)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    workspace, cache, manifest_path = ROOT/'workspace', ROOT/'downloads', ROOT/'release-manifest.json'
    assert workspace.is_dir() and cache.is_dir() and manifest_path.is_file()
    assert sha(ROOT/'resume-prefix.py') == plan['resume_helper_sha256']
    assert plan['failed_download_job'] == 54818848
    accounting = subprocess.check_output(['sacct','-j','54818848','--format=JobID,State,ExitCode','-n','-P'],text=True)
    assert '54818848|FAILED|1:0' in accounting
    assert not (ROOT/'DOWNLOAD_COMPLETE.json').exists()
    with urllib.request.urlopen(URL,timeout=120) as response:
        raw = response.read()
    assert hashlib.sha256(raw).hexdigest() == plan['manifest_sha256']
    manifest = json.loads(raw); p = manifest['provenance']
    assert manifest['schema'] == 1 and p['training_runs'] == 100 and p['score_files'] == 400
    assert p['models'] == 40 and p['conditions_per_model'] == 42 and p['expected_candidates'] == 13625
    assert p['freeze_sha256'] == plan['freeze_sha256']
    assert p['builder_sha256'] == sha(SOURCE/'reproducibility/build_matched_kge_scores.py')
    assert len(p['expected_files']) == 886 and sum('/scores-' in n for n in p['expected_files']) == 400
    assert not any(n.endswith('.pt') for n in p['expected_files'])
    assert all(c['url'] == URL.rsplit('/',1)[0]+'/'+c['archive'] for c in manifest['components'])
    components = {c['name']:c for c in manifest['components']}
    assert len(components) == len(manifest['components'])
    assert manifest_path.read_bytes() == raw
    resume_spec = importlib.util.spec_from_file_location('resume_prefix',ROOT/'resume-prefix.py')
    resume_module = importlib.util.module_from_spec(resume_spec); resume_spec.loader.exec_module(resume_module)
    prefix = resume_module.verify_prefix(workspace,cache,manifest,module.verify_workspace)
    write(ROOT/'RESUME_PREFIX_VERIFIED.json',prefix)
    print('RESUME_PREFIX_VERIFIED',prefix['cached_archives'],prefix['installed_components'],flush=True)
    fetch(manifest_path,workspace,cache)
    module.verify_workspace(workspace,p['expected_files'],components)
    assert sha(workspace/p['score_manifest']) == p['score_manifest_sha256']
    assert sha(workspace/p['reference_report']) == p['replay']['report_sha256']
    assert sha(workspace/p['reference_summary']) == p['replay']['summary_sha256']
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True)
    proof = dict(status='passed', source_commit=COMMIT, source_proof_sha256=sha(SHARED/'SOURCE_READY.json'),
        download_plan_sha256=sha(ROOT/'download-plan-v2.json'), script_sha256=sha(Path(__file__)),
        manifest_sha256=sha(manifest_path), inventory_helper_sha256=sha(helper),
        resume_prefix_proof_sha256=sha(ROOT/'RESUME_PREFIX_VERIFIED.json'),
        resume_helper_sha256=sha(ROOT/'resume-prefix.py'),failed_download_job=54818848,
        artifact_fetcher_sha256=sha(SOURCE/'code/reproduce/artifacts.py'), files=886, score_files=400,
        variants=40, conditions=42, freeze_sha256=p['freeze_sha256'], archives=len(components),
        archive_bytes=sum(c['bytes'] for c in components.values()), elapsed_seconds=time.time()-started,
        scope='Fresh unauthenticated public matched-score download with complete member verification. Statistical replay and independent inference remain separate.')
    write(ROOT/'DOWNLOAD_COMPLETE.json',proof)
    print(json.dumps(proof,indent=2),flush=True)


if __name__ == '__main__': main()
