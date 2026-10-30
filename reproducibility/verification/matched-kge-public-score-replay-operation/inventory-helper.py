"""Install public matched weights beside the independently verified public inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'source'
sys.path[:0] = [str(SOURCE/'code'), str(SOURCE/'reproducibility')]
from phenoquery.common import read, sha, write
from reproduce.artifacts import fetch


def verify_workspace(root, expected, components):
    paths = list(root.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    embedded = {'MANIFESTS/'+name+'.json' for name in components}
    assert not (set(expected) & (embedded | {'.artifacts.lock'}))
    actual = {p.relative_to(root).as_posix() for p in paths if p.is_file()}
    assert actual == set(expected) | embedded | {'.artifacts.lock'}
    assert (root/'.artifacts.lock').read_bytes() == b''
    members = {}
    for name,component in components.items():
        assert name == component['name']
        assert read(root/'MANIFESTS'/(name+'.json')) == dict(schema=1, component=name, files=component['files'])
        assert not (set(members) & set(component['files']))
        members.update({n:{k:r[k] for k in ('sha256','bytes')} for n,r in component['files'].items()})
    assert members == expected
    for name,record in expected.items():
        path = root/name
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], name


def main():
    started = time.time()
    plan = read(ROOT/'download-plan.json')
    source, inputs = read(ROOT/'SOURCE_READY.json'), read(ROOT/'INPUTS_READY.json')
    assert source['status'] == inputs['status'] == 'passed'
    assert plan['source_proof_sha256'] == inputs['source_proof_sha256'] == sha(ROOT/'SOURCE_READY.json')
    assert plan['input_proof_sha256'] == sha(ROOT/'INPUTS_READY.json')
    assert plan['script_sha256'] == sha(Path(__file__))
    assert plan['sbatch_sha256'] == sha(ROOT/'download-checkpoints.sbatch')
    assert inputs['input_download_plan_sha256'] == sha(ROOT/'input-download-plan.json')
    assert inputs['script_sha256'] == sha(ROOT/'download-inputs.py')
    assert source['commits']['source'] == '68101ed08e3e271f5f2276724f8a5338b9cf32a2'
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() == source['commits']['source']
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True)
    for name,digest in source['analysis_files'].items():
        assert sha(SOURCE/name) == digest, name
    workspace, cache = ROOT/'workspace', ROOT/'checkpoint-downloads'
    assert not cache.exists() and not (ROOT/'DOWNLOAD_COMPLETE.json').exists()
    manifest_path = ROOT/'checkpoint-release-manifest.json'
    assert not manifest_path.exists()
    input_components = {}
    input_archives = {c['name']:c for c in inputs['archives']}
    assert len(input_archives) == 11 and inputs['file_count'] == len(inputs['files']) == 142
    for filename,digest in inputs['manifests_sha256'].items():
        assert Path(filename).name == filename and sha(ROOT/filename) == digest
        for component in read(ROOT/filename)['components']:
            if component['name'] in input_archives:
                assert component['name'] not in input_components
                assert all(component[k] == v for k,v in input_archives[component['name']].items())
                input_components[component['name']] = component
    assert set(input_components) == set(input_archives)
    verify_workspace(workspace, inputs['files'], input_components)
    url = plan['manifest_url']
    assert url == 'https://github.com/bio-ontology-research-group/link-gda/releases/download/matched-kge-checkpoints-20261029-v1/release-manifest.json'
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read()
    assert hashlib.sha256(raw).hexdigest() == plan['manifest_sha256']
    manifest = json.loads(raw)
    p = manifest['provenance']
    assert manifest['schema'] == 1 and p['training_runs'] == 100 and p['model_checkpoints'] == 400
    assert p['freeze_sha256'] == plan['freeze_sha256']
    assert p['protocol_sha256'] == source['training_files']['reproducibility/protocols/matched-kge-20261029.md']
    assert p['source'] == {n.removeprefix('code/'):d for n,d in source['training_files'].items() if n.startswith('code/')}
    assert p['gene_phenotypes_sha256'] == '94829ab611a3f02c4c33d3686028d8eded187c256f852d4e2cecb2f548b4d0b6'
    assert len(p['expected_files']) == 701 and sum(n.endswith('.pt') for n in p['expected_files']) == 400
    assert all(n.startswith('matched-kge/') for n in p['expected_files'])
    assert not any('/scores-' in n or n.endswith('/SCORED.json') for n in p['expected_files'])
    assert not (set(p['expected_files']) & set(inputs['files']))
    components = dict(input_components)
    for component in manifest['components']:
        assert component['url'] == url.rsplit('/',1)[0]+'/'+component['archive']
        assert component['name'] not in components
        components[component['name']] = component
    manifest_path.write_bytes(raw)
    fetch(manifest_path, workspace, cache)
    expected = {**inputs['files'], **p['expected_files']}
    assert len(expected) == 843
    verify_workspace(workspace, expected, components)
    root = workspace/'matched-kge'
    assert sha(root/'FROZEN.json') == p['freeze_sha256']
    frozen = read(root/'FROZEN.json')
    runs = {f'fold{fold}/{graph}/{model}' for fold in range(10)
            for graph in ('original','nobridge') for model in ('U1','T1','INDIGENA','CONVKB','PPOOL')}
    assert frozen['schema'] == 1 and set(frozen['training']) == runs
    for relative,digest in frozen['training'].items():
        folder = root/relative
        assert sha(folder/'TRAINED.json') == digest
        trained, identity = read(folder/'TRAINED.json'), read(folder/'identity.json')
        assert trained['identity_sha256'] == sha(folder/'identity.json')
        assert identity['source'] == p['source'] and identity['protocol_sha256'] == p['protocol_sha256']
        for name,digest in identity['inputs'].items():
            assert expected[f"benchmark/data/fold{identity['fold']}/{name}"]['sha256'] == digest
        assert not (folder/'SCORED.json').exists() and not list(folder.glob('scores-*'))
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True)
    proof = dict(status='passed', source_proof_sha256=sha(ROOT/'SOURCE_READY.json'),
        input_proof_sha256=sha(ROOT/'INPUTS_READY.json'), download_plan_sha256=sha(ROOT/'download-plan.json'),
        script_sha256=sha(Path(__file__)), artifact_fetcher_sha256=sha(SOURCE/'code/reproduce/artifacts.py'),
        manifest_sha256=sha(manifest_path), freeze_sha256=p['freeze_sha256'],
        checkpoint_files=400, checkpoint_collection_files=701, input_files=142,
        files=expected, file_count=843, no_predictions_before_inference=True,
        checkpoint_archives=len(manifest['components']), archive_bytes=sum(c['bytes'] for c in manifest['components']),
        elapsed_seconds=time.time()-started,
        scope='Fresh public checkpoint download with all input and weight hashes verified, exact original source/freeze/input bindings, and no existing predictions. Independent inference and numerical replay remain separate.')
    write(ROOT/'DOWNLOAD_COMPLETE.json', proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='files'},indent=2),flush=True)


if __name__ == '__main__':
    main()
