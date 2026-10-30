"""Fetch fresh public analysis and original matched training source checkouts."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan = json.loads((ROOT/'source-plan.json').read_text())
    assert plan['bootstrap_sha256'] == sha(Path(__file__))
    assert not (ROOT/'SOURCE_READY.json').exists()
    assert not (ROOT/'source').exists() and not (ROOT/'training-source').exists()
    results = {}
    for name, key in [('source', 'analysis_source_commit'), ('training-source', 'training_source_commit')]:
        path = ROOT/name
        subprocess.run(['git', 'init', str(path)], check=True)
        subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', plan['repository']], check=True)
        subprocess.run(['git', '-C', str(path), 'fetch', '--depth=1', 'origin', plan[key]], check=True)
        subprocess.run(['git', '-C', str(path), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == plan[key]
        assert not subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
        results[name] = actual
    for name, digest in plan['analysis_files'].items():
        assert sha(ROOT/'source'/name) == digest, name
    expected = plan['training_files']
    actual = {p.relative_to(ROOT/'training-source').as_posix()
              for p in (ROOT/'training-source/code').rglob('*.py')}
    actual.add('reproducibility/protocols/matched-kge-20261029.md')
    assert actual == set(expected)
    for name, digest in expected.items():
        assert sha(ROOT/'training-source'/name) == digest, name
    manifest = ''.join(f'{expected[name]}  {name}\n' for name in sorted(expected))
    assert hashlib.sha256(manifest.encode()).hexdigest() == plan['training_source_manifest_sha256']
    (ROOT/'training-source.sha256').write_text(manifest)
    proof = dict(status='passed', commits=results, public_origin=plan['repository'],
        source_plan_sha256=sha(ROOT/'source-plan.json'), bootstrap_sha256=sha(Path(__file__)),
        analysis_files=plan['analysis_files'], training_files=expected,
        training_source_manifest_sha256=sha(ROOT/'training-source.sha256'), scope=plan['scope'])
    (ROOT/'SOURCE_READY.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(dict(status='passed', commits=results, training_files=len(expected),
        analysis_files=len(plan['analysis_files']), proof_sha256=sha(ROOT/'SOURCE_READY.json'))), flush=True)


if __name__ == '__main__':
    main()
