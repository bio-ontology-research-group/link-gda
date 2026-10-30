"""Stage small execution evidence; inspect exact destination inventory after relay."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from inventory import inventory, plan, read, sha

ROOT = Path(__file__).resolve().parent


def safe_path(root, name):
    path = root / name
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError('Refuse a symlink in destination path: ' + str(candidate))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-transfer', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'metadata.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p, capture = plan(ROOT)
    destination = Path(p['destination']).resolve(strict=True)
    assert ROOT == Path(p['operation_destination']).resolve(strict=True)
    checkpoint_proof = destination / 'RETAINED_INPUTS_STAGED.json'
    assert sha(checkpoint_proof) == p['checkpoint_stage_proof_sha256']
    staged_inputs = read(checkpoint_proof)
    assert staged_inputs['status'] == 'passed' and staged_inputs['files'] == 843
    assert staged_inputs['model_checkpoints'] == 400 and staged_inputs['no_test_predictions']
    captured_download = json.loads(capture['metadata_files']['STAGED_INPUTS.json']['text'])
    assert staged_inputs['staging_proof_sha256'] == capture['metadata_files']['STAGED_INPUTS.json']['sha256']
    marker = ROOT / 'METADATA_STAGED.json'
    expected = inventory(capture)
    metadata = {n: r['text'].encode() for n, r in capture['metadata_files'].items()}
    metadata.update({n: (ROOT / n).read_bytes() for n in ('NATIVE_CAPTURE.json', 'native-completion.log')})
    if not args.verify_transfer and not marker.exists():
        assert all(not safe_path(destination, n).exists() for n in expected)
        # Check every existing destination before writing any metadata.
        for name, payload in metadata.items():
            target = safe_path(destination, name)
            if target.exists():
                assert target.is_file() and target.read_bytes() == payload, name
        for name, payload in metadata.items():
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open('xb') as stream:
                    stream.write(payload)
        marker.write_text(json.dumps(dict(status='passed', plan_sha256=sha(ROOT/'plan.json'),
            capture_sha256=p['capture_sha256'], metadata_files=len(metadata),
            no_predictions_before_staging=True, recorded_unix=time.time()), indent=2)+'\n')
    assert read(marker)['plan_sha256'] == sha(ROOT / 'plan.json')
    for name, payload in metadata.items():
        assert safe_path(destination, name).read_bytes() == payload, name
    if args.verify_transfer:
        actual = set()
        folder = destination/'workspace'
        for path in folder.rglob('*'):
            assert not path.is_symlink(), path
            if path.is_file():
                actual.add(path.relative_to(destination).as_posix())
        public = {'workspace/'+n for n in captured_download['files']}
        assert len(public) == 843 and actual == expected | public
        for name, record in captured_download['files'].items():
            assert safe_path(destination, 'workspace/'+name).stat().st_size == record['bytes'], name
        for name, record in capture['binary_files'].items():
            target = safe_path(destination, name)
            assert target.is_file() and target.stat().st_size == record['bytes'], name
        proof = dict(status='inventory_ready', plan_sha256=sha(ROOT/'plan.json'),
            capture_sha256=p['capture_sha256'], test_files=len(expected), binary_files=400,
            binary_bytes=sum(r['bytes'] for r in capture['binary_files'].values()),
            metadata_files=len(metadata), recorded_unix=time.time(),
            scope='Exact destination paths and sizes, all metadata bytes verified. Binary hashing and all scientific comparisons require the separate CPU analysis job.')
        target = ROOT / 'DESTINATION_READY.json'
        if target.exists():
            previous = read(target)
            assert {k: v for k, v in previous.items() if k != 'recorded_unix'} == {
                k: v for k, v in proof.items() if k != 'recorded_unix'}
            proof = previous
        else:
            target.write_text(json.dumps(proof, indent=2)+'\n')
        print(json.dumps(proof), flush=True)
    else:
        print(marker.read_text(), flush=True)


if __name__ == '__main__':
    main()
