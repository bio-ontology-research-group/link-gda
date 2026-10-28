from pathlib import Path
import pytest
from phenoquery.campaign import freeze_global, verify_global, verify_gates
from phenoquery.common import sha, write


def prepare(root):
    names = ['general', 'query']
    write(root / 'CONTROLLED.json', {'campaigns': names, 'runs_per_campaign': 1,
                                    'expected_runs': 2, 'initializations': names,
                                    'source_sha256': 'fixed-source'})
    for name in names:
        child = root / name
        run = child / 'training' / 'typed_and-fold0'
        run.mkdir(parents=True)
        completion = {}
        for file, key in [('best.pt', 'checkpoint_sha256'), ('initial.pt', 'initial_checkpoint_sha256'),
                          ('validation.npz', 'validation_sha256'), ('initial-validation.npz', 'initial_validation_sha256')]:
            (run / file).write_bytes(file.encode())
            completion[key] = sha(run / file)
        write(run / 'COMPLETE.json', completion)
        write(child / 'SELECTION.json', {'runs': {'typed_and-fold0': {
            'completion_sha256': sha(run / 'COMPLETE.json')}}})
    return root


def test_requires_every_campaign_before_test(tmp_path):
    root = prepare(tmp_path)
    missing = root / 'query' / 'SELECTION.json'
    contents = missing.read_bytes()
    missing.unlink()
    with pytest.raises(FileNotFoundError): freeze_global(root)
    assert not (root / 'GLOBAL-SELECTION.json').exists()
    missing.write_bytes(contents)
    digest = freeze_global(root)
    verify_global(root / 'general', digest)


def test_detects_changed_initial_checkpoint(tmp_path):
    root = prepare(tmp_path)
    (root / 'query/training/typed_and-fold0/initial.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed artifact'): freeze_global(root)


def test_detects_changed_other_seed_after_freeze(tmp_path):
    root = prepare(tmp_path)
    digest = freeze_global(root)
    (root / 'query/SELECTION.json').write_text('{}')
    with pytest.raises(ValueError, match='selection changed'): verify_global(root / 'general', digest)


def test_rejects_partial_run_count(tmp_path):
    root = prepare(tmp_path)
    write(root / 'query/SELECTION.json', {'runs': {}})
    with pytest.raises(ValueError, match='Incomplete selection'): freeze_global(root)


def test_gate_binds_evidence_and_source(tmp_path):
    root = prepare(tmp_path)
    evidence = root / 'pilot-complete.json'
    evidence.write_text('{}')
    for name in ['general', 'query']:
        write(root / f'GATE-{name}.json', {'source_sha256': 'fixed-source', 'steps': 4,
                                         'validation_boundary': {'steps':4,'restored_state_bitwise_equal':True,'pending_validation_completed':True},
                                         'artifacts': {'pilot-complete.json': sha(evidence)}})
    verify_gates(root)
    evidence.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='artifact changed'): verify_gates(root)
