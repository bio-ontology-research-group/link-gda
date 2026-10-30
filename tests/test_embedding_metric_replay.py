import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'reproducibility/replay_embedding_metrics.py'
spec = importlib.util.spec_from_file_location('embedding_replay', SCRIPT)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def fixture(tmp_path):
    files = {'predictions': ('scores.tsv', 'g1\td1\t0\t4\t3\t2\t1\ng3\td2\t2\t1\t3\t2\t4\ng4\td3\t3\t2\t1\t3\t4\n'),
             'candidates': ('candidates.txt', 'g1\ng2\ng3\ng4\n'),
             'training': ('training.tsv', 'Gene\tDisease\ng1\ttrain1\ng2\ttrain2\n')}
    inputs = {}
    for kind, (name, content) in files.items():
        (tmp_path / name).write_text(content)
        inputs[kind] = dict(path=name, sha256=hashlib.sha256(content.encode()).hexdigest())
    manifest = dict(schema=1, entries=[dict(output='example/fold_0.json', inputs=inputs)])
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_relocated_isolated_cli_replays_hand_computed_ranks(tmp_path):
    manifest, _ = fixture(tmp_path)
    output = tmp_path / 'workspace'
    result = subprocess.run([sys.executable, '-I', str(SCRIPT), '--manifest', str(manifest),
                             '--input-root', str(tmp_path), '--workspace', str(output)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    value = json.loads((output / 'revised-metrics/example/fold_0.json').read_text())
    assert value['settings']['raw']['ranks'] == [1, 3, 1]
    assert value['settings']['raw']['within_training_status']['seen']['ranks'] == [1]
    assert value['settings']['raw']['within_training_status']['unseen']['ranks'] == [2, 1]
    assert (output / 'embedding-replay-shard-0-of-1.json').exists()
    with pytest.raises(FileExistsError):
        replay.replay(manifest, tmp_path, output)
    (tmp_path / 'scores.tsv').write_text('changed')
    with pytest.raises(ValueError, match='checksum differs'):
        replay.replay(manifest, tmp_path, output, replace=True)
    assert not (output / 'embedding-replay-shard-0-of-1.json').exists()


@pytest.mark.parametrize('change', ['traversal', 'duplicate', 'missing_training', 'missing_sidecars'])
def test_invalid_manifest_rejected_before_output(tmp_path, change):
    path, manifest = fixture(tmp_path)
    entry = manifest['entries'][0]
    if change == 'traversal': entry['inputs']['predictions']['path'] = '../escape.tsv'
    elif change == 'duplicate': manifest['entries'].append(entry)
    elif change == 'missing_training': del entry['inputs']['training']
    else: entry['inputs']['reference_stats'] = entry['inputs']['predictions']
    path.write_text(json.dumps(manifest))
    output = tmp_path / 'workspace'
    with pytest.raises(ValueError):
        replay.replay(path, tmp_path, output)
    assert not output.exists()


@pytest.mark.parametrize('script', ['query_permutation.py', 'annotation_audit.py',
                                   'paired_diagnostics.py', 'rank_strata.py', 'revised_controls.py'])
def test_diagnostic_cli_isolated_help(tmp_path, script):
    result = subprocess.run([sys.executable, '-I', str(ROOT / 'reproducibility' / script), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_wrong_query_expectation_enumerates_other_diseases():
    import numpy as np
    from query_permutation import expected_wrong_query_ranks
    ranks = np.array([[1, 2.5, 2.5, 4], [4, 2, 3, 1], [2, 1, 4, 3]])
    inverse = np.array([0, 0, 1, 2])
    genes = np.array([0, 1, 3, 2])
    expected = np.array([np.mean([ranks[q, g] for q in range(3) if q != d])
                         for d, g in zip(inverse, genes)])
    np.testing.assert_array_equal(expected_wrong_query_ranks(ranks, inverse, genes), expected)
