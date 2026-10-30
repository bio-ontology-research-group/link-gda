"""The released legacy metric command must work without checkout PYTHONPATH."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'reproducibility/revision_metrics.py'


def invoke(tmp_path, wrong_index=False):
    working = tmp_path / 'independent working directory'
    working.mkdir()
    scores = working / 'scores.tsv'
    scores.write_text('g1\td1\t' + ('1' if wrong_index else '0') +
                      '\t2\t2\t0\ng2\td2\t1\t6\t1\t2\ng3\td3\t2\t0\t3\t5\n')
    candidates = working / 'candidates.txt'
    candidates.write_text('g1\ng2\ng3\n')
    output = working / 'metrics.json'
    result = subprocess.run(
        [sys.executable, '-I', str(SCRIPT), '--predictions', str(scores),
         '--candidate-file', str(candidates), '--out', str(output)],
        cwd=working, capture_output=True, text=True, timeout=30)
    return result, output, scores


def test_command_outside_checkout_with_isolated_python(tmp_path):
    result, output, scores = invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report['source_rows'] == report['evaluated_rows'] == 3
    assert report['keys'] == [dict(gene=f'g{i}', disease=f'd{i}') for i in (1, 2, 3)]
    assert report['settings']['raw']['ranks'] == [1.5, 3.0, 1.0]
    assert report['settings']['raw']['unfiltered']['pair_average']['mr'] == pytest.approx(11 / 6)
    # For d1, leave-one-disease-out scores are -1/3, 0, and -7/3.
    assert report['settings']['disease_loo']['ranks'] == [2.0, 3.0, 1.0]
    assert report['provenance']['predictions']['sha256'] == hashlib.sha256(scores.read_bytes()).hexdigest()


def test_command_rejects_wrong_candidate_identity(tmp_path):
    result, output, _ = invoke(tmp_path, wrong_index=True)
    assert result.returncode != 0
    assert 'true-gene index mismatch' in result.stderr
    assert not output.exists()
