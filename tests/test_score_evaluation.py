"""Exercise the actual evaluator with explicit score identities and co-causal genes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from link_gda.score_io import convert_prediction_tsv, load_scores, sha256
REPO = Path(__file__).resolve().parents[1]


def test_prediction_conversion_preserves_scores_and_rejects_conflicting_queries(tmp_path):
    source = tmp_path / 'predictions.tsv'
    Path(str(source) + '.candidates.txt').write_text('g0\ng1\ng2\n')
    source.write_text('g0\td0\t0\t0.123456789012345\t0.2\t0.9\n'
                      'g1\td0\t1\t0.123456789012345\t0.2\t0.9\n')
    output = tmp_path / 'scores.npz'
    meta = convert_prediction_tsv(source, output)
    assert meta['pairs'] == 2 and meta['diseases'] == 1
    x = load_scores(output, ['g2', 'g0', 'g1'], ['d0'], expected_sha256=meta['scores_sha256'])
    np.testing.assert_array_equal(x, [[.9, .123456789012345, .2]])
    with pytest.raises(ValueError, match='checksum'):
        load_scores(output, ['g0', 'g1', 'g2'], ['d0'], expected_sha256='0' * 64)
    source.write_text(source.read_text().replace('g1\td0\t1\t0.123456789012345', 'g1\td0\t1\t0.1'))
    with pytest.raises(ValueError, match='Inconsistent'):
        convert_prediction_tsv(source, output)


def test_evaluator_computes_every_stage_from_real_scores(tmp_path):
    fold = tmp_path / 'data/fold0'
    fold.mkdir(parents=True)
    (fold / 'candidates.txt').write_text('g0\ng1\ng2\ng3\n')
    (fold / 'training.tsv').write_text('Gene\tDisease\ng0\ttrain\ng1\ttrain\n')
    (fold / 'test.tsv').write_text('Gene\tDisease\ng0\td1\ng1\td1\ng2\td2\ng2\td3\n')
    (fold / 'profiles.json').write_text(json.dumps({'train': ['p'], 'd1': ['p'], 'd2': ['q'], 'd3': ['q']}))
    scorefile = tmp_path / 'scores.npz'
    np.savez(scorefile, scores=np.array([[5., 4., 3., 1.], [1., 2., 4., 3.], [1., 2., 4., 3.]]),
             genes=np.array(['g0', 'g1', 'g2', 'g3']), diseases=np.array(['d1', 'd2', 'd3']))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'models': [{'name': 'Exomiser-Phive', 'folds': {
        '0': {'path': 'scores.npz', 'sha256': sha256(scorefile)}}}]}))
    out = tmp_path / 'report.json'
    result = subprocess.run([sys.executable, str(REPO / 'code/analysis/evaluate_stepwise_bias.py'),
                             '--base-dir', str(tmp_path), '--score-manifest', str(manifest),
                             '--folds', '0', '--workers', '1', '--out-json', str(out),
                             '--ranks-dir', str(tmp_path / 'ranks')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60,
                            env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'})
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())['Exomiser-Phive']
    assert data['S0_Raw_All']['mr'] == 1.25
    assert data['S3_ZeroDirectOverlap_Raw']['n'] == 2
    assert data['S4_Cumulative_All4']['n'] == 1
    assert data['S1_PositiveFilter_All']['mr'] < data['S1_Cal_All']['mr']
    assert data['S1_EqualProfile_All']['n'] == 2
    with np.load(tmp_path / 'ranks/fold0-ranks.npz') as ranks:
        assert ranks['Exomiser-Phive::r_s1_positive_filter'][1] < ranks['Exomiser-Phive::r_s1'][1]
    meta = json.loads(out.with_suffix('.metadata.json').read_text())
    assert meta['cohorts']['combined_diseases'] == 2
    assert meta['cohorts']['combined_profiles'] == 1
