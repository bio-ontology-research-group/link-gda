import hashlib
import json
from pathlib import Path
import subprocess
import sys

from test_baseline_tables import inputs

ROOT = Path(__file__).resolve().parents[1]


def invoke(tmp_path, broken_digest=False):
    report, metadata, manifest = inputs(tmp_path)
    manifest_path = tmp_path / 'scores.json'
    manifest_path.write_text(json.dumps(manifest))
    metadata['score_manifest_sha256'] = ('incorrect' if broken_digest else
                                        hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    report_path = tmp_path / 'baselines.json'
    report_path.write_text(json.dumps(report))
    report_path.with_suffix('.metadata.json').write_text(json.dumps(metadata))
    out = tmp_path / 'tables'
    result = subprocess.run([sys.executable, str(ROOT / 'reproducibility/generate_bias_tables.py'),
                             '--baseline-report', str(report_path), '--baseline-score-manifest',
                             str(manifest_path), '--out-dir', str(out)], capture_output=True, text=True)
    return result, out


def test_complete_baselines_replace_obsolete_rows(tmp_path):
    result, out = invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    main = (out / 'revised-bias-stepwise.tex').read_text()
    assert 'Phrank, native HPO, direct & 0.12 / 1234.0' in main
    assert 'PhenoRank, native 2016, 1,000 simulations' in main
    assert 'PhenoRank, native 2016, continuous' in main
    assert '2016 resources, human refit' not in main
    assert 'PhenoQuery, no disease + EQ-typed' in main


def test_mismatched_report_manifest_cannot_generate_tables(tmp_path):
    result, out = invoke(tmp_path, broken_digest=True)
    assert result.returncode != 0
    assert 'different score manifest' in result.stderr
    assert not list(out.glob('*.tex'))
