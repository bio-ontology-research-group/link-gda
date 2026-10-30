"""Legacy table entry points own explicit input and output workspaces."""
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script', ['finalize_results.py', 'supporting_tables.py'])
def test_help_without_data_or_checkout_import_path(tmp_path, script):
    result = subprocess.run([sys.executable, '-I', str(ROOT / 'reproducibility' / script), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--workspace' in result.stdout and '--out-dir' in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('script', ['finalize_results.py', 'supporting_tables.py'])
def test_missing_metric_workspace_does_not_write_output(tmp_path, script):
    output = tmp_path / 'generated paper assets'
    result = subprocess.run([
        sys.executable, '-I', str(ROOT / 'reproducibility' / script),
        '--workspace', str(tmp_path / 'missing data'), '--out-dir', str(output)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'Missing per-association metrics' in result.stderr
    assert not output.exists()
