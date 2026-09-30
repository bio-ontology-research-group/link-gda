"""Exercise the schematic in an isolated checkout without touching manuscript files."""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_calibration_figure_renders_in_isolated_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout with spaces"
    target = checkout / "code/figures/make_calibration_fig.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(root / "code/figures/make_calibration_fig.py", target)
    (checkout / "paper/fig").mkdir(parents=True)
    result = subprocess.run(
        [sys.executable, str(target)], cwd=checkout,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    output = checkout / "paper/fig/fig_calibration.pdf"
    assert output.read_bytes().startswith(b"%PDF-")
    assert output.stat().st_size > 1000
