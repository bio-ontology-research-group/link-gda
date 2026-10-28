"""Tests for the reproducibility/ suite and new debiasing CLI entrypoints."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "code/training/retrain_nobridge_kge.py",
        "code/baselines/run_phenoquery.py",
        "code/baselines/run_ultra_matched.py",
        "code/baselines/run_query_campaign.py",
        "code/baselines/run_controlled_query.py",
        "code/baselines/native_phrank/run.py",
        "code/baselines/native_phenorank/run.py",
        "code/analysis/convert_prediction_scores.py",
        "code/baselines/run_matched_phrank_phenorank.py",
        "code/analysis/evaluate_stepwise_bias.py",
        "reproducibility/generate_bias_tables.py",
        "reproducibility/generate_factor_table.py",
    ],
)
def test_new_cli_entrypoints_answer_help(script, tmp_path):
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--help"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, res.stderr
    assert "usage" in res.stdout.lower()


def test_run_pipeline_sh_help(tmp_path):
    res = subprocess.run(
        ["bash", str(REPO_ROOT / "reproducibility" / "run_pipeline.sh"), "--help"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, res.stderr
    assert "train-nobridge-kge" in res.stdout
    assert "run-phrank-phenorank" in res.stdout
    assert "evaluate-stepwise-bias" in res.stdout
    assert "generate-tables" in res.stdout


def test_generate_bias_tables_from_saved_report(tmp_path):
    report_path = REPO_ROOT / "reproducibility" / "results" / "stepwise-bias-report-v4.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text())
    assert len(data) == 66

    out_dir = tmp_path / "tables"
    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "reproducibility" / "generate_bias_tables.py"),
            "--report-json",
            str(report_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, res.stderr
    main_tex = out_dir / "revised-bias-stepwise.tex"
    supp_tex = out_dir / "revised-bias-baselines-full.tex"
    assert main_tex.is_file()
    assert supp_tex.is_file()
    assert "1993.2" in main_tex.read_text()
    assert "1536.9" in main_tex.read_text()
