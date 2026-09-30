"""Layout checks for the batch-1 migration of analysis tools into code/.

Every moved CLI must answer --help with no import error both from the
repository root and from a foreign working directory (the excluded-benchmark
setup launches these scripts by absolute path), and the excluded-table tools
must reproduce rq1_table.py numbers from small fixtures regardless of the
working directory.
"""
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from analysis.rq1_table import calibrate, load, metrics, summarise

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = REPO_ROOT / "code" / "analysis"
FIGURES = REPO_ROOT / "code" / "figures"

CLICK_CLIS = [
    "aggregated_sem_sim_metrics.py",
    "analyze_excluded_seeds.py",
    "check_data_leakage.py",
    "compare_calibration_panels.py",
    "excluded_table.py",
    "gen_overlap_tables.py",
    "make_overlap_labels.py",
    "ultra_excluded_table.py",
    "ultra_metrics.py",
]

RANK_CDF_CONFIGS = {
    "LinkGDA-p": "dim_400_bs_32768_lr_0.001_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma",
    "LinkGDA-pf": "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma",
    "LinkGDA-pfs": "dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma",
    "INDIGENA": "dim_200_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_inductive_bma",
}

EXCLUDED_CELLS = {
    ("owl2vecstar", "uncalibrated"): "dim_100_bs_16384_lr_0.001",
    ("owl2vecstar_gda", "uncalibrated"): "dim_800_bs_131072_lr_0.01",
    ("owl2vecstar", "calibrated"): "dim_200_bs_65536_lr_0.001",
    ("owl2vecstar_gda", "calibrated"): "dim_200_bs_65536_lr_0.001",
}
EXCLUDED_SOURCES = ["func", "func_expr"]
EXCLUDED_PROJECTIONS = ["owl2vecstar", "owl2vecstar_gda"]
EXCLUDED_KEYS = ["mr", "mrr", "h1", "h3", "h10", "h100", "auc"]
UNSEEN_LABELS = {"LinkGDA-f": "func", "LinkGDA-fs": "func_expr"}

POOL = [f"G{i}" for i in range(6)]
UNSEEN_COLS = [2, 3, 4, 5]

SCORE_ROWS = (
    "GA\tDX\t3\t0.1\t0.2\t0.3\t0.9\t0.4\t0.5\n"
    "GB\tDY\t4\t0.5\t0.6\t0.1\t0.2\t0.9\t0.3\n"
)


def run_cli(script, args, cwd):
    return subprocess.run([sys.executable, str(script), *args],
                          cwd=str(cwd), capture_output=True, text=True, timeout=60,
                          env={**os.environ, "MPLCONFIGDIR": str(cwd / ".mplconfig")})


def build_benchmark(root):
    bench = root / "bench"
    (bench / "folds" / "fold_0").mkdir(parents=True)
    (bench / "gene_diseases.csv").write_text(
        "Gene,Disease\n" + "".join(f"{g},D{i}\n" for i, g in enumerate(POOL)))
    (bench / "folds" / "fold_0" / "train.csv").write_text(
        "Gene\tDisease\nG0\tD0\nG1\tD1\n")
    return bench


def excluded_filename(source, projection, setting, seed):
    hps = EXCLUDED_CELLS[(projection, setting)]
    tag = "_calsel" if setting == "calibrated" else ""
    return (f"kge_results_transd_fold_0_seed_{seed}_{hps}_{source}_"
            f"proj_{projection}_use_graph_True_tol_15{tag}_by_graph_bma.tsv")


def build_rank_cdf_workspace(root):
    data = root / "data"
    (data / "folds").mkdir(parents=True)
    (data / "results").mkdir(parents=True)
    (data / "disease_phenotypes.csv").write_text(
        "Disease,Phenotype\nD0,HP:0\nDX,HP:0\nDY,HP:1\n")
    for fold in range(10):
        fold_dir = data / "folds" / f"fold_{fold}"
        fold_dir.mkdir()
        (fold_dir / "train.csv").write_text("Gene\tDisease\nG0\tD0\nGA\tD0\n")
        (fold_dir / "test.csv").write_text("Gene\tDisease\nGA\tDX\nGB\tDY\n")
        for config in RANK_CDF_CONFIGS.values():
            (data / "results" / f"kge_results_transd_fold_{fold}_seed_0_{config}.tsv"
             ).write_text("GA\tDX\t0\t3.0\t1.0\t2.0\nGB\tDY\t1\t1.0\t3.0\t2.0\n")


@pytest.mark.parametrize("name", CLICK_CLIS)
def test_cli_help_from_repo_root(name):
    result = run_cli(ANALYSIS / name, ["--help"], REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


@pytest.mark.parametrize("name", CLICK_CLIS)
def test_cli_help_from_foreign_cwd(name, tmp_path):
    result = run_cli(ANALYSIS / name, ["--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_rank_cdf_median_runs_from_foreign_cwd(tmp_path):
    build_rank_cdf_workspace(tmp_path)
    result = run_cli(ANALYSIS / "rank_cdf_median.py", ["kge"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "SKIPPED" not in result.stdout
    for name in RANK_CDF_CONFIGS:
        assert f"=== {name} (n=20) ===" in result.stdout
    assert result.stdout.count("median  all=1.0  zero=1.0  some=1.0") == 4
    assert result.stdout.count("CDF-all  (1,100.0) (3,100.0)") == 4


def test_excluded_table_matches_rq1_table_from_foreign_cwd(tmp_path):
    build_benchmark(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    seeds = 2
    for source in EXCLUDED_SOURCES:
        for projection in EXCLUDED_PROJECTIONS:
            for setting in ("uncalibrated", "calibrated"):
                for seed in range(seeds):
                    (results / excluded_filename(source, projection, setting, seed)
                     ).write_text(SCORE_ROWS)
    out = tmp_path / "out.tsv"
    result = run_cli(ANALYSIS / "excluded_table.py",
                     ["--results", "results", "--benchmark", "bench",
                      "--seeds", str(seeds), "--out", "out.tsv"], tmp_path)
    assert result.returncode == 0, result.stderr
    rows = {tuple(r[:4]): r for r in list(csv.reader(out.open(), delimiter="\t"))[1:]}
    label_names = {"func": "LinkGDA-f", "func_expr": "LinkGDA-fs"}
    projection_names = {"owl2vecstar": "OWL2Vec*", "owl2vecstar_gda": "GDAProjector"}
    for source in EXCLUDED_SOURCES:
        for projection in EXCLUDED_PROJECTIONS:
            for setting in ("uncalibrated", "calibrated"):
                scored = []
                indices = []
                for seed in range(seeds):
                    scores, idx = load(results / excluded_filename(source, projection, setting, seed))
                    prepared = calibrate(scores) if setting == "calibrated" else scores
                    scored.append((prepared, prepared[:, UNSEEN_COLS]))
                    indices.append((idx, idx - 2))
                for pool in ("unseen", "full"):
                    j = 1 if pool == "unseen" else 0
                    expected = {key: summarise([metrics(m[j], i[j]) for m, i in
                                                zip(scored, indices)], key)
                                for key in EXCLUDED_KEYS}
                    row = rows[(label_names[source], projection_names[projection],
                                setting, pool)]
                    assert row[4] == ("4" if pool == "unseen" else "6")
                    assert row[5] == str(seeds)
                    for k, key in enumerate(EXCLUDED_KEYS):
                        assert float(row[6 + 2 * k]) == pytest.approx(expected[key][0], abs=1e-6)
                        assert float(row[7 + 2 * k]) == pytest.approx(expected[key][1], abs=1e-6)


def test_ultra_excluded_table_matches_rq1_table_from_foreign_cwd(tmp_path):
    build_benchmark(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    specs = [("func_expr", "owl2vecstar", "bma"), ("func_expr", "owl2vecstar_gda", "bmm")]
    for modality, projection, aggregation in specs:
        (results / f"kge_results_ultra4g_zeroshot_excluded_fold_0_seed_0_{modality}_"
                   f"proj_{projection}_use_graph_True_by_graph_{aggregation}.tsv"
         ).write_text(SCORE_ROWS)
    out = tmp_path / "out.tsv"
    result = run_cli(ANALYSIS / "ultra_excluded_table.py",
                     ["--results", "results", "--benchmark", "bench",
                      "--out", "out.tsv"], tmp_path)
    assert result.returncode == 0, result.stderr
    rows = {tuple(r[:5]): r for r in list(csv.reader(out.open(), delimiter="\t"))[1:]}
    for modality, projection, aggregation in specs:
        scores, idx = load(results / f"kge_results_ultra4g_zeroshot_excluded_fold_0_seed_0_{modality}_"
                                    f"proj_{projection}_use_graph_True_by_graph_{aggregation}.tsv")
        for setting, prepared in (("raw", scores), ("calibrated", calibrate(scores))):
            for pool, (s, i) in (("unseen", (prepared[:, UNSEEN_COLS], idx - 2)),
                                 ("full", (prepared, idx))):
                expected = metrics(s, i)
                row = rows[(modality, projection, aggregation, setting, pool)]
                assert row[5] == ("4" if pool == "unseen" else "6")
                assert row[6] == "2"
                for k, key in enumerate(EXCLUDED_KEYS):
                    assert float(row[7 + k]) == pytest.approx(expected[key], abs=1e-6)


def test_figure_generator_writes_only_from_cwd(tmp_path):
    source = (FIGURES / "make_rankcdf_fig.py").read_text()
    assert 'savefig("paper/fig/fig_rankcdf_two.pdf")' in source
    assert 'savefig("paper/fig/fig_rankcdf_single.pdf")' in source
    assert "argparse" not in source and "click" not in source
    (tmp_path / "paper" / "fig").mkdir(parents=True)
    result = run_cli(FIGURES / "make_rankcdf_fig.py", [], tmp_path)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "paper" / "fig" / "fig_rankcdf_two.pdf").is_file()
    assert (tmp_path / "paper" / "fig" / "fig_rankcdf_single.pdf").is_file()


def test_old_root_paths_gone_and_production_files_in_place():
    for name in [*CLICK_CLIS, "make_rankcdf_fig.py"]:
        assert not (REPO_ROOT / name).exists(), name
    for name in ["code/link_gda/data.py", "code/link_gda/evaluation.py",
                 "code/link_gda/evaluate_sem_sim.py", "code/link_gda/pykeen_utils.py",
                 "code/link_gda/negative_sampling.py",
                 "code/analysis/rq1_table.py", "code/analysis/calibrate_scores.py",
                 "code/analysis/graph_statistics.py", "code/analysis/rq1_stats.py",
                 "code/analysis/rq2_stats.py", "code/training/kge_transd.py",
                 "code/training/kge_convkb_d.py", "code/figures/make_calibration_fig.py"]:
        assert (REPO_ROOT / name).is_file(), name
    for name in ["rq1_table.py", "calibrate_scores.py", "graph_statistics.py",
                 "kge_transd.py", "kge_convkb_d.py", "data.py", "evaluation.py",
                 "evaluate_sem_sim.py", "pykeen_utils.py", "negative_sampling.py",
                 "analysis/rq1_stats.py", "analysis/rq2_stats.py"]:
        assert not (REPO_ROOT / name).exists(), name
