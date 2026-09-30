"""Layout checks for the batch-2 migration.

The data-preparation pipeline moves into code/data/ and the remaining
root analysis scripts into code/analysis/. Every CLI must answer
--help with no import error both from the repository root and from a
foreign working directory, generate_folds.py must still write a
deterministic disease-disjoint 10-fold split from a small fixture, and
build_excluded_benchmark.py must still apply both leakage filters on a
small fixture. Scripts that need full annotation or prediction inputs are covered separately
by the recorded source comparison, rather than executed on real data here.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = REPO_ROOT / "code" / "analysis"
DATA = REPO_ROOT / "code" / "data"

HELP_CLIS = [
    ANALYSIS / "leakage_overlap.py",
    ANALYSIS / "leakage_overlap_perfold.py",
    ANALYSIS / "popularity_controls.py",
    ANALYSIS / "strata_from_labels.py",
    ANALYSIS / "stratified_metrics.py",
    DATA / "build_excluded_benchmark.py",
]

def run_cli(script, args, cwd, timeout=60):
    return subprocess.run([sys.executable, str(script), *args],
                          cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout,
                          env={**os.environ, "MPLCONFIGDIR": str(cwd / ".mplconfig")})


@pytest.mark.parametrize("script", HELP_CLIS)
def test_cli_help_from_repo_root(script):
    result = run_cli(script, ["--help"], REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


@pytest.mark.parametrize("script", HELP_CLIS)
def test_cli_help_from_foreign_cwd(script, tmp_path):
    result = run_cli(script, ["--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_build_association_files_help_from_foreign_cwd(tmp_path):
    result = run_cli(DATA / "build_association_files.py", ["--help"], tmp_path, timeout=180)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


FOLD_DISEASES = 12
FOLD_COUNT = 10
FOLD_TEST_SIZES = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]


def write_fold_workspace(root):
    (root / "data").mkdir(parents=True, exist_ok=True)
    lines = ["Gene,Disease"]
    for i in range(FOLD_DISEASES):
        genes = [f"G{i}a", f"G{i}b"] if i != 5 else [f"G{i}a"]
        lines += [f"{g},D{i}" for g in genes]
    (root / "data" / "gene_diseases.csv").write_text("\n".join(lines) + "\n")


def fold_file(root, fold, kind):
    lines = (root / "data" / "folds" / f"fold_{fold}" / f"{kind}.csv").read_text().splitlines()
    assert lines[0] == "Gene\tDisease"
    return [tuple(line.split("\t")) for line in lines[1:]]


def fold_workspace_bytes(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted((root / "data" / "folds").rglob("*.csv"))}


def test_generate_folds_deterministic_disease_disjoint(tmp_path):
    write_fold_workspace(tmp_path)
    result = run_cli(DATA / "generate_folds.py", [], tmp_path)
    assert result.returncode == 0, result.stderr
    first = fold_workspace_bytes(tmp_path)

    result = run_cli(DATA / "generate_folds.py", [], tmp_path)
    assert result.returncode == 0, result.stderr
    assert fold_workspace_bytes(tmp_path) == first

    input_pairs = set()
    for line in (tmp_path / "data" / "gene_diseases.csv").read_text().splitlines()[1:]:
        gene, disease = line.split(",")
        input_pairs.add((gene, disease))

    test_diseases = set()
    for fold in range(FOLD_COUNT):
        train = fold_file(tmp_path, fold, "train")
        test = fold_file(tmp_path, fold, "test")
        train_pairs, test_pairs = set(train), set(test)
        fold_test_diseases = {disease for _, disease in test_pairs}
        assert len(fold_test_diseases) == FOLD_TEST_SIZES[fold]
        assert not train_pairs & test_pairs
        assert not test_diseases & fold_test_diseases
        assert not fold_test_diseases & {disease for _, disease in train_pairs}
        test_diseases |= fold_test_diseases

    assert test_diseases == {disease for _, disease in input_pairs}
    for fold in range(FOLD_COUNT):
        test = set(fold_file(tmp_path, fold, "test"))
        train = set(fold_file(tmp_path, fold, "train"))
        assert test | train == input_pairs
        assert not test & train


def test_build_excluded_benchmark_fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "genes_to_disease.txt").write_text(
        "gene_id\tsymbol\tassociation_type\tdisease_id\tsource\n"
        + "".join(f"{gene}\t{symbol}\tassoc\t{disease}\tsrc\n" for gene, symbol, disease in [
            ("NCBI:111", "GS1", "OMIM:111111"),
            ("NCBI:222", "GS2", "OMIM:222222"),
            ("NCBI:222", "GS2", "OMIM:111111"),
            ("NCBI:333", "GS3", "OMIM:333333"),
            ("NCBI:222", "GS2", "OMIM:444444"),
            ("NCBI:222", "GS2", "OMIM:555555"),
        ]))
    (data / "gene_phenotypes.csv").write_text("Gene,Phenotype\nhttp://mowl.borg/111,MP:0000001\n")
    (data / "gene_functions.csv").write_text("Gene,Function\nhttp://mowl.borg/222,GO:0000001\n")
    (data / "disease_phenotypes.csv").write_text(
        "Disease,Phenotype\n"
        "http://mowl.borg/OMIM_111111,HP:0000001\n"
        "http://mowl.borg/OMIM_444444,HP:0000001\n"
        "http://mowl.borg/OMIM_555555,HP:0000002\n")
    (data / "gene_diseases.csv").write_text(
        "Gene,Disease\nhttp://mowl.borg/111,http://mowl.borg/OMIM_111111\n")

    out = tmp_path / "out"
    result = run_cli(DATA / "build_excluded_benchmark.py",
                     ["--data-dir", "data", "--out-dir", "out"], tmp_path)
    assert result.returncode == 0, result.stderr

    funnel = (out / "funnel.txt").read_text()
    assert "all pairs in genes_to_disease.txt                  :       6" in funnel
    assert "main benchmark, after both filters               :       1" in funnel
    assert "and phenotype profile not near-duplicate       :       1  <- test set" in funnel

    fold_dir = out / "data" / "folds" / "fold_0"
    assert fold_file_with_header(fold_dir / "train.csv") == [
        ("http://mowl.borg/111", "http://mowl.borg/OMIM_111111")]
    assert fold_file_with_header(fold_dir / "test.csv") == [
        ("http://mowl.borg/222", "http://mowl.borg/OMIM_555555")]

    pool = (out / "data" / "gene_diseases.csv").read_text().splitlines()
    assert pool[0] == "Gene,Disease"
    assert set(pool[1:]) == {"http://mowl.borg/111,http://mowl.borg/OMIM_111111",
                             "http://mowl.borg/222,http://mowl.borg/OMIM_555555"}
    assert (out / "disease_leakage.csv").is_file()


def fold_file_with_header(path):
    lines = path.read_text().splitlines()
    assert lines[0] == "Gene\tDisease"
    return [tuple(line.split("\t")) for line in lines[1:]]
