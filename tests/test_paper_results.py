"""Regression tests for code/reproduce/paper_results.py on small synthetic TSVs.

The tests never touch the real 30 prediction files; they exercise the same
validation, alignment, and display-check helpers with parameterized small
fold/seed counts, while the production CLI keeps the exact 10-fold/10-seed
contract.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "paper_results_under_test", ROOT / "code" / "reproduce" / "paper_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tsv(path, rows):
    path.write_text("".join(
        "\t".join([gene, disease, str(true_idx)]) + "".join(f"\t{score}" for score in scores) + "\n"
        for gene, disease, true_idx, scores in rows
    ))


def keys_for(n_pairs, tag):
    return [(f"g{i}_{tag}", f"d{i}_{tag}") for i in range(n_pairs)]


LINK_ROWS = [
    [10, 1, 1, 1, 1],
    [10, 2, 1, 1, 1],
    [10, 1, 2, 1, 1],
]
ALT_LINK_ROWS = [
    [10, 1, 2, 1, 1],
    [10, 2, 1, 1, 1],
    [10, 1, 1, 3, 1],
]
IND_ROWS = [
    [9, 10, 1, 1, 1],
    [10, 2, 1, 1, 1],
    [10, 1, 2, 1, 1],
]
ALT_IND_ROWS = [
    [9, 10, 1, 1, 1],
    [9, 2, 1, 1, 1],
    [10, 1, 2, 1, 1],
]
EXCLUDED_ROWS = [[1, 2, 3, 0, 0], [1, 2, 4, 0, 0], [1, 3, 2, 0, 0]]
ALT_EXCLUDED_ROWS = [[1, 2, 3, 4, 0], [1, 2, 4, 3, 0], [1, 3, 2, 5, 0]]


def write_rq1_files(directory, n_folds=2, n_pairs=3):
    checker = load_checker()
    templates = dict(checker.rq1_stats.CONFIGS["owl2vecstar"])
    for fold in range(n_folds):
        keys = keys_for(n_pairs, f"fold{fold}")
        link = LINK_ROWS if fold % 2 == 0 else ALT_LINK_ROWS
        ind = IND_ROWS if fold % 2 == 0 else ALT_IND_ROWS
        write_tsv(directory / templates["linkgda"].format(fold=fold),
                  [(g, d, 0, link[i]) for i, (g, d) in enumerate(keys)])
        write_tsv(directory / templates["indigena"].format(fold=fold),
                  [(g, d, 0, ind[i]) for i, (g, d) in enumerate(keys)])


def mr_vector(checker, directory, template, count, fmt_key):
    return [
        checker.load_validated(directory / template.format(**{fmt_key: value}))["mr"]
        for value in range(count)
    ]


def rq1_expected(directory, n_folds=2, n_pairs=3, mr_mean_override=None):
    checker = load_checker()
    import numpy as np
    from scipy import stats as scipy_stats

    templates = dict(checker.rq1_stats.CONFIGS["owl2vecstar"])
    link = mr_vector(checker, directory, templates["linkgda"], n_folds, "fold")
    ind = mr_vector(checker, directory, templates["indigena"], n_folds, "fold")
    diff = np.array(ind) - np.array(link)
    se = diff.std(ddof=1) * (1.0 / n_folds + 1.0 / (n_folds - 1)) ** 0.5
    p = float(scipy_stats.t.sf(diff.mean() / se, n_folds - 1))
    expected = {
        "source": "synthetic fixture, not a paper value",
        "candidate_pool": 5,
        "total_pairs": n_folds * n_pairs,
        "methods": {
            "linkgda": {"mr_mean_display": checker.display(float(np.mean(link)), 2),
                        "mr_sd_display": checker.display(float(np.std(link, ddof=1)), 2)},
            "indigena": {"mr_mean_display": checker.display(float(np.mean(ind)), 2),
                         "mr_sd_display": checker.display(float(np.std(ind, ddof=1)), 2)},
        },
        "p_raw_display": checker.display(p, 4),
        "p_bonferroni_display": checker.display(min(1.0, 6.0 * p), 4),
    }
    if mr_mean_override is not None:
        expected["methods"]["linkgda"]["mr_mean_display"] = mr_mean_override
    return expected


def write_excluded_files(directory, n_seeds=2, n_pairs=3):
    checker = load_checker()
    for seed in range(n_seeds):
        keys = keys_for(n_pairs, "q")
        rows = EXCLUDED_ROWS if seed % 2 == 0 else ALT_EXCLUDED_ROWS
        write_tsv(directory / checker.EXCLUDED_TEMPLATE.format(seed=seed),
                  [(g, d, 0, rows[i]) for i, (g, d) in enumerate(keys)])


def excluded_expected(directory, n_seeds=2):
    checker = load_checker()
    import numpy as np

    mrs = [checker.load_validated(directory / checker.EXCLUDED_TEMPLATE.format(seed=seed))["mr"]
           for seed in range(n_seeds)]
    h10s = [checker.load_validated(directory / checker.EXCLUDED_TEMPLATE.format(seed=seed))["h10"]
            for seed in range(n_seeds)]
    return {
        "source": "synthetic fixture, not a paper value",
        "pairs": 3,
        "unique_genes": 3,
        "unique_diseases": 3,
        "candidate_pool": 5,
        "mr_mean_display": checker.display(float(np.mean(mrs)), 2),
        "mr_sd_display": checker.display(float(np.std(mrs, ddof=1)), 2),
        "h10_mean_display": checker.display(float(np.mean(h10s)), 2),
        "h10_sd_display": checker.display(float(np.std(h10s, ddof=1)), 2),
    }


class DisplayRoundingTests(unittest.TestCase):
    def test_paper_values_round_to_printed_displays(self):
        checker = load_checker()
        self.assertEqual(checker.display(601.5953908996693, 2), 601.60)
        self.assertEqual(checker.display(37.49002856919921, 2), 37.49)
        self.assertEqual(checker.display(673.2069245007872, 2), 673.21)
        self.assertEqual(checker.display(58.13250196688708, 2), 58.13)
        self.assertEqual(checker.display(0.004267755316762704, 4), 0.0043)
        self.assertEqual(checker.display(0.02560653190057623, 4), 0.0256)
        self.assertEqual(checker.display(1038.16, 2), 1038.16)
        self.assertEqual(checker.display(54.44, 2), 54.44)
        self.assertEqual(checker.display(0.074, 2), 0.07)

    def test_default_expected_constants_match_declared_paper_values(self):
        checker = load_checker()
        rq1 = checker.DEFAULT_EXPECTED["rq1"]
        self.assertEqual(rq1["candidate_pool"], 4399)
        self.assertEqual(rq1["total_pairs"], 6571)
        self.assertEqual(rq1["methods"]["linkgda"], {"mr_mean_display": 601.60, "mr_sd_display": 37.49})
        self.assertEqual(rq1["methods"]["indigena"], {"mr_mean_display": 673.21, "mr_sd_display": 58.13})
        self.assertEqual(rq1["p_raw_display"], 0.0043)
        self.assertEqual(rq1["p_bonferroni_display"], 0.0256)
        excluded = checker.DEFAULT_EXPECTED["excluded"]
        self.assertEqual(excluded["pairs"], 409)
        self.assertEqual(excluded["unique_genes"], 350)
        self.assertEqual(excluded["unique_diseases"], 402)
        self.assertEqual(excluded["candidate_pool"], 4749)
        self.assertEqual(excluded["mr_mean_display"], 1038.16)
        self.assertEqual(excluded["mr_sd_display"], 54.44)
        self.assertEqual(excluded["h10_mean_display"], 0.07)
        self.assertEqual(excluded["h10_sd_display"], 0.01)


class Rq1CheckTests(unittest.TestCase):
    def test_matching_synthetic_files_pass(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_rq1_files(directory)
            report = checker.run_rq1(directory, n_folds=2, expected=rq1_expected(directory),
                                     precision_reference="none")
        self.assertEqual(report["status"], "pass", report["errors"])
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["files"]), 4)
        for item in report["files"]:
            self.assertEqual(item["pairs"], 3)
            self.assertEqual(item["candidates"], 5)
            self.assertEqual(len(item["sha256"]), 64)
        self.assertTrue(all(row["case_keys_match"] and row["true_indices_match"]
                            for row in report["query_alignment"]))
        self.assertTrue(all(item["ok"] for item in report["comparisons"]))

    def test_mismatched_value_fails_with_nonzero_exit(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_rq1_files(directory)
            report = checker.run_rq1(directory, n_folds=2,
                                     expected=rq1_expected(directory, mr_mean_override=999.99))
        self.assertEqual(report["status"], "fail")
        self.assertEqual(checker.build_report({"rq1": report})["exit_code"], 1)
        self.assertTrue(any("rq1.linkgda.mr_mean" in error for error in report["errors"]))

    def test_missing_file_is_unverified_not_pass(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_rq1_files(directory)
            expected = rq1_expected(directory)
            victim = directory / checker.rq1_stats.CONFIGS["owl2vecstar"]["indigena"].format(fold=1)
            victim.unlink()
            report = checker.run_rq1(directory, n_folds=2, expected=expected)
        self.assertEqual(report["status"], "unverified")
        self.assertEqual(checker.build_report({"rq1": report})["exit_code"], 2)
        self.assertNotEqual(report["status"], "pass")

    def test_altered_query_pairing_is_rejected(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_rq1_files(directory)
            expected = rq1_expected(directory)
            victim = directory / checker.rq1_stats.CONFIGS["owl2vecstar"]["indigena"].format(fold=0)
            victim.write_text(victim.read_text().replace("d0_fold0", "dX_fold0", 1))
            report = checker.run_rq1(directory, n_folds=2, expected=expected)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("fold 0" in error and "differ" in error for error in report["errors"]))

    def test_duplicate_case_keys_are_rejected(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_rq1_files(directory)
            expected = rq1_expected(directory)
            victim = directory / checker.rq1_stats.CONFIGS["owl2vecstar"]["linkgda"].format(fold=0)
            victim.write_text(victim.read_text().replace("g1_fold0\td1_fold0", "g0_fold0\td0_fold0", 1))
            report = checker.run_rq1(directory, n_folds=2, expected=expected)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("duplicate case keys" in error for error in report["errors"]))


class LoaderRejectionTests(unittest.TestCase):
    def setUp(self):
        self.checker = load_checker()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "scores.tsv"

    def load(self, text):
        self.path.write_text(text)
        return self.checker.load_validated(self.path)

    def test_rejects_malformed_row(self):
        with self.assertRaisesRegex(ValueError, "malformed row"):
            self.load("g0\td0\t0\n")

    def test_rejects_non_integer_true_index(self):
        with self.assertRaisesRegex(ValueError, "not an integer"):
            self.load("g0\td0\tx\t0.1\t0.2\t0.3\ng1\td1\t0\t0.2\t0.1\t0.3\n")

    def test_rejects_non_finite_score(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            self.load("g0\td0\t0\tnan\t0.2\t0.3\ng1\td1\t0\t0.2\t0.1\t0.3\n")

    def test_rejects_non_numeric_score(self):
        with self.assertRaisesRegex(ValueError, "not a number"):
            self.load("g0\td0\t0\t0.1\t0.2\t0.3\ng1\td1\t0\t0.2\t0.1\tzzz\n")

    def test_rejects_unequal_candidate_counts(self):
        with self.assertRaisesRegex(ValueError, "unequal candidate counts"):
            self.load("g0\td0\t0\t0.1\t0.2\t0.3\ng1\td1\t0\t0.2\t0.1\n")

    def test_rejects_out_of_range_true_index(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.load("g0\td0\t5\t0.1\t0.2\t0.3\ng1\td1\t0\t0.2\t0.1\t0.3\n")

    def test_rejects_single_row_file(self):
        with self.assertRaisesRegex(ValueError, "at least 2 rows"):
            self.load("g0\td0\t0\t0.1\t0.2\t0.3\n")

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.checker.load_validated(Path(self.tmp.name) / "absent.tsv")


class ExcludedCheckTests(unittest.TestCase):
    def test_matching_synthetic_files_pass(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_excluded_files(directory)
            report = checker.run_excluded(directory, n_seeds=2, expected=excluded_expected(directory))
        self.assertEqual(report["status"], "pass", report["errors"])
        self.assertEqual(len(report["files"]), 2)
        self.assertTrue(all(item["ok"] for item in report["comparisons"]))

    def test_seed_pairing_mismatch_fails(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_excluded_files(directory)
            victim = directory / checker.EXCLUDED_TEMPLATE.format(seed=1)
            victim.write_text(victim.read_text().replace("d1_q", "dX_q", 1))
            report = checker.run_excluded(directory, n_seeds=2, expected=excluded_expected(directory))
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("seed 1" in error for error in report["errors"]))

    def test_missing_seed_file_is_unverified(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_excluded_files(directory)
            expected = excluded_expected(directory)
            (directory / checker.EXCLUDED_TEMPLATE.format(seed=1)).unlink()
            report = checker.run_excluded(directory, n_seeds=2, expected=expected)
        self.assertEqual(report["status"], "unverified")
        self.assertEqual(checker.build_report({"excluded": report})["exit_code"], 2)

    def test_wrong_structure_fails(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_excluded_files(directory)
            expected = excluded_expected(directory)
            expected["pairs"] = 999
            report = checker.run_excluded(directory, n_seeds=2, expected=expected)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("pairs is 3" in error for error in report["errors"]))


class CliContractTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "code" / "reproduce" / "paper_results.py"), *args],
            check=False, capture_output=True, text=True, cwd=ROOT,
        )

    def test_rq1_required_with_check_rq1(self):
        completed = self.run_cli("--check", "rq1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--rq1-results", completed.stderr)

    def test_excluded_required_with_check_all(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_rq1_files(directory)
            completed = self.run_cli("--check", "all", "--rq1-results", str(directory),
                                     "--expected", str(ROOT / "nope.json"))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--excluded-results", completed.stderr)

    def test_full_run_writes_report_and_exits_by_status(self):
        checker = load_checker()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rq1_dir = directory / "rq1"
            excluded_dir = directory / "excluded"
            rq1_dir.mkdir()
            excluded_dir.mkdir()
            write_rq1_files(rq1_dir, n_folds=10)
            write_excluded_files(excluded_dir, n_seeds=10)
            overrides = {"rq1": rq1_expected(rq1_dir, n_folds=10),
                         "excluded": excluded_expected(excluded_dir, n_seeds=10)}
            overrides_path = directory / "expected.json"
            overrides_path.write_text(json.dumps(overrides))
            report_path = directory / "report.json"
            completed = self.run_cli(
                "--check", "all",
                "--rq1-results", str(rq1_dir),
                "--excluded-results", str(excluded_dir),
                "--expected", str(overrides_path),
                "--precision-reference", "none",
                "--output", str(report_path),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(report_path.read_text())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["scientific_scope"], checker.SCIENTIFIC_SCOPE)
        self.assertEqual(len(report["checks"]["rq1"]["files"]), 20)
        self.assertEqual(len(report["checks"]["excluded"]["files"]), 10)
        self.assertIn("rq1_table", report["source_hashes"])
        self.assertIn("analysis.rq1_stats", report["source_hashes"])


if __name__ == "__main__":
    unittest.main()
