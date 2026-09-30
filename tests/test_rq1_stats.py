"""Fixed-fixture tests for analysis/rq1_stats.py: parsing, pairing, statistics,
input rejection, and a separately characterized known issue in rq1_table's
fold-pairing.
"""
import hashlib
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def load_rq1_stats():
    spec = importlib.util.spec_from_file_location("rq1_stats_under_test", ROOT / "analysis" / "rq1_stats.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FIXED_TSV = "g0\td0\t1\t0.1\t0.9\t0.5\ng1\td1\t2\t0.2\t0.2\t0.8\n"


class FixedScoreTsvTests(unittest.TestCase):
    def test_parses_to_expected_matrix_indices_keys_and_digest(self):
        rq1_stats = load_rq1_stats()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.tsv"
            path.write_text(FIXED_TSV)
            matrix, indices, keys, digest = rq1_stats.load(path)
        np.testing.assert_allclose(matrix, np.array([[0.1, 0.9, 0.5], [0.2, 0.2, 0.8]]))
        self.assertEqual(indices.tolist(), [1, 2])
        self.assertEqual(keys, [("g0", "d0"), ("g1", "d1")])
        self.assertEqual(digest, hashlib.sha256(FIXED_TSV.encode()).hexdigest())


class MeanRankTests(unittest.TestCase):
    def test_hand_computed_with_tie(self):
        rq1_stats = load_rq1_stats()
        scores = np.array(
            [[0.9, 0.5, 0.2, 0.1], [0.1, 0.2, 0.5, 0.9], [0.5, 0.5, 0.5, 0.5]]
        )
        self.assertAlmostEqual(rq1_stats.mean_rank(scores, np.array([0, 3, 0])), 1.5)


class CorrectedTestTests(unittest.TestCase):
    def test_matches_hand_computed_statistics(self):
        rq1_stats = load_rq1_stats()
        linkgda = np.array([1.0, 1.0, 1.0])
        indigena = np.array([2.0, 2.0, 3.0])
        result = rq1_stats.corrected_test(indigena, linkgda)
        mean = 4.0 / 3.0
        sd = math.sqrt(1.0 / 3.0)
        se = sd * math.sqrt(5.0 / 6.0)
        t = mean / se
        self.assertAlmostEqual(result["mean_difference"], mean)
        self.assertAlmostEqual(result["sample_sd_difference"], sd)
        self.assertAlmostEqual(result["correction_factor"], 5.0 / 6.0)
        self.assertAlmostEqual(result["corrected_standard_error"], se)
        self.assertAlmostEqual(result["t"], t)
        self.assertEqual(result["df"], 2)
        self.assertAlmostEqual(result["p_one_sided"], float(stats.t.sf(t, 2)))
        self.assertAlmostEqual(result["cohens_dz"], mean / sd)
        self.assertEqual(result["folds_linkgda_better"], 3)
        self.assertEqual(result["folds_tied"], 0)


class CliRejectionTests(unittest.TestCase):
    def write_all(self, results):
        rq1_stats = load_rq1_stats()
        for methods in rq1_stats.CONFIGS.values():
            for template in methods.values():
                for fold in range(10):
                    path = results / template.format(fold=fold)
                    path.write_text("g0\td0\t0\t0.3\t0.4\t0.5\n")

    def test_cli_rejects_missing_input_file(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            results.mkdir()
            self.write_all(results)
            rq1_stats = load_rq1_stats()
            victim = next(iter(rq1_stats.CONFIGS["owl2vecstar"].values())).format(fold=3)
            (results / victim).unlink()
            completed = subprocess.run(
                [sys.executable, str(ROOT / "analysis" / "rq1_stats.py"), "--results-dir", str(results)],
                check=False, capture_output=True, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("FileNotFoundError", completed.stderr)


class MissingFoldPairingKnownIssueTests(unittest.TestCase):
    """KNOWN ISSUE (characterized, not endorsed).

    rq1_table.over_folds silently skips folds whose file is absent, and main()
    silently drops a method from the paired comparison whenever its number of
    available folds differs from the reference's. A 9-fold method next to a
    10-fold reference produces no error and no warning; the missing fold is
    invisible in the output. This test pins the current behavior so that a
    later fix is a visible, deliberate change.
    """

    def test_over_folds_silently_skips_missing_fold_files(self):
        import rq1_table

        with tempfile.TemporaryDirectory() as directory:
            for fold in range(10):
                path = Path(directory) / f"fold_{fold}.tsv"
                if fold != 7:
                    path.write_text("g0\td0\t0\t0.3\t0.4\t0.5\n")
            rows = rq1_table.over_folds(str(Path(directory) / "fold_{f}.tsv"), False)
            self.assertEqual(len(rows), 9)

    def test_main_silently_drops_mismatched_method_from_paired_test(self):
        from click.testing import CliRunner

        import rq1_table

        with tempfile.TemporaryDirectory() as directory:
            dir_full, dir_short = Path(directory) / "full", Path(directory) / "short"
            dir_full.mkdir()
            dir_short.mkdir()
            for fold in range(10):
                (dir_full / f"ref_{fold}.tsv").write_text("g0\td0\t0\t0.3\t0.4\t0.5\n")
            for fold in range(9):
                (dir_short / f"m_{fold}.tsv").write_text("g0\td0\t0\t0.3\t0.4\t0.5\n")
            spec = Path(directory) / "spec.tsv"
            spec.write_text(
                "ref\tlearned\tref_{f}.tsv\tref_{f}.tsv\n"
                f"m9\tlearned\t{dir_short / 'm_{{f}}.tsv'}\t{dir_short / 'm_{{f}}.tsv'}\n"
            )
            runner = CliRunner()
            result = runner.invoke(rq1_table.main, ["--spec", str(spec), "--reference", "ref"])
            self.assertEqual(result.exit_code, 0, result.output)
            paired = result.output.split("paired t-test over folds")[1]
            self.assertNotIn("m9", paired)


if __name__ == "__main__":
    unittest.main()
