import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import scipy
    SCIPY_AVAILABLE = True
except ModuleNotFoundError:
    SCIPY_AVAILABLE = False


CONFIG_TEMPLATES = {
    "owl2vecstar": {
        "linkgda": "kge_results_transd_fold_{fold}_seed_0_dim_200_bs_65536_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_True_tol_15_calsel_by_graph_bma.tsv",
        "indigena": "kge_results_transd_fold_{fold}_seed_0_dim_400_bs_65536_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_tol_15_calsel_inductive_bma.tsv",
    },
    "gda_projector": {
        "linkgda": "kge_results_transd_fold_{fold}_seed_0_dim_200_bs_65536_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_tol_15_calsel_by_graph_bma.tsv",
        "indigena": "kge_results_transd_fold_{fold}_seed_0_dim_400_bs_65536_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_False_tol_15_calsel_inductive_bma.tsv",
    },
}


def write_fixture(results):
    for projection in CONFIG_TEMPLATES.values():
        for method, template in projection.items():
            for fold in range(10):
                path = results / template.format(fold=fold)
                ind_rank_three = fold % 2 == 1
                ind_rows = (
                    ["0 1 2", "0 1.5 2.5", "0 2 1"]
                    if ind_rank_three
                    else ["1 2 0", "0.5 2 1", "1 2 0"]
                )
                score_rows = (
                    ind_rows
                    if method == "indigena"
                    else ["3 1 0", "2.5 0.5 1", "0.5 2 1"]
                )
                path.write_text(
                    "".join(
                        f"g{i}\td{i}\t0\t{scores.replace(' ', chr(9))}\n"
                        for i, scores in enumerate(score_rows)
                    )
                )


class RQ1StatsCliTests(unittest.TestCase):
    @unittest.skipUnless(SCIPY_AVAILABLE, "run in the pinned scientific environment")
    def test_cli_reads_aligned_files_and_computes_corrected_test(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            results.mkdir()
            write_fixture(results)

            completed = subprocess.run(
                [
                    sys.executable,
                    "code/analysis/rq1_stats.py",
                    "--results-dir",
                    str(results),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode:
                self.fail(f"calculator failed:\n{completed.stdout}\n{completed.stderr}")
            output = json.loads(completed.stdout)
            self.assertEqual(len(output["manifest"]), 40)
            self.assertEqual(output["protocol"]["rank_ties"], "deterministic average rank")
            expected_link = [2.0] * 10
            expected_ind = [5.0 / 3.0 if fold % 2 == 0 else 13.0 / 6.0 for fold in range(10)]
            expected_diff = [ind - link for ind, link in zip(expected_ind, expected_link)]
            expected_mean = sum(expected_diff) / len(expected_diff)
            expected_sd = math.sqrt(
                sum((value - expected_mean) ** 2 for value in expected_diff) / (len(expected_diff) - 1)
            )
            expected_se = expected_sd * math.sqrt(1.0 / 10.0 + 1.0 / 9.0)
            expected_t = expected_mean / expected_se
            expected_p = scipy.stats.t.sf(expected_t, 9)
            critical = scipy.stats.t.ppf(0.975, 9)
            expected_ci = (expected_mean - critical * expected_se, expected_mean + critical * expected_se)
            expected_dz = expected_mean / expected_sd
            for projection in ("owl2vecstar", "gda_projector"):
                folds = output["projections"][projection]["fold_mr"]
                self.assertEqual(folds["linkgda"], expected_link)
                for actual, expected in zip(folds["indigena"], expected_ind):
                    self.assertAlmostEqual(actual, expected)
                test = output["projections"][projection]["test"]
                self.assertEqual(test["k"], 10)
                self.assertAlmostEqual(test["mean_difference"], expected_mean)
                self.assertAlmostEqual(test["sample_sd_difference"], expected_sd)
                self.assertAlmostEqual(test["corrected_standard_error"], expected_se)
                self.assertAlmostEqual(test["p_one_sided"], expected_p)
                self.assertAlmostEqual(test["ci_95"][0], expected_ci[0])
                self.assertAlmostEqual(test["ci_95"][1], expected_ci[1])
                self.assertAlmostEqual(test["cohens_dz"], expected_dz)

    @unittest.skipUnless(SCIPY_AVAILABLE, "run in the pinned scientific environment")
    def test_cli_rejects_query_order_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            results.mkdir()
            write_fixture(results)
            path = results / CONFIG_TEMPLATES["owl2vecstar"]["linkgda"].format(fold=0)
            path.write_text(path.read_text().replace("g0\td0\t", "g0\tdX\t", 1))
            completed = subprocess.run(
                [sys.executable, "code/analysis/rq1_stats.py", "--results-dir", str(results)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("query order differs", completed.stderr)


if __name__ == "__main__":
    unittest.main()
