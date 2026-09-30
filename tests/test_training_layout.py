"""Layout checks for the batch-3 trainer and shared-module migration.

The trainers boot the full link_gda stack (JVM, torch, PyKEEN) on import, but
--help exits before any training, data loading, or file writing. Both trainers
must answer --help from the repository root and from a foreign working
directory. The moved analysis CLIs must answer --help from a foreign working
directory. Every subprocess is bounded, so a hung import fails the test
instead of blocking the suite; nothing here trains a model.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 120

TRAINERS = ["code/training/kge_transd.py", "code/training/kge_convkb_d.py"]
ANALYSIS_CLIS = [
    "code/analysis/rq1_table.py",
    "code/analysis/calibrate_scores.py",
    "code/analysis/graph_statistics.py",
    "code/analysis/rq1_stats.py",
    "code/analysis/rq2_stats.py",
]


def run_help(script, cwd):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), "--help"],
        check=False, capture_output=True, text=True,
        cwd=str(cwd), timeout=TIMEOUT,
    )


class TrainerHelpLayoutTests(unittest.TestCase):
    def test_trainers_answer_help_from_root(self):
        for script in TRAINERS:
            with self.subTest(script=script):
                completed = run_help(script, REPO_ROOT)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage", completed.stdout.lower())

    def test_trainers_answer_help_from_foreign_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            for script in TRAINERS:
                with self.subTest(script=script):
                    completed = run_help(script, directory)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn("usage", completed.stdout.lower())


class AnalysisCliHelpLayoutTests(unittest.TestCase):
    def test_analysis_clis_answer_help_from_foreign_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            for script in ANALYSIS_CLIS:
                with self.subTest(script=script):
                    completed = run_help(script, directory)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn("usage", completed.stdout.lower())


if __name__ == "__main__":
    unittest.main()
