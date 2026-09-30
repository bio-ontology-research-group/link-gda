import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "code" / "projector" / "audit_go_projection.py"
SPEC = importlib.util.spec_from_file_location("audit_go_projection", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ComparisonTests(unittest.TestCase):
    def test_reports_set_and_multiset_differences(self):
        existing = Counter({"a\tr\tb": 2, "c\tr\td": 1})
        projected = Counter({"a\tr\tb": 1, "c\tr\td": 2})

        result = AUDIT.comparison(existing, projected, 3)

        self.assertTrue(result["exact_set_equality"])
        self.assertFalse(result["exact_multiset_equality"])
        self.assertEqual(result["intersection"], 2)
        self.assertEqual(result["only_in_existing"], 0)
        self.assertEqual(result["only_in_new"], 0)
        self.assertEqual(result["row_count_intersection"], 2)
        self.assertEqual(result["row_count_only_in_existing"], 1)
        self.assertEqual(result["row_count_only_in_new"], 1)


class CliGuardTests(unittest.TestCase):
    def test_rejects_missing_warmup_before_importing_mowl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for name in ("go.owl", "go-plus.owl", "existing.tsv"):
                path = root / name
                path.touch()
                inputs.append(path)
            missing = root / "missing.owl"

            with self.assertRaises(FileNotFoundError) as context:
                AUDIT.main([
                    "--go", str(inputs[0]), "--go-plus", str(inputs[1]),
                    "--existing", str(inputs[2]), "--output-dir", str(root / "out"),
                    "--warmup", str(missing),
                ])

            self.assertEqual(context.exception.args[0], missing)

    def test_rejects_existing_output_before_importing_mowl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for name in ("go.owl", "go-plus.owl", "existing.tsv"):
                path = root / name
                path.touch()
                inputs.append(path)

            with self.assertRaises(FileExistsError):
                AUDIT.main([
                    "--go", str(inputs[0]), "--go-plus", str(inputs[1]),
                    "--existing", str(inputs[2]), "--output-dir", str(root),
                ])


if __name__ == "__main__":
    unittest.main()
