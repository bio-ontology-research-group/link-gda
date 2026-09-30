"""Tests for the baseline recorder/comparator itself, including deliberate
perturbations so the comparator cannot falsely pass."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "code" / "reproduce" / "record_baseline.py"
FIXTURE_PATH = ROOT / "code" / "reproduce" / "fixture_metrics.py"

TEST_IDS = ["m::a", "m::b", "m::c"]


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_unit_tests(test_ids=None, exit_code=0, skipped_ids=(), failed_ids=(), error_ids=()):
    test_ids = list(test_ids if test_ids is not None else TEST_IDS)
    return {
        "runner": "pytest",
        "command": [sys.executable, "-m", "pytest", "tests"],
        "exit_code": exit_code,
        "tests": len(test_ids),
        "passed": len(test_ids) - len(skipped_ids) - len(failed_ids) - len(error_ids),
        "skipped": len(skipped_ids),
        "failures": len(failed_ids),
        "errors": len(error_ids),
        "test_ids": sorted(test_ids),
        "skipped_ids": sorted(skipped_ids),
        "failed_ids": sorted(failed_ids),
        "error_ids": sorted(error_ids),
    }


def make_record(label, hashes=None, unit_tests=None, gaps=(), static_files=None):
    static_files = static_files if static_files is not None else ["rq1_table.py"]
    return {
        "label": label,
        "repo_root": "/fake",
        "health": "ok",
        "health_issues": [],
        "fixture_metrics": {"exit_code": 0, "json_file": "fixture_metrics.json", "ok": True},
        "unit_tests": unit_tests if unit_tests is not None else make_unit_tests(),
        "static_smoke": {
            "results": [
                {"file": f, "check": "python_ast", "ok": True, "detail": "parses"}
                for f in static_files
            ]
        },
        "source_hashes": hashes if hashes is not None else {"rq1_table.py": "a" * 64},
        "gaps": list(gaps),
    }


def write_record(directory, record, fixture=None, raw_text=None):
    directory.mkdir(parents=True)
    if raw_text is not None:
        (directory / "record.json").write_text(raw_text)
    else:
        (directory / "record.json").write_text(json.dumps(record, sort_keys=True))
    if fixture is not None:
        (directory / "fixture_metrics.json").write_text(json.dumps(fixture, sort_keys=True))
    return directory


class RecorderGuardTests(unittest.TestCase):
    def test_refuses_to_overwrite_an_existing_baseline(self):
        recorder = import_module("record_baseline_guard", RECORD_PATH)
        with tempfile.TemporaryDirectory() as directory:
            out_root = Path(directory)
            write_record(out_root / "before", make_record("before"))
            with self.assertRaises(FileExistsError):
                recorder.record(ROOT, "before", python=sys.executable, out_root=out_root)

    def test_rejects_labels_with_path_separators_or_dotdot(self):
        recorder = import_module("record_baseline_label", RECORD_PATH)
        for bad in ("../escape", "a/b", ".reproducibility/x", "/", ".."):
            with self.assertRaises(ValueError, msg=bad):
                recorder.validate_label(bad)
        recorder.validate_label("before-cleanup_2.x")

    def test_missing_explicit_python_fails(self):
        recorder = import_module("record_baseline_py", RECORD_PATH)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                recorder.record(ROOT, "x", python="/nonexistent/python",
                                out_root=Path(directory))


class FixtureMetricsDeterminismTests(unittest.TestCase):
    def test_same_outputs_across_repeated_runs(self):
        fixture = import_module("fixture_metrics_det", FIXTURE_PATH)
        first = json.dumps(fixture.build_outputs(), sort_keys=True)
        second = json.dumps(fixture.build_outputs(), sort_keys=True)
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertFalse(payload["provenance"]["reproduces_paper"])
        self.assertEqual(payload["provenance"]["type"], "synthetic_fixture")

    def test_outputs_contain_both_auc_definitions_and_row_metrics(self):
        fixture = import_module("fixture_metrics_keys", FIXTURE_PATH)
        outputs = fixture.build_outputs()["outputs"]
        self.assertIn("auc", outputs["rq1_table_metrics_uncalibrated"])
        self.assertIn("evaluate_sem_sim_auc_trapezoid", outputs)
        self.assertIn("evaluate_sem_sim_metrics_rows", outputs)


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.recorder = import_module("record_baseline_cmp", RECORD_PATH)
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def dir(self, name):
        return Path(self.tmp.name) / name

    _NO_FIXTURE = object()

    def pair(self, base_record=None, new_record=None, base_fixture=None, new_fixture=None,
             base_dir="base", new_dir="new"):
        fixture = base_fixture if base_fixture is not None else {"outputs": {"auc": 0.5}}
        base = write_record(self.dir(base_dir), base_record or make_record("base"), fixture)
        if new_fixture is self._NO_FIXTURE:
            new = write_record(self.dir(new_dir), new_record or make_record("new"))
        else:
            effective = new_fixture if new_fixture is not None else fixture
            new = write_record(self.dir(new_dir), new_record or make_record("new"), effective)
        return self.recorder.compare_records(base, new)

    def test_identical_records_pass(self):
        result = self.pair()
        self.assertTrue(result["ok"], result["failures"])

    def test_numerical_perturbation_beyond_tolerance_fails(self):
        result = self.pair(new_fixture={"outputs": {"auc": 0.5 + 1e-4}})
        self.assertFalse(result["ok"])
        self.assertTrue(any("auc" in failure for failure in result["failures"]))

    def test_perturbation_within_tolerance_passes(self):
        result = self.pair(new_fixture={"outputs": {"auc": 0.5 + 1e-13}})
        self.assertTrue(result["ok"], result["failures"])

    def test_missing_new_fixture_file_fails(self):
        result = self.pair(new_fixture=self._NO_FIXTURE)
        self.assertFalse(result["ok"])

    def test_missing_record_fails(self):
        write_record(self.dir("base"), make_record("base"), {"outputs": {"auc": 0.5}})
        result = self.recorder.compare_records(
            self.dir("base"), self.dir("does_not_exist"))
        self.assertFalse(result["ok"])

    def test_malformed_record_json_fails(self):
        write_record(self.dir("base"), make_record("base"), {"outputs": {"auc": 0.5}})
        write_record(self.dir("new"), None, raw_text="{not json")
        result = self.recorder.compare_records(self.dir("base"), self.dir("new"))
        self.assertFalse(result["ok"])

    def test_missing_numeric_key_in_outputs_fails(self):
        result = self.pair(new_fixture={"outputs": {}})
        self.assertFalse(result["ok"])
        self.assertTrue(any("missing output key" in f for f in result["failures"]))

    def test_renamed_output_fails(self):
        result = self.pair(new_fixture={"outputs": {"auc_renamed": 0.5}})
        self.assertFalse(result["ok"])

    def test_nan_in_new_output_fails(self):
        result = self.pair(new_fixture={"outputs": {"auc": float("nan")}})
        self.assertFalse(result["ok"])
        self.assertTrue(any("non-finite" in f for f in result["failures"]))

    def test_inf_in_new_output_fails(self):
        result = self.pair(new_fixture={"outputs": {"auc": float("inf")}})
        self.assertFalse(result["ok"])

    def test_nonfinite_or_negative_tolerances_raise(self):
        base = write_record(self.dir("base"), make_record("base"), {"outputs": {"auc": 0.5}})
        new = write_record(self.dir("new"), make_record("new"), {"outputs": {"auc": 0.5}})
        for rtol, atol in ((float("nan"), 1e-12), (float("inf"), 1e-12),
                           (1e-9, -1.0), (-1e-9, 1e-12), (0.0, 1e-12)):
            with self.assertRaises(ValueError, msg=(rtol, atol)):
                self.recorder.compare_records(base, new, rtol=rtol, atol=atol)

    def test_changed_fixture_identity_fails(self):
        base_fixture = {"outputs": {"auc": 0.5}, "fixture": {"seed": 1}, "provenance": {"p": 1}}
        new_fixture = {"outputs": {"auc": 0.5}, "fixture": {"seed": 2}, "provenance": {"p": 1}}
        result = self.pair(base_fixture=base_fixture, new_fixture=new_fixture)
        self.assertFalse(result["ok"])
        self.assertTrue(any("identity" in f for f in result["failures"]))

    def test_changed_provenance_fails(self):
        base_fixture = {"outputs": {"auc": 0.5}, "fixture": {"s": 1},
                        "provenance": {"reproduces_paper": False}}
        new_fixture = {"outputs": {"auc": 0.5}, "fixture": {"s": 1},
                       "provenance": {"reproduces_paper": True}}
        result = self.pair(base_fixture=base_fixture, new_fixture=new_fixture)
        self.assertFalse(result["ok"])

    def test_changed_categorical_output_fails(self):
        base_fixture = {"outputs": {"auc": 0.5, "alt": "one-sided: x > 0"}}
        new_fixture = {"outputs": {"auc": 0.5, "alt": "two-sided"}}
        result = self.pair(base_fixture=base_fixture, new_fixture=new_fixture)
        self.assertFalse(result["ok"])

    def test_new_output_key_is_informational_not_failure(self):
        result = self.pair(new_fixture={"outputs": {"auc": 0.5, "extra": 1.0}})
        self.assertTrue(result["ok"], result["failures"])
        self.assertTrue(any("extra" in note for note in result["informational"]))

    def test_source_hash_drift_is_informational_not_failure(self):
        drifted = make_record("new", hashes={"rq1_table.py": "b" * 64})
        result = self.pair(new_record=drifted)
        self.assertTrue(result["ok"], result["failures"])
        self.assertTrue(any("rq1_table.py" in note for note in result["informational"]))

    def test_new_run_with_test_failures_fails(self):
        unit = make_unit_tests(failed_ids=["m::a"])
        result = self.pair(new_record=make_record("new", unit_tests=unit))
        self.assertFalse(result["ok"])

    def test_nonzero_exit_with_zero_reported_failures_fails(self):
        unit = make_unit_tests(exit_code=1)
        result = self.pair(new_record=make_record("new", unit_tests=unit))
        self.assertFalse(result["ok"])

    def test_new_skip_is_a_failure(self):
        unit = make_unit_tests(skipped_ids=["m::a"])
        result = self.pair(new_record=make_record("new", unit_tests=unit))
        self.assertFalse(result["ok"])

    def test_dropped_test_id_fails(self):
        unit = make_unit_tests(test_ids=["m::a", "m::b"])
        result = self.pair(new_record=make_record("new", unit_tests=unit))
        self.assertFalse(result["ok"])
        self.assertTrue(any("m::c" in failure for failure in result["failures"]))

    def test_added_test_id_is_informational(self):
        unit = make_unit_tests(test_ids=TEST_IDS + ["m::d"])
        result = self.pair(new_record=make_record("new", unit_tests=unit))
        self.assertTrue(result["ok"], result["failures"])
        self.assertTrue(any("m::d" in note for note in result["informational"]))

    def test_static_check_failure_fails(self):
        record = make_record("new", static_files=["rq1_table.py"])
        record["static_smoke"]["results"][0]["ok"] = False
        record["static_smoke"]["results"][0]["detail"] = "SyntaxError"
        result = self.pair(new_record=record)
        self.assertFalse(result["ok"])

    def test_dropped_static_check_fails(self):
        result = self.pair(new_record=make_record("new", static_files=[]))
        self.assertFalse(result["ok"])

    def test_new_gaps_fail(self):
        result = self.pair(new_record=make_record("new", gaps=["pinned source file missing: x"]))
        self.assertFalse(result["ok"])

    def test_new_fixture_gaps_fail(self):
        result = self.pair(new_fixture={"outputs": {"auc": 0.5}, "gaps": ["boom"]})
        self.assertFalse(result["ok"])

    def test_unusable_baseline_is_rejected(self):
        for i, baseline in enumerate((
            make_record("base", gaps=["something gapped"]),
            make_record("base", unit_tests=make_unit_tests(skipped_ids=["m::a"])),
            make_record("base", unit_tests=make_unit_tests(failed_ids=["m::a"])),
            make_record("base", unit_tests=make_unit_tests(test_ids=[])),
        )):
            result = self.pair(base_record=baseline, base_dir=f"base{i}", new_dir=f"new{i}")
            self.assertFalse(result["ok"])
            self.assertTrue(any("unusable" in failure for failure in result["failures"]),
                            result["failures"])


class SourceRelocationTests(unittest.TestCase):
    def test_static_check_reads_relocated_source_with_original_identifier(self):
        module = import_module("recorder_relocated", RECORD_PATH)
        module.SOURCE_FILES = ["rq1_table.py"]
        module.SHELL_FILES = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rq1_table.py").write_text("this is not valid Python !")
            target = root / "code" / "analysis" / "rq1_table.py"
            target.parent.mkdir(parents=True)
            target.write_text("value = 1\n")
            result = module.run_static_smoke(root)
            self.assertTrue(result["results"][0]["ok"])
            self.assertEqual(result["results"][0]["file"], "rq1_table.py")

    def test_missing_relocated_source_does_not_fall_back_to_old_file(self):
        module = import_module("recorder_missing_relocated", RECORD_PATH)
        module.SOURCE_FILES = ["rq1_table.py"]
        module.SHELL_FILES = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rq1_table.py").write_text("value = 1\n")
            result = module.run_static_smoke(root)
            self.assertFalse(result["results"][0]["ok"])
            self.assertEqual(result["results"][0]["detail"], "missing")


if __name__ == "__main__":
    unittest.main()
