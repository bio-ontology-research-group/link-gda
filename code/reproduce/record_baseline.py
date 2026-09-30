"""Record a repeatable baseline and compare later runs against it.

    python code/reproduce/record_baseline.py --label before
    python code/reproduce/record_baseline.py --compare .reproducibility/before --new .reproducibility/after

Recording writes a machine-readable record under .reproducibility/<label>/ and
refuses to overwrite an existing baseline. The record contains git state,
interpreter and package versions, the full pytest run with per-test ids and
outcomes (captured via JUnit XML), a static syntax-only smoke check, sha256
hashes of the pinned source files, and the numerical outputs of
code/reproduce/fixture_metrics.py on a fixed synthetic fixture (explicitly
marked as a fixture, not a paper reproduction).

The interpreter defaults to sys.executable; an explicit --python that does not
exist is an error, never a silent fallback.

Comparison rejects an unusable baseline (failures, skips, gaps, zero tests,
static failures, or a missing/non-zero-exit fixture output) instead of
interpreting it loosely. Missing numeric keys, changed fixture identity,
dropped test ids, new or changed skips, non-zero unit or fixture exits,
static-check failures, non-finite values, and new gaps are failures;
source-hash drift and newly added keys are informational.

SOURCE_FILES keeps the stable logical id of every recorded file; when a
recorded file is relocated, SOURCE_RELOCATIONS maps its old logical name to
the current physical path and the hash and static loops read through that
mapping, so earlier baselines remain comparable.
"""
import argparse
import ast
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPRODUCIBILITY_DIR = ".reproducibility"
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

PACKAGE_NAMES = ["numpy", "scipy", "click", "torch", "pytest"]

SOURCE_FILES = [
    "tests/conftest.py",
    "code/link_gda/__init__.py",
    "code/analysis/__init__.py",
    "code/link_gda/data.py",
    "code/link_gda/evaluation.py",
    "code/link_gda/pykeen_utils.py",
    "code/link_gda/negative_sampling.py",
    "code/training/kge_transd.py",
    "code/training/kge_convkb_d.py",
    "tests/test_training_layout.py",
    "code/analysis/leakage_overlap.py",
    "code/analysis/leakage_overlap_perfold.py",
    "code/analysis/leakage_overlap_verify.py",
    "code/analysis/popularity_controls.py",
    "code/analysis/sem_sim_overlap.py",
    "code/analysis/strata_from_labels.py",
    "code/analysis/stratified_metrics.py",
    "code/data/build_association_files.py",
    "code/data/build_excluded_benchmark.py",
    "code/data/download_data.py",
    "code/data/generate_folds.py",
    "tests/test_data_analysis_layout.py",
    "code/analysis/aggregated_sem_sim_metrics.py",
    "code/analysis/analyze_excluded_seeds.py",
    "code/analysis/check_data_leakage.py",
    "code/analysis/compare_calibration_panels.py",
    "code/analysis/excluded_table.py",
    "code/analysis/gen_overlap_tables.py",
    "code/analysis/make_overlap_labels.py",
    "code/analysis/rank_cdf_median.py",
    "code/analysis/ultra_excluded_table.py",
    "code/analysis/ultra_metrics.py",
    "code/figures/make_rankcdf_fig.py",
    "code/figures/make_calibration_fig.py",
    "tests/test_calibration_figure_layout.py",
    "tests/test_analysis_layout.py",
    "rq1_table.py",
    "calibrate_scores.py",
    "evaluate_sem_sim.py",
    "analysis/rq1_stats.py",
    "analysis/rq2_stats.py",
    "graph_statistics.py",
    "tests/test_graph_statistics.py",
    "tests/test_rq1_stats_cli.py",
    "tests/test_rq1_table_metrics.py",
    "tests/test_calibration.py",
    "tests/test_auc_definitions.py",
    "tests/test_rq1_stats.py",
    "tests/test_rq2_stats.py",
    "tests/test_baseline_recorder.py",
    "code/reproduce/paper_results.py",
    "tests/test_paper_results.py",
    "code/reproduce/fixture_metrics.py",
    "code/reproduce/record_baseline.py",
    "code/baselines/exomiser_eval.py",
    "code/baselines/prepare_ultra_data.py",
    "code/baselines/score_ultra.py",
    "code/projector/project_ontologies.py",
    "code/analysis/wandb/extract_metrics_from_folds.py",
    "code/analysis/wandb/extract_metrics_from_sweep.py",
    "code/analysis/wandb/extract_metrics_from_sweep_per_projector.py",
    "code/analysis/wandb/best_config_from_sweep.py",
    "code/analysis/wandb/best_config_cv.py",
    "tests/test_external_layout.py",
    "tests/test_external_resource_paths.py",
]

SOURCE_RELOCATIONS = {
    "rq1_table.py": "code/analysis/rq1_table.py",
    "calibrate_scores.py": "code/analysis/calibrate_scores.py",
    "evaluate_sem_sim.py": "code/link_gda/evaluate_sem_sim.py",
    "analysis/rq1_stats.py": "code/analysis/rq1_stats.py",
    "analysis/rq2_stats.py": "code/analysis/rq2_stats.py",
    "graph_statistics.py": "code/analysis/graph_statistics.py",
}


def source_path(root, rel):
    return root / SOURCE_RELOCATIONS.get(rel, rel)

SHELL_FILES = [
    "code/projector/compile_projector.sh",
    "run_all_sem_sim.sh",
    "run_ultra_export.sh",
    "run_ultra_score.sh",
    "setup_ultra_env.sh",
    "validate_ultra_env.sh",
]

SHELL_RELOCATIONS = {
    "run_all_sem_sim.sh": "code/baselines/run_all_sem_sim.sh",
    "run_ultra_export.sh": "code/baselines/run_ultra_export.sh",
    "run_ultra_score.sh": "code/baselines/run_ultra_score.sh",
    "setup_ultra_env.sh": "code/baselines/setup_ultra_env.sh",
    "validate_ultra_env.sh": "code/baselines/validate_ultra_env.sh",
}


def shell_path(root, rel):
    return root / SHELL_RELOCATIONS.get(rel, rel)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label(label):
    """--label must be a single, safe directory name; no separators or dot runs."""
    if not LABEL_RE.match(label) or label in (".", ".."):
        raise ValueError(
            f"invalid --label {label!r}: must be a single directory name using "
            "letters, digits, '_', '.', '-'; no path separators or '..'"
        )


def git_state(root):
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)

    head = run("rev-parse", "HEAD")
    if head.returncode:
        return {"available": False, "reason": head.stderr.strip()}
    status = run("status", "--porcelain")
    branch = run("branch", "--show-current")
    return {
        "available": True,
        "head": head.stdout.strip(),
        "branch": branch.stdout.strip() or None,
        "dirty": bool(status.stdout.strip()),
        "changed_paths": [
            line[:2].strip() + line[2:] for line in status.stdout.splitlines() if line
        ],
    }


def environment_info(python):
    script = (
        "import sys; from importlib import metadata; out = {'python': sys.version.split()[0]};\n"
        "for name in %r:\n"
        "    try:\n"
        "        out[name] = metadata.version(name)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        out[name] = None\n"
        "print(__import__('json').dumps(out, sort_keys=True))\n"
    ) % PACKAGE_NAMES
    completed = subprocess.run([str(python), "-c", script], capture_output=True, text=True)
    try:
        info = json.loads(completed.stdout)
    except json.JSONDecodeError:
        info = {"python": None, "error": completed.stderr.strip()}
    info["python_executable"] = str(python)
    return info


def parse_junit(path):
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError("junit XML has no testsuite element")
    cases = list(suite.iter("testcase"))
    declared = {
        key: int(suite.get(key, 0)) for key in ("tests", "skipped", "failures", "errors")
    }
    test_ids, skipped_ids, failed_ids, error_ids = [], [], [], []
    for case in cases:
        test_id = f"{case.get('classname')}::{case.get('name')}"
        test_ids.append(test_id)
        if case.find("skipped") is not None:
            skipped_ids.append(test_id)
        elif case.find("error") is not None:
            error_ids.append(test_id)
        elif case.find("failure") is not None:
            failed_ids.append(test_id)
    observed = {
        "tests": len(cases),
        "skipped": len(skipped_ids),
        "failures": len(failed_ids),
        "errors": len(error_ids),
    }
    if observed != declared:
        raise ValueError(f"junit counts inconsistent: declared {declared}, observed {observed}")
    return {
        "test_ids": sorted(test_ids),
        "skipped_ids": sorted(skipped_ids),
        "failed_ids": sorted(failed_ids),
        "error_ids": sorted(error_ids),
        "passed": len(cases) - len(skipped_ids) - len(failed_ids) - len(error_ids),
        **observed,
    }


def run_unit_tests(root, python, out_dir):
    command = [str(python), "-m", "pytest", "tests", "-q",
               f"--junitxml={out_dir / 'pytest.junit.xml'}"]
    if subprocess.run([str(python), "-c", "import pytest"], capture_output=True).returncode:
        return {
            "runner": "pytest", "command": command, "exit_code": None,
            "error": (
                f"pytest is not importable with {python}; refusing to fall back to "
                "unittest because pytest-only tests (tests/test_graph_statistics.py) "
                "would be silently omitted"
            ),
            "tests": 0, "passed": 0, "skipped": 0, "failures": 0, "errors": 0,
            "test_ids": [], "skipped_ids": [], "failed_ids": [], "error_ids": [],
            "stdout_file": "unit_tests.stdout.txt", "stderr_file": "unit_tests.stderr.txt",
        }
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=1800)
    (out_dir / "unit_tests.stdout.txt").write_text(completed.stdout)
    (out_dir / "unit_tests.stderr.txt").write_text(completed.stderr)
    result = {"runner": "pytest", "command": command, "exit_code": completed.returncode,
              "stdout_file": "unit_tests.stdout.txt", "stderr_file": "unit_tests.stderr.txt"}
    try:
        result.update(parse_junit(out_dir / "pytest.junit.xml"))
    except (ET.ParseError, ValueError, OSError) as exc:
        result["error"] = f"junit XML unusable: {exc}"
        result.update({"tests": 0, "passed": 0, "skipped": 0, "failures": 0, "errors": 0,
                       "test_ids": [], "skipped_ids": [], "failed_ids": [], "error_ids": []})
    return result


def run_static_smoke(root):
    """Static-only checks: AST parse for pinned Python files, bash -n for shell scripts."""
    results = []
    for rel in SOURCE_FILES:
        path = source_path(root, rel)
        if not path.exists():
            results.append({"file": rel, "check": "python_ast", "ok": False, "detail": "missing"})
            continue
        try:
            ast.parse(path.read_text())
            detail = "parses"
        except SyntaxError as exc:
            detail = str(exc)
        results.append({"file": rel, "check": "python_ast", "ok": detail == "parses", "detail": detail})
    for rel in SHELL_FILES:
        path = shell_path(root, rel)
        if not path.exists():
            results.append({"file": rel, "check": "bash_n", "ok": False, "detail": "missing"})
            continue
        completed = subprocess.run(["bash", "-n", str(path)], cwd=root,
                                   capture_output=True, text=True)
        ok = completed.returncode == 0
        results.append({"file": rel, "check": "bash_n", "ok": ok,
                        "detail": "parses" if ok else completed.stderr.strip()})
    return {
        "kind": "static_syntax_only",
        "note": "AST/bash -n syntax smoke only; not a behavioral test, nothing executed.",
        "results": results,
    }


def run_fixture_metrics(root, python, out_dir):
    command = [str(python), str(root / "code" / "reproduce" / "fixture_metrics.py")]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=600)
    (out_dir / "fixture_metrics.stdout.txt").write_text(completed.stdout)
    (out_dir / "fixture_metrics.stderr.txt").write_text(completed.stderr)
    result = {"command": command, "exit_code": completed.returncode,
              "json_file": None, "ok": False, "error": None}
    if completed.returncode != 0:
        result["error"] = f"fixture_metrics exited {completed.returncode}"
        return result
    try:
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or "outputs" not in payload:
            raise ValueError("output is not an object with an 'outputs' section")
    except json.JSONDecodeError as exc:
        result["error"] = f"fixture_metrics output is not valid JSON: {exc}"
        return result
    (out_dir / "fixture_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    result["json_file"] = "fixture_metrics.json"
    result["gaps"] = payload.get("gaps", [])
    result["ok"] = completed.returncode == 0 and not result["gaps"]
    return result


def record(root, label, python=None, out_root=None):
    root = Path(root)
    validate_label(label)
    python = Path(python) if python else Path(sys.executable)
    if not python.exists():
        raise FileNotFoundError(f"python interpreter not found: {python}")
    out_root = Path(out_root) if out_root else root / REPRODUCIBILITY_DIR
    out_dir = (out_root / label).resolve()
    if out_dir.parent != out_root.resolve():
        raise ValueError(f"label {label!r} would write outside {out_root}")
    if out_dir.exists():
        raise FileExistsError(
            f"baseline already exists at {out_dir}; baselines are never overwritten. "
            "Use a different --label."
        )
    out_dir.mkdir(parents=True)

    tests = run_unit_tests(root, python, out_dir)
    static = run_static_smoke(root)
    fixture = run_fixture_metrics(root, python, out_dir)

    gaps = list(fixture.get("gaps", []))
    hashes = {}
    for rel in SOURCE_FILES:
        path = source_path(root, rel)
        if path.exists():
            hashes[rel] = sha256_of(path)
        else:
            gaps.append(f"pinned source file missing: {rel}")

    issues = []
    if tests.get("error"):
        issues.append(f"unit tests: {tests['error']}")
    if tests["exit_code"] != 0:
        issues.append(f"unit tests exited {tests['exit_code']}")
    if tests["tests"] == 0:
        issues.append("unit tests collected zero tests")
    if tests["failures"] or tests["errors"]:
        issues.append(f"unit tests: {tests['failures']} failures, {tests['errors']} errors")
    if tests["skipped"]:
        issues.append(f"unit tests skipped {tests['skipped']} test(s): {tests['skipped_ids']}")
    bad_static = [r["file"] for r in static["results"] if not r["ok"]]
    if bad_static:
        issues.append(f"static smoke failures: {bad_static}")
    if not fixture.get("ok"):
        issues.append(f"fixture metrics not ok: {fixture.get('error') or '; '.join(gaps)}")
    if gaps:
        issues.append(f"gaps: {gaps}")

    record = {
        "label": label,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "repo_root": str(root),
        "baseline_kind": "pre-layout-migration",
        "reproduces_paper": False,
        "health": "ok" if not issues else "failed",
        "health_issues": issues,
        "git": git_state(root),
        "environment": environment_info(python),
        "unit_tests": tests,
        "static_smoke": static,
        "fixture_metrics": fixture,
        "source_hashes": hashes,
        "gaps": gaps,
    }
    (out_dir / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def _walk_leaves(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_leaves(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_leaves(value, f"{path}[{i}]")
    else:
        yield path, node


def _compare_outputs(base_outputs, new_outputs, rtol, atol):
    failures, informational = [], []
    base_leaves = dict(_walk_leaves(base_outputs, "outputs"))
    new_leaves = dict(_walk_leaves(new_outputs, "outputs"))
    for path, base_value in base_leaves.items():
        if path not in new_leaves:
            failures.append(f"missing output key: {path}")
            continue
        new_value = new_leaves[path]
        base_is_num = isinstance(base_value, (int, float)) and not isinstance(base_value, bool)
        new_is_num = isinstance(new_value, (int, float)) and not isinstance(new_value, bool)
        if not (base_is_num and new_is_num):
            if base_value != new_value:
                failures.append(f"output changed: {path}: {base_value!r} -> {new_value!r}")
            continue
        base_num, new_num = float(base_value), float(new_value)
        if not (math.isfinite(base_num) and math.isfinite(new_num)):
            failures.append(f"non-finite output value: {path}: {base_value!r} -> {new_value!r}")
        elif not math.isclose(new_num, base_num, rel_tol=rtol, abs_tol=atol):
            failures.append(f"numerical drift: {path}: {base_value!r} -> {new_value!r}")
    for path in sorted(set(new_leaves) - set(base_leaves)):
        informational.append(f"new output key: {path}")
    return failures, informational


def _record_problems(record, role):
    problems = []
    unit = record.get("unit_tests", {})
    if record.get("gaps"):
        problems.append(f"{role} has gaps: {record['gaps']}")
    if unit.get("error") or unit.get("exit_code", 0) != 0:
        problems.append(f"{role} unit tests did not exit cleanly")
    if not unit.get("tests"):
        problems.append(f"{role} collected zero tests")
    if unit.get("failures") or unit.get("errors") or unit.get("skipped"):
        problems.append(
            f"{role} unit tests: {unit.get('failures')} failures, "
            f"{unit.get('errors')} errors, {unit.get('skipped')} skips"
        )
    if any(not r["ok"] for r in record.get("static_smoke", {}).get("results", [])):
        problems.append(f"{role} has static smoke failures")
    if not record.get("fixture_metrics", {}).get("ok"):
        problems.append(f"{role} fixture metrics are not ok")
    return problems


def _load_json(path, failures):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"malformed or missing file {path}: {exc}")
        return None


def compare_records(base_path, new_path, rtol=1e-9, atol=1e-12):
    """Compare a later record against a baseline record.

    An unusable baseline is rejected outright. Failures: missing outputs,
    numerical drift beyond tolerance, non-finite values, changed fixture
    identity, dropped test ids, new or changed skips, non-zero exits, static
    failures, and new gaps. Informational: source-hash drift and new keys.
    """
    if not (math.isfinite(rtol) and math.isfinite(atol)) or rtol <= 0 or atol < 0:
        raise ValueError("tolerances must be finite, rtol > 0 and atol >= 0")
    failures, informational = [], []
    base_dir, new_dir = Path(base_path), Path(new_path)
    base = _load_json(base_dir / "record.json", failures)
    new = _load_json(new_dir / "record.json", failures)
    if base is None or new is None:
        return {"ok": False, "failures": failures, "informational": [],
                "tolerances": {"rtol": rtol, "atol": atol},
                "baseline_label": (base or {}).get("label"),
                "new_label": (new or {}).get("label")}

    baseline_problems = _record_problems(base, "baseline")
    if baseline_problems:
        return {"ok": False,
                "failures": ["baseline is unusable and is rejected: " + "; ".join(baseline_problems)],
                "informational": [], "tolerances": {"rtol": rtol, "atol": atol},
                "baseline_label": base.get("label"), "new_label": new.get("label")}

    for problem in _record_problems(new, "new"):
        failures.append(problem)

    base_fixture = _load_json(
        base_dir / base.get("fixture_metrics", {}).get("json_file", "fixture_metrics.json"), failures)
    new_fixture = _load_json(
        new_dir / new.get("fixture_metrics", {}).get("json_file", "fixture_metrics.json"), failures)
    if base_fixture is not None and new_fixture is not None:
        for section in ("provenance", "fixture"):
            if base_fixture.get(section) != new_fixture.get(section):
                failures.append(f"fixture identity changed in {section!r}")
        if new_fixture.get("gaps"):
            failures.append(f"new run has fixture gaps: {new_fixture['gaps']}")
        out_failures, out_informational = _compare_outputs(
            base_fixture.get("outputs", {}), new_fixture.get("outputs", {}), rtol, atol)
        failures.extend(out_failures)
        informational.extend(out_informational)

    base_tests, new_tests = base["unit_tests"], new["unit_tests"]
    base_ids, new_ids = set(base_tests.get("test_ids", [])), set(new_tests.get("test_ids", []))
    dropped = sorted(base_ids - new_ids)
    if dropped:
        failures.append(f"tests dropped vs baseline: {dropped}")
    added = sorted(new_ids - base_ids)
    if added:
        informational.append(f"new tests added: {added}")
    base_skips, new_skips = set(base_tests.get("skipped_ids", [])), set(new_tests.get("skipped_ids", []))
    unexpected_skips = sorted(new_skips - base_skips)
    if unexpected_skips:
        failures.append(f"new or changed skips vs baseline: {unexpected_skips}")

    base_static = {r["file"] for r in base.get("static_smoke", {}).get("results", [])}
    new_static = {r["file"]: r for r in new.get("static_smoke", {}).get("results", [])}
    for rel in sorted(base_static - set(new_static)):
        failures.append(f"static check dropped for {rel}")

    for rel in sorted(set(base.get("source_hashes", {})) | set(new.get("source_hashes", {}))):
        if base.get("source_hashes", {}).get(rel) != new.get("source_hashes", {}).get(rel):
            informational.append(f"source hash drift (expected during migration): {rel}")

    return {
        "ok": not failures,
        "failures": failures,
        "informational": informational,
        "tolerances": {"rtol": rtol, "atol": atol},
        "baseline_label": base.get("label"),
        "new_label": new.get("label"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="before")
    parser.add_argument("--root", default=None, help="Repo root (default: derived from this file)")
    parser.add_argument("--python", default=None,
                        help="Python interpreter (default: the one running this script)")
    parser.add_argument("--compare", default=None,
                        help="Record directory to compare the new record against (baseline)")
    parser.add_argument("--new", default=None,
                        help="With --compare: the later record directory to compare")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else REPO_ROOT

    if args.compare:
        base_path = Path(args.compare)
        new_path = Path(args.new) if args.new else root / REPRODUCIBILITY_DIR / args.label
        try:
            result = compare_records(base_path, new_path, args.rtol, args.atol)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1

    if args.python and not Path(args.python).exists():
        parser.error(f"--python {args.python} not found; refusing to fall back silently")
    try:
        baseline = record(root, args.label, python=args.python)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"recorded {root / REPRODUCIBILITY_DIR / args.label}")
    tests = baseline["unit_tests"]
    print(
        f"tests: {tests['tests']} passed {tests['passed']} skipped {tests['skipped']} "
        f"failures {tests['failures']} errors {tests['errors']} exit {tests['exit_code']}"
    )
    for gap in baseline["gaps"]:
        print(f"gap: {gap}")
    for issue in baseline["health_issues"]:
        print(f"health: {issue}")
    return 0 if baseline["health"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
