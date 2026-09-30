"""Recompute two paper tables from saved prediction files, with strict input validation.

Two regression checks recompute, from saved per-instance score TSV files, the
numbers the manuscript prints. The script never retrains a model and never runs
checkpoint inference; it only re-runs the production load, calibration, and
metric code (``rq1_table`` and ``analysis.rq1_stats``) on the artifacts written
out at evaluation time. Scientific scope, repeated in every report: recompute
saved predictions, no retraining/no checkpoint inference.

Checks
  rq1       paper/main.tex Table 2, overall, calibrated arm, owl2vecstar
            projection, seed 0, folds 0-9: LinkGDA versus INDIGENA fold mean
            rank, aggregated over folds, plus the one-sided Nadeau-Bengio
            corrected resampled t-test from ``analysis.rq1_stats.corrected_test``
            and the 6-way Bonferroni adjustment used in the paper.
  excluded  paper/main.tex Table 3, LinkGDA-fs, owl2vecstar projection, full
            candidate pool, calibrated arm, fold 0, seeds 0-9: mean rank and
            H@10 aggregated over seeds.

Input TSV rows are ``gene<TAB>disease<TAB>true_idx<TAB>score ...``. The files
carry no candidate-identifier column, so candidate-set identity across methods
or seeds cannot be verified from these artifacts; only candidate counts,
case-key order, and true-gene indices can. The limitation is repeated in the
report.

The loader pre-validates every row (field count, non-empty gene/disease,
integer true index, finite scores, uniform candidate count) and then parses
with the production ``analysis.rq1_stats.load`` so the check exercises the
production parsing; it rejects out-of-range true indices and duplicate case
keys instead of silently skipping, because the production loader skips short
lines on its own.

Status and exit codes
  pass        0  every selected check verifies against its expected values
  fail        1  a computed value, structural property, or input is rejected
  unverified  2  a required artifact file is missing
A mixed run reports fail if any check fails, else unverified if any check is
unverified.

    python code/reproduce/paper_results.py --check all \
        --rq1-results data/results --excluded-results data/results_excluded \
        [--expected overrides.json] [--precision-reference rq1_recomputed.json] \
        [--output report.json]
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

if "__file__" in globals():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import rq1_table
try:
    import analysis.rq1_stats as rq1_stats
except ModuleNotFoundError:
    rq1_stats = sys.modules["analysis.rq1_stats"]
corrected_test = rq1_stats.corrected_test
production_load = rq1_stats.load

SCIENTIFIC_SCOPE = "recompute saved predictions, no retraining/no checkpoint inference"

DISPLAY_DIGITS = 2
P_DISPLAY_DIGITS = 4
PRECISION_REL_TOL = 1e-12
BONFERRONI_TERMS = 6

EXCLUDED_TEMPLATE = (
    "kge_results_transd_fold_0_seed_{seed}_dim_200_bs_65536_lr_0.001_func_expr_"
    "proj_owl2vecstar_use_graph_True_tol_15_calsel_by_graph_bma.tsv"
)

DEFAULT_EXPECTED = {
    "rq1": {
        "source": "paper/main.tex Table 2 (overall, calibrated, owl2vecstar) and main text; "
                  "supplementary Nadeau-Bengio test table",
        "candidate_pool": 4399,
        "total_pairs": 6571,
        "methods": {
            "linkgda": {"mr_mean_display": 601.60, "mr_sd_display": 37.49},
            "indigena": {"mr_mean_display": 673.21, "mr_sd_display": 58.13},
        },
        "p_raw_display": 0.0043,
        "p_bonferroni_display": 0.0256,
    },
    "excluded": {
        "source": "paper/main.tex Table 3 (LinkGDA-fs, owl2vecstar, full pool, calibrated)",
        "pairs": 409,
        "unique_genes": 350,
        "unique_diseases": 402,
        "candidate_pool": 4749,
        "mr_mean_display": 1038.16,
        "mr_sd_display": 54.44,
        "h10_mean_display": 0.07,
        "h10_sd_display": 0.01,
    },
}

RQ1_TEMPLATE_SOURCE = "analysis.rq1_stats.CONFIGS['owl2vecstar'] (production filenames for the main-table arms)"


def display(value, digits):
    """Round the way the tables print: fixed-point string, then back to float."""
    return float(f"{float(value):.{digits}f}")


def load_validated(path):
    """Load one saved prediction TSV strictly, parsing through the production loader.

    Raises FileNotFoundError for a missing file and ValueError for any
    malformed row, unequal candidate counts, out-of-range true index, or
    duplicate case key.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing saved prediction file: {path}")
    rows = 0
    widths = set()
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            fields = raw.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"{path}:{lineno}: malformed row with {len(fields)} fields")
            rows += 1
            widths.add(len(fields) - 3)
            if not fields[0] or not fields[1]:
                raise ValueError(f"{path}:{lineno}: empty gene or disease field")
            try:
                int(fields[2])
            except ValueError:
                raise ValueError(f"{path}:{lineno}: true_idx is not an integer: {fields[2]!r}") from None
            for column, text in enumerate(fields[3:], start=4):
                try:
                    number = float(text)
                except ValueError:
                    raise ValueError(f"{path}:{lineno}: field {column} is not a number: {text!r}") from None
                if not math.isfinite(number):
                    raise ValueError(f"{path}:{lineno}: field {column} is not finite")
    if len(widths) != 1:
        raise ValueError(f"{path}: unequal candidate counts across rows: {sorted(widths)}")
    scores, indices, keys, sha256 = production_load(path)
    if scores.shape != (rows, min(widths)):
        raise ValueError(f"{path}: production loader parsed shape {scores.shape}, expected {(rows, min(widths))}")
    if rows < 2:
        raise ValueError(f"{path}: leave-one-out calibration needs at least 2 rows, got {rows}")
    width = scores.shape[1]
    bad = [row for row, index in enumerate(indices) if index < 0 or index >= width]
    if bad:
        raise ValueError(f"{path}: true_idx out of range for {width} candidates at rows {bad[:5]}")
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{path}: duplicate case keys {duplicates[:5]} (not silently deduplicated; investigate)"
        )
    metrics = rq1_table.metrics(rq1_table.calibrate(scores), indices)
    return {
        "path": str(path),
        "sha256": sha256,
        "bytes": path.stat().st_size,
        "pairs": int(rows),
        "candidates": int(width),
        "unique_genes": len({key[0] for key in keys}),
        "unique_diseases": len({key[1] for key in keys}),
        "keys": keys,
        "indices": indices,
        "mr": float(metrics["mr"]),
        "h10": float(metrics["h10"]),
    }


def file_summary(entry):
    return {key: entry[key] for key in ("path", "sha256", "bytes", "pairs", "candidates",
                                        "unique_genes", "unique_diseases")}


def display_comparison(quantity, computed, expected_display, digits):
    shown = display(computed, digits)
    return {
        "quantity": quantity,
        "computed_full": float(computed),
        "display_digits": digits,
        "display": shown,
        "expected": float(expected_display),
        "ok": shown == expected_display,
    }


def precision_comparison(quantity, computed, reference):
    computed, reference = float(computed), float(reference)
    return {
        "quantity": quantity,
        "computed_full": computed,
        "reference_full": reference,
        "rel_tol": PRECISION_REL_TOL,
        "ok": math.isclose(computed, reference, rel_tol=PRECISION_REL_TOL, abs_tol=0.0),
    }


def default_precision_reference():
    if "__file__" not in globals():
        return None
    candidate = (Path(__file__).resolve().parents[2] / "paper" / "reviews"
                 / "point36_multiplicity_2026-09-29" / "rq1_recomputed.json")
    return candidate if candidate.is_file() else None


def resolve_precision_reference(explicit):
    if explicit is None:
        return default_precision_reference()
    if str(explicit).lower() == "none":
        return None
    return explicit


def run_rq1(results_dir, *, n_folds=10, expected=None, precision_reference=None):
    """Recompute the RQ1 (Table 2) check from the saved fold files under results_dir."""
    expected = DEFAULT_EXPECTED["rq1"] if expected is None else expected
    templates = dict(rq1_stats.CONFIGS["owl2vecstar"])
    report = {
        "scope": SCIENTIFIC_SCOPE,
        "template_source": RQ1_TEMPLATE_SOURCE,
        "expected": expected,
        "files": [],
        "query_alignment": [],
        "per_fold": [],
        "aggregate": {},
        "test": None,
        "comparisons": [],
        "precision_reference": None,
        "errors": [],
        "status": "pass",
    }
    loaded = {"linkgda": [], "indigena": []}
    for fold in range(n_folds):
        row = {"fold": fold}
        for method in ("linkgda", "indigena"):
            path = Path(results_dir) / templates[method].format(fold=fold)
            try:
                entry = load_validated(path)
            except FileNotFoundError as exc:
                report["status"] = "unverified"
                report["errors"].append(str(exc))
                return report
            except (ValueError, OSError) as exc:
                report["status"] = "fail"
                report["errors"].append(f"input validation: {exc}")
                return report
            loaded[method].append(entry)
            report["files"].append(file_summary(entry))
            row[method] = {"mr": entry["mr"], "pairs": entry["pairs"], "candidates": entry["candidates"]}
        link, ind = loaded["linkgda"][-1], loaded["indigena"][-1]
        keys_match = link["keys"] == ind["keys"]
        idx_match = bool(np.array_equal(link["indices"], ind["indices"]))
        report["query_alignment"].append({"fold": fold, "case_keys_match": keys_match,
                                          "true_indices_match": idx_match})
        if not (keys_match and idx_match):
            report["status"] = "fail"
            report["errors"].append(
                f"fold {fold}: case keys or true-gene indices differ between LinkGDA and INDIGENA")
            return report
        report["per_fold"].append(row)

    link_mr = [entry["mr"] for entry in loaded["linkgda"]]
    ind_mr = [entry["mr"] for entry in loaded["indigena"]]
    for method, entries in loaded.items():
        if any(entry["candidates"] != expected["candidate_pool"] for entry in entries):
            report["errors"].append(
                f"{method}: candidate count differs from expected {expected['candidate_pool']}")
    total_pairs = sum(entry["pairs"] for entry in loaded["linkgda"])
    if total_pairs != expected["total_pairs"]:
        report["errors"].append(f"total pairs {total_pairs} differs from expected {expected['total_pairs']}")

    report["aggregate"] = {
        method: {
            "mr_mean": float(np.mean(mr)),
            "mr_sample_sd": float(np.std(mr, ddof=1)),
        }
        for method, mr in (("linkgda", link_mr), ("indigena", ind_mr))
    }
    test = corrected_test(ind_mr, link_mr)
    p_raw = float(test["p_one_sided"])
    p_adjusted = min(1.0, float(BONFERRONI_TERMS) * p_raw)
    report["test"] = dict(test, bonferroni_terms=BONFERRONI_TERMS, p_adjusted=p_adjusted)

    comparisons = []
    for method in ("linkgda", "indigena"):
        comparisons.append(display_comparison(
            f"rq1.{method}.mr_mean", report["aggregate"][method]["mr_mean"],
            expected["methods"][method]["mr_mean_display"], DISPLAY_DIGITS))
        comparisons.append(display_comparison(
            f"rq1.{method}.mr_sample_sd", report["aggregate"][method]["mr_sample_sd"],
            expected["methods"][method]["mr_sd_display"], DISPLAY_DIGITS))
    comparisons.append(display_comparison("rq1.p_raw", p_raw, expected["p_raw_display"], P_DISPLAY_DIGITS))
    comparisons.append(display_comparison("rq1.p_bonferroni", p_adjusted,
                                          expected["p_bonferroni_display"], P_DISPLAY_DIGITS))
    report["comparisons"] = comparisons
    if report["errors"] or not all(item["ok"] for item in comparisons):
        report["status"] = "fail"
        report["errors"].extend(
            f"{item['quantity']}: displayed {item['display']} != expected {item['expected']}"
            for item in comparisons if not item["ok"])
        return report

    reference = resolve_precision_reference(precision_reference)
    if reference is None:
        report["precision_reference"] = {
            "available": False,
            "note": "no full-precision reference file; paper display check only",
        }
    else:
        try:
            reference_doc = json.loads(Path(reference).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            report["status"] = "fail"
            report["errors"].append(f"precision reference unreadable: {exc}")
            return report
        comparisons = precision_comparisons(reference_doc, loaded, report["aggregate"], p_raw, p_adjusted)
        report["precision_reference"] = {"available": True, "path": str(reference), "comparisons": comparisons}
        bad = [item for item in comparisons if not item["ok"]]
        if bad:
            report["status"] = "fail"
            report["errors"].extend(
                f"precision mismatch {item['quantity']}: computed {item['computed_full']} "
                f"vs reference {item['reference_full']}" for item in bad)
    return report


def precision_comparisons(reference_doc, loaded, aggregate, p_raw, p_adjusted):
    comparisons = []
    names = {"linkgda": "LinkGDA", "indigena": "INDIGENA"}
    ref_folds = reference_doc.get("folds")
    if not isinstance(ref_folds, dict):
        comparisons.append({"quantity": "reference.folds", "ok": False,
                            "note": "reference JSON has no folds section"})
        return comparisons
    for method in ("linkgda", "indigena"):
        entries = ref_folds.get(names[method])
        if not isinstance(entries, list) or len(entries) != len(loaded[method]):
            comparisons.append({"quantity": f"reference.folds.{names[method]}", "ok": False,
                                "note": "missing or wrong-length fold list"})
            continue
        for fold, entry in enumerate(entries):
            comparisons.append(precision_comparison(
                f"rq1.{method}.fold_{fold}.mr", loaded[method][fold]["mr"], entry["MR"]))
            comparisons.append(precision_comparison(
                f"rq1.{method}.fold_{fold}.pairs", loaded[method][fold]["pairs"], entry["pairs"]))
            comparisons.append(precision_comparison(
                f"rq1.{method}.fold_{fold}.candidates", loaded[method][fold]["candidates"], entry["genes"]))
    ref_test = reference_doc.get("test", {})
    means = ref_test.get("means", {})
    sds = ref_test.get("sd", {})
    for method in ("linkgda", "indigena"):
        if names[method] in means:
            comparisons.append(precision_comparison(
                f"rq1.{method}.mr_mean", aggregate[method]["mr_mean"], means[names[method]]))
        if names[method] in sds:
            comparisons.append(precision_comparison(
                f"rq1.{method}.mr_sample_sd", aggregate[method]["mr_sample_sd"], sds[names[method]]))
    if "one_sided_p" in ref_test:
        comparisons.append(precision_comparison("rq1.p_raw", p_raw, ref_test["one_sided_p"]))
    if "bonferroni_6_p" in ref_test:
        comparisons.append(precision_comparison("rq1.p_bonferroni", p_adjusted, ref_test["bonferroni_6_p"]))
    return comparisons


def run_excluded(results_dir, *, n_seeds=10, expected=None):
    """Recompute the Table 3 (excluded benchmark, full pool) check from seed files."""
    expected = DEFAULT_EXPECTED["excluded"] if expected is None else expected
    report = {
        "scope": SCIENTIFIC_SCOPE,
        "template_source": "code/analysis/excluded_table.py template, calibrated owl2vecstar func_expr cell, full pool",
        "expected": expected,
        "files": [],
        "query_alignment": {"reference_seed": 0, "seeds": []},
        "per_seed": [],
        "aggregate": {},
        "comparisons": [],
        "errors": [],
        "status": "pass",
    }
    loaded = []
    for seed in range(n_seeds):
        path = Path(results_dir) / EXCLUDED_TEMPLATE.format(seed=seed)
        try:
            entry = load_validated(path)
        except FileNotFoundError as exc:
            report["status"] = "unverified"
            report["errors"].append(str(exc))
            return report
        except (ValueError, OSError) as exc:
            report["status"] = "fail"
            report["errors"].append(f"input validation: {exc}")
            return report
        loaded.append(entry)
        report["files"].append(file_summary(entry))

    for seed, entry in enumerate(loaded):
        keys_match = entry["keys"] == loaded[0]["keys"]
        idx_match = bool(np.array_equal(entry["indices"], loaded[0]["indices"]))
        report["query_alignment"]["seeds"].append({"seed": seed, "case_keys_match": keys_match,
                                                   "true_indices_match": idx_match})
        if not (keys_match and idx_match):
            report["status"] = "fail"
            report["errors"].append(f"seed {seed}: case keys or true-gene indices differ from seed 0")
            return report
        report["per_seed"].append({"seed": seed, "mr": entry["mr"], "h10": entry["h10"],
                                   "pairs": entry["pairs"], "candidates": entry["candidates"]})

    for seed, entry in enumerate(loaded):
        checks = (
            ("pairs", expected["pairs"]),
            ("unique_genes", expected["unique_genes"]),
            ("unique_diseases", expected["unique_diseases"]),
            ("candidates", expected["candidate_pool"]),
        )
        for name, wanted in checks:
            if entry[name] != wanted:
                report["errors"].append(
                    f"seed {seed}: {name} is {entry[name]}, expected {wanted}")

    mrs = [entry["mr"] for entry in loaded]
    h10s = [entry["h10"] for entry in loaded]
    report["aggregate"] = {
        "mr_mean": float(np.mean(mrs)),
        "mr_sample_sd": float(np.std(mrs, ddof=1)),
        "h10_mean": float(np.mean(h10s)),
        "h10_sample_sd": float(np.std(h10s, ddof=1)),
    }
    comparisons = [
        display_comparison("excluded.mr_mean", report["aggregate"]["mr_mean"],
                           expected["mr_mean_display"], DISPLAY_DIGITS),
        display_comparison("excluded.mr_sample_sd", report["aggregate"]["mr_sample_sd"],
                           expected["mr_sd_display"], DISPLAY_DIGITS),
        display_comparison("excluded.h10_mean", report["aggregate"]["h10_mean"],
                           expected["h10_mean_display"], DISPLAY_DIGITS),
        display_comparison("excluded.h10_sample_sd", report["aggregate"]["h10_sample_sd"],
                           expected["h10_sd_display"], DISPLAY_DIGITS),
    ]
    report["comparisons"] = comparisons
    if report["errors"] or not all(item["ok"] for item in comparisons):
        report["status"] = "fail"
        report["errors"].extend(
            f"{item['quantity']}: displayed {item['display']} != expected {item['expected']}"
            for item in comparisons if not item["ok"])
    return report


def build_report(checks):
    if any(check["status"] == "fail" for check in checks.values()):
        status, code = "fail", 1
    elif any(check["status"] == "unverified" for check in checks.values()):
        status, code = "unverified", 2
    else:
        status, code = "pass", 0
    return {
        "status": status,
        "exit_code": code,
        "scientific_scope": SCIENTIFIC_SCOPE,
        "limitations": [
            "score files carry no candidate-identifier column; candidate-set identity across methods or seeds "
            "is not verifiable from these artifacts, only candidate counts, case-key order, and true-gene indices",
            "display checks compare fixed-point rounding of the recomputed value against the printed table value; "
            "they do not bound the underlying full-precision difference",
        ],
        "source_hashes": source_hashes(),
        "checks": checks,
    }


def source_hashes():
    hashes = {}
    for label, module in (("rq1_table", rq1_table), ("analysis.rq1_stats", rq1_stats)):
        path = getattr(module, "__file__", None)
        try:
            hashes[label] = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
        except OSError:
            hashes[label] = None
    own = globals().get("__file__")
    try:
        hashes["paper_results"] = hashlib.sha256(Path(own).read_bytes()).hexdigest() if own else None
    except OSError:
        hashes["paper_results"] = None
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Recompute the paper's RQ1 and excluded-benchmark tables from saved predictions.")
    parser.add_argument("--check", choices=("rq1", "excluded", "all"), default="all")
    parser.add_argument("--rq1-results", type=Path, default=None,
                        help="directory holding the RQ1 fold files (seed 0, folds 0-9)")
    parser.add_argument("--excluded-results", type=Path, default=None,
                        help="directory holding the excluded-benchmark files (fold 0, seeds 0-9)")
    parser.add_argument("--expected", type=Path, default=None,
                        help="JSON overriding DEFAULT_EXPECTED entries, keyed by check name")
    parser.add_argument("--precision-reference", type=Path, default=None,
                        help="full-precision RQ1 reference JSON, or 'none' to skip; "
                             "defaults to the point 3.6 recomputation if present")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the JSON report here instead of stdout")
    args = parser.parse_args(argv)

    if args.check in ("rq1", "all") and args.rq1_results is None:
        parser.error("--rq1-results is required with --check rq1 or all")
    if args.check in ("excluded", "all") and args.excluded_results is None:
        parser.error("--excluded-results is required with --check excluded or all")

    overrides = {}
    if args.expected is not None:
        overrides = json.loads(args.expected.read_text())

    checks = {}
    if args.check in ("rq1", "all"):
        checks["rq1"] = run_rq1(args.rq1_results, expected=overrides.get("rq1"),
                                precision_reference=args.precision_reference)
    if args.check in ("excluded", "all"):
        checks["excluded"] = run_excluded(args.excluded_results, expected=overrides.get("excluded"))

    report = build_report(checks)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
