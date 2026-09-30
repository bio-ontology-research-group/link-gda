"""Machine-readable numerical outputs from a fixed synthetic score fixture.

Runs the actual production functions (rq1_table, calibrate_scores,
evaluate_sem_sim, analysis/rq1_stats) on one small deterministic fixture and
prints a JSON document. The fixture is synthetic: it validates behavior only,
it does not reproduce any paper number. The JSON marks this explicitly.

Any production function that fails, or a required dependency that is missing,
is recorded as a gap and makes this script exit non-zero; it is never
swallowed into a passing record. Pre-existing, already-characterized issues
are kept in a separate "known_issues" section so they are neither hidden
nor presented as fixed.

    python code/reproduce/fixture_metrics.py
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

KNOWN_ISSUES = [
    (
        "Real fold score artifacts (kge_results_transd_fold_*.tsv) are absent from "
        "this checkout, so rq1_stats is exercised only on synthetic fold vectors, "
        "not on any real fold file. Known issue, not fixed here; the silent "
        "missing-fold behavior is characterized in tests/test_rq1_stats.py."
    )
]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value
    return float(value)


def build_outputs(root=None):
    """Run the production functions on the fixed fixture and return the result dict."""
    root = Path(root) if root else REPO_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    gaps = []
    outputs = {}

    try:
        import numpy as np

        import calibrate_scores
        import rq1_table
    except Exception as exc:
        gaps.append(f"core dependencies unavailable: {exc!r}")
        return _payload(gaps, {}, {})

    fixture = {
        "description": (
            "3 synthetic queries x 4 synthetic candidates; includes an untied rank 1, "
            "another untied rank 1, and an all-tied row. Deterministic constant; not paper data."
        ),
        "scores": [[0.9, 0.5, 0.2, 0.1], [0.1, 0.2, 0.5, 0.9], [0.5, 0.5, 0.5, 0.5]],
        "true_index": [0, 3, 0],
        "keys": [["g0", "d0"], ["g1", "d1"], ["g2", "d2"]],
        "fold_mr_linkgda": [2.0, 2.2, 1.8, 2.4, 2.1, 1.9, 2.3, 2.0, 2.6, 2.2],
        "fold_mr_indigena": [3.0, 3.1, 2.9, 3.3, 3.0, 2.8, 3.2, 3.0, 3.4, 3.1],
    }

    scores = np.asarray(fixture["scores"], dtype=np.float64)
    true_index = np.asarray(fixture["true_index"], dtype=int)

    outputs["rq1_table_metrics_uncalibrated"] = rq1_table.metrics(scores, true_index)
    outputs["rq1_table_metrics_calibrated"] = rq1_table.metrics(rq1_table.calibrate(scores), true_index)
    outputs["rq1_table_calibrate"] = rq1_table.calibrate(scores).tolist()

    try:
        loo_mean, loo_spread = calibrate_scores.baselines(scores)
        outputs["calibrate_scores_loo"] = {
            "mean": loo_mean.tolist(),
            "spread": loo_spread.tolist(),
        }
    except Exception as exc:
        gaps.append(f"calibrate_scores.baselines failed: {exc!r}")

    if not hasattr(np, "trapezoid"):
        gaps.append(
            f"evaluate_sem_sim.compute_rank_roc not executable here: numpy {np.__version__} "
            "lacks np.trapezoid (requires numpy>=2.0); the trapezoid AUC is only "
            "characterized by the independent reference in tests/test_auc_definitions.py"
        )

    try:
        import evaluate_sem_sim
    except Exception as exc:
        gaps.append(f"evaluate_sem_sim unavailable: {exc!r}")
    else:
        if hasattr(np, "trapezoid"):
            try:
                ranks = {}
                for i in range(scores.shape[0]):
                    row = scores[i]
                    true = row[true_index[i]]
                    rank = float((row > true).sum()) + (float((row == true).sum()) + 1.0) / 2.0
                    ranks[rank] = ranks.get(rank, 0) + 1
                outputs["evaluate_sem_sim_auc_trapezoid"] = evaluate_sem_sim.compute_rank_roc(
                    ranks, scores.shape[1]
                )
            except Exception as exc:
                gaps.append(f"evaluate_sem_sim.compute_rank_roc failed: {exc!r}")
        try:
            rows = [
                (keys[0], keys[1], int(true_index[i]), scores[i].tolist())
                for i, keys in enumerate(fixture["keys"])
            ]
            micro, macro = evaluate_sem_sim.compute_metrics_from_rows(rows)
            outputs["evaluate_sem_sim_metrics_rows"] = {
                "micro": _json_safe(micro),
                "macro": _json_safe(macro),
            }
        except Exception as exc:
            gaps.append(f"evaluate_sem_sim.compute_metrics_from_rows failed: {exc!r}")

    rq1_stats = None
    try:
        spec = importlib.util.spec_from_file_location(
            "rq1_stats_under_test", root / "analysis" / "rq1_stats.py"
        )
        rq1_stats = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rq1_stats)
    except Exception as exc:
        gaps.append(f"rq1_stats unavailable: {exc!r}")

    if rq1_stats is not None:
        try:
            outputs["rq1_stats_mean_rank_uncalibrated"] = rq1_stats.mean_rank(scores, true_index)
            outputs["rq1_stats_mean_rank_calibrated"] = rq1_stats.mean_rank(
                rq1_stats.calibrate(scores), true_index
            )
            outputs["rq1_stats_corrected_test"] = rq1_stats.corrected_test(
                np.asarray(fixture["fold_mr_indigena"]), np.asarray(fixture["fold_mr_linkgda"])
            )
        except Exception as exc:
            gaps.append(f"rq1_stats computation failed: {exc!r}")

    return _payload(gaps, fixture, outputs)


def _payload(gaps, fixture, outputs):
    provenance = {
        "type": "synthetic_fixture",
        "reproduces_paper": False,
        "note": (
            "End-to-end synthetic pipeline: validates that the production functions run and "
            "stay numerically stable across the refactor. It is NOT a reproduction of any "
            "paper table; real prediction artifacts are absent from this checkout."
        ),
    }
    payload = {
        "provenance": provenance,
        "fixture": fixture,
        "outputs": outputs,
        "gaps": gaps,
        "known_issues": KNOWN_ISSUES,
        "ok": not gaps,
    }
    return payload


def main():
    result = build_outputs()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
