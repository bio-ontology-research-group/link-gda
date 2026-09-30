"""Compute the five two-sided RQ2 comparisons from matched fold means."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


COMPARISONS = {
    "f_vs_p": ("LinkGDA-f", "LinkGDA-p"),
    "fs_vs_p": ("LinkGDA-fs", "LinkGDA-p"),
    "s_vs_ultra_s": ("LinkGDA-s", "ULTRA-s"),
    "f_vs_ultra_f": ("LinkGDA-f", "ULTRA-f"),
    "fs_vs_ultra_fs": ("LinkGDA-fs", "ULTRA-fs"),
}
FOLD_IDS = tuple(str(fold) for fold in range(10))
BONFERRONI_FAMILY_SIZE = 6


def _fold_vector(value, comparison, side):
    if not isinstance(value, dict):
        raise ValueError(f"{comparison}.{side} must be an object keyed by fold ID")
    if set(value) != set(FOLD_IDS):
        missing = sorted(set(FOLD_IDS) - set(value))
        extra = sorted(set(value) - set(FOLD_IDS))
        raise ValueError(f"{comparison}.{side} must contain exactly folds 0..9; missing={missing}, extra={extra}")
    vector = []
    for fold in FOLD_IDS:
        item = value[fold]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"{comparison}.{side}[{fold}] must be a finite number")
        vector.append(float(item))
    return np.asarray(vector, dtype=np.float64)


def corrected_test(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != (10,) or right.shape != (10,):
        raise ValueError("left and right must each be a one-dimensional vector of ten folds")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("left and right must contain only finite values")
    difference = left - right
    mean = float(difference.mean())
    sample_sd = float(difference.std(ddof=1))
    correction = 1 / 10 + 1 / 9
    standard_error = sample_sd * math.sqrt(correction)
    if standard_error == 0:
        raise ValueError("corrected t-test is undefined when paired differences have zero variance")
    t_statistic = mean / standard_error
    p_two_sided = float(stats.t.sf(abs(t_statistic), 9) * 2)
    return {
        "mean_difference": mean,
        "sample_sd_difference": sample_sd,
        "correction_factor": correction,
        "corrected_standard_error": standard_error,
        "t": t_statistic,
        "df": 9,
        "alternative": "two-sided",
        "p_two_sided": p_two_sided,
        "bonferroni_family_size": BONFERRONI_FAMILY_SIZE,
        "p_bonferroni": min(1.0, BONFERRONI_FAMILY_SIZE * p_two_sided),
    }


def calculate(document, sha256):
    if not isinstance(document, dict):
        raise ValueError("input must be a JSON object")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("metadata must be a non-empty object describing input provenance")
    supplied = document.get("comparisons")
    if not isinstance(supplied, dict) or set(supplied) != set(COMPARISONS):
        raise ValueError(f"comparisons must contain exactly: {', '.join(COMPARISONS)}")
    results = {}
    for name, (left_name, right_name) in COMPARISONS.items():
        comparison = supplied[name]
        if not isinstance(comparison, dict) or set(comparison) != {"left", "right"}:
            raise ValueError(f"{name} must contain exactly left and right fold maps")
        left = _fold_vector(comparison["left"], name, "left")
        right = _fold_vector(comparison["right"], name, "right")
        results[name] = {
            "left_method": left_name,
            "right_method": right_name,
            "difference_definition": f"{left_name} MR minus {right_name} MR; negative favors {left_name}",
            "fold_ids": list(FOLD_IDS),
            "test": corrected_test(left, right),
        }
    return {
        "protocol": {
            "unit": "matched fold-level mean rank",
            "folds": 10,
            "test": "two-sided Nadeau-Bengio corrected resampled t-test",
            "correction": "1/10 + 1/9",
            "multiplicity": "Bonferroni over one RQ1 and five RQ2 comparisons",
            "alignment_limit": "Fold-level means cannot prove that underlying cases were matched; input provenance must establish this.",
        },
        "input": {"sha256": sha256, "metadata": metadata},
        "comparisons": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    document = json.loads(raw)
    print(json.dumps(
        calculate(document, hashlib.sha256(raw).hexdigest()),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
