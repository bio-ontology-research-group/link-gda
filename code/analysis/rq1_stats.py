"""Recompute calibrated fold mean ranks and the prespecified RQ1 tests."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


CONFIGS = {
    "owl2vecstar": {
        "linkgda": "kge_results_transd_fold_{fold}_seed_0_dim_200_bs_65536_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_True_tol_15_calsel_by_graph_bma.tsv",
        "indigena": "kge_results_transd_fold_{fold}_seed_0_dim_400_bs_65536_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_tol_15_calsel_inductive_bma.tsv",
    },
    "gda_projector": {
        "linkgda": "kge_results_transd_fold_{fold}_seed_0_dim_200_bs_65536_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_tol_15_calsel_by_graph_bma.tsv",
        "indigena": "kge_results_transd_fold_{fold}_seed_0_dim_400_bs_65536_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_False_tol_15_calsel_inductive_bma.tsv",
    },
}


def load(path):
    scores = []
    indices = []
    keys = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            parts = raw.rstrip(b"\n").split(b"\t")
            if len(parts) < 4:
                continue
            keys.append((parts[0].decode(), parts[1].decode()))
            indices.append(int(parts[2]))
            scores.append(np.fromiter((float(value) for value in parts[3:]), dtype=np.float64))
    matrix = np.vstack(scores)
    return matrix, np.asarray(indices), keys, digest.hexdigest()


def calibrate(scores):
    n_queries = scores.shape[0]
    means = (scores.sum(axis=0, keepdims=True) - scores) / (n_queries - 1)
    variances = ((scores ** 2).sum(axis=0, keepdims=True) - scores ** 2) / (n_queries - 1) - means ** 2
    return (scores - means) / (np.sqrt(np.clip(variances, 0, None)) + 1e-12)


def mean_rank(scores, indices):
    ranks = []
    for row, true_index in zip(scores, indices):
        true_score = row[true_index]
        greater = int(np.count_nonzero(row > true_score))
        equal = int(np.count_nonzero(row == true_score))
        ranks.append(greater + (equal + 1) / 2)
    return float(np.mean(ranks))


def corrected_test(indigena, linkgda):
    difference = np.asarray(indigena) - np.asarray(linkgda)
    k = len(difference)
    sd = float(difference.std(ddof=1))
    correction = 1 / k + 1 / (k - 1)
    standard_error = sd * math.sqrt(correction)
    mean = float(difference.mean())
    t_statistic = mean / standard_error
    degrees_freedom = k - 1
    p_one_sided = float(stats.t.sf(t_statistic, degrees_freedom))
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    ci = [mean - critical * standard_error, mean + critical * standard_error]
    effect_size_dz = mean / sd
    return {
        "difference_definition": "INDIGENA MR minus LinkGDA-pfs MR; positive favors LinkGDA-pfs",
        "mean_difference": mean,
        "sample_sd_difference": sd,
        "folds_linkgda_better": int(np.count_nonzero(difference > 0)),
        "folds_tied": int(np.count_nonzero(difference == 0)),
        "k": k,
        "correction_factor": correction,
        "corrected_standard_error": standard_error,
        "t": t_statistic,
        "df": degrees_freedom,
        "alternative": "one-sided: mean difference > 0",
        "p_one_sided": p_one_sided,
        "ci_95_definition": "two-sided 95% t interval using the Nadeau-Bengio corrected standard error",
        "ci_95": ci,
        "paired_effect_size_definition": "Cohen's dz = mean paired difference / sample SD of paired differences",
        "cohens_dz": effect_size_dz,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "protocol": {
            "unit": "fold-level mean rank",
            "folds": 10,
            "setting": "leave-one-out per-gene calibrated, calibrated-selected arm",
            "rank_ties": "deterministic average rank",
            "test": "one-sided Nadeau-Bengio corrected resampled t-test",
            "correction": "1/k + n_test/n_train = 1/10 + 1/9",
        },
        "manifest": [],
        "projections": {},
    }
    for projection, methods in CONFIGS.items():
        vectors = {"linkgda": [], "indigena": []}
        for fold in range(10):
            loaded = {}
            for method, template in methods.items():
                path = args.results_dir / template.format(fold=fold)
                scores, indices, keys, sha256 = load(path)
                loaded[method] = (scores, indices, keys)
                vectors[method].append(mean_rank(calibrate(scores), indices))
                stat = path.stat()
                output["manifest"].append({
                    "projection": projection,
                    "method": method,
                    "fold": fold,
                    "path": str(path),
                    "bytes": stat.st_size,
                    "sha256": sha256,
                    "queries": scores.shape[0],
                    "candidates": scores.shape[1],
                })
            if loaded["linkgda"][2] != loaded["indigena"][2]:
                raise ValueError(f"query order differs for {projection}, fold {fold}")
            if not np.array_equal(loaded["linkgda"][1], loaded["indigena"][1]):
                raise ValueError(f"true-gene indices differ for {projection}, fold {fold}")
        output["projections"][projection] = {
            "fold_mr": vectors,
            "aggregate": {
                method: {
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)),
                }
                for method, values in vectors.items()
            },
            "test": corrected_test(vectors["indigena"], vectors["linkgda"]),
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
