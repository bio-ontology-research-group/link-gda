"""Compare per-gene calibration under two reference panels.

The reported calibration estimates each gene's mean and spread leave-one-out over
the other queries in the evaluation set. A deployable alternative estimates them
once over the training diseases; kge_transd.py --write_baselines writes those as
a per-gene (mean, sd) vector file. calibrate_scores.py cannot consume that file:
its --baseline file option expects a full score matrix and derives the statistics
itself. This script applies the saved vectors directly and reports both readouts
side by side, so the question of whether the panel matters can be answered from
artifacts that already exist.

Ranks use the deterministic average-rank convention the paper reports, not the
optimistic convention in calibrate_scores.py and p_value_per_fold.py.

    python compare_calibration_panels.py --results-glob 'data/results/kge_results_transd_fold_*_CFG_bma.tsv' \
        --baseline-glob 'data/results/baselines_transd_fold_*_CFG.tsv'
"""
import glob
import re

import click as ck
import numpy as np


def fold_of(path):
    m = re.search(r"fold_(\d+)", path)
    return int(m.group(1)) if m else None


def load_scores(path):
    rows = []
    targets = []
    for line in open(path):
        q = line.rstrip("\n").split("\t")
        if len(q) < 4:
            continue
        targets.append(int(q[2]))
        rows.append(np.asarray(q[3:], dtype=np.float64))
    return np.vstack(rows), np.asarray(targets, dtype=int)


def load_vectors(path):
    mean = []
    sd = []
    with open(path) as fh:
        header = fh.readline()
        if "gene_index" not in header:
            fh.seek(0)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            mean.append(float(parts[1]))
            sd.append(float(parts[2]))
    return np.asarray(mean), np.asarray(sd)


def average_ranks(scores, targets):
    true_scores = scores[np.arange(scores.shape[0]), targets][:, None]
    greater = (scores > true_scores).sum(axis=1)
    tied = (scores == true_scores).sum(axis=1)
    return 1.0 + greater + (tied - 1) / 2.0


def loo_calibrate(scores):
    n = scores.shape[0]
    if n < 3:
        return scores
    total = scores.sum(axis=0, keepdims=True)
    total_sq = (scores ** 2).sum(axis=0, keepdims=True)
    mean = (total - scores) / (n - 1)
    var = (total_sq - scores ** 2) / (n - 1) - mean ** 2
    return (scores - mean) / (np.sqrt(np.clip(var, 0, None)) + 1e-12)


def panel_calibrate(scores, mean, sd):
    return (scores - mean[None, :]) / (sd[None, :] + 1e-12)


def summarize(ranks):
    return {"mr": ranks.mean(), "mrr": (1.0 / ranks).mean(),
            "h10": (ranks <= 10).mean(), "h100": (ranks <= 100).mean()}


@ck.command()
@ck.option("--results-glob", required=True, help="Glob for the per-fold prediction TSVs")
@ck.option("--baseline-glob", required=True, help="Glob for the matching per-fold vector TSVs")
@ck.option("--label", default="", help="Name printed with the summary")
def main(results_glob, baseline_glob, label):
    results = {fold_of(p): p for p in sorted(glob.glob(results_glob))}
    vectors = {fold_of(p): p for p in sorted(glob.glob(baseline_glob))}
    folds = sorted(set(results) & set(vectors))
    missing = sorted(set(results) ^ set(vectors))
    if not folds:
        raise SystemExit(f"no matching folds\n  results: {len(results)}\n  vectors: {len(vectors)}")

    per_fold = {"raw": [], "loo": [], "panel": []}
    for fold in folds:
        scores, targets = load_scores(results[fold])
        mean, sd = load_vectors(vectors[fold])
        if mean.shape[0] != scores.shape[1]:
            raise SystemExit(
                f"fold {fold}: vector length {mean.shape[0]} != candidate count {scores.shape[1]}")
        per_fold["raw"].append(summarize(average_ranks(scores, targets)))
        per_fold["loo"].append(summarize(average_ranks(loo_calibrate(scores), targets)))
        per_fold["panel"].append(summarize(average_ranks(panel_calibrate(scores, mean, sd), targets)))

    print(f"=== {label or results_glob}")
    print(f"folds used: {folds}")
    if missing:
        print(f"folds without a pair: {missing}")
    print(f"{'setting':<28}{'MR':>10}{'MRR':>9}{'H@10':>8}{'H@100':>8}")
    for key, name in (("raw", "uncalibrated"),
                      ("loo", "calibrated, leave-one-out"),
                      ("panel", "calibrated, training panel")):
        mr = np.array([f["mr"] for f in per_fold[key]])
        mrr = np.array([f["mrr"] for f in per_fold[key]])
        h10 = np.array([f["h10"] for f in per_fold[key]])
        h100 = np.array([f["h100"] for f in per_fold[key]])
        print(f"{name:<28}{mr.mean():>10.2f}{mrr.mean():>9.4f}{h10.mean():>8.4f}{h100.mean():>8.4f}")

    loo_mr = np.array([f["mr"] for f in per_fold["loo"]])
    panel_mr = np.array([f["mr"] for f in per_fold["panel"]])
    diff = panel_mr - loo_mr
    print(f"panel minus leave-one-out, mean rank: {diff.mean():+.2f} "
          f"(sd {diff.std(ddof=1):.2f}, max |diff| {np.abs(diff).max():.2f})")
    print(f"per-fold differences: {' '.join(f'{d:+.2f}' for d in diff)}")


if __name__ == "__main__":
    main()
