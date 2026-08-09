"""Ranking accuracy stratified by how often the true gene was seen in training.

This consolidates the ad-hoc probes behind the memorisation analysis into one
reproducible pass. It answers two questions from a set of saved score files:

  1. Does accuracy depend on the true gene's training association degree? A method
     that ranks well only for genes it already saw associated with some disease is
     recalling the training panel, not reasoning about the query.

  2. How large is the membership prior? Independently of any query, the scoring
     function assigns systematically higher scores to genes carrying supervised
     association edges. The gap between those two candidate groups, measured in
     units of the per-query score spread, is that prior. Calibration is expected to
     remove it, so running with and without --calibrate quantifies what it removes.

Degree is computed from the fold's own train.csv, so it never touches test data.
Ranks use average-rank tie handling, matching rq1_table.py and evaluate_sem_sim.py.

Run from the repository root, where data/ lives:

    python stratified_metrics.py --scores "data/results/transd_fold_{f}_dim_400.txt"
    python stratified_metrics.py --scores "..." --calibrate
"""
import argparse
import os

import numpy as np
import pandas as pd

BUCKETS = [(0, 0, "unseen (0)"), (1, 1, "1"), (2, 4, "2-4"), (5, 10 ** 9, "5+")]


def load_pool():
    gd = pd.read_csv("data/gene_diseases.csv")
    genes = sorted(set(gd["Gene"]))
    return genes, {g: i for i, g in enumerate(genes)}


def load_scores(path):
    genes, idx, scores = [], [], []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            genes.append(parts[0])
            idx.append(int(parts[2]))
            scores.append(np.fromiter((float(x) for x in parts[3:]), dtype=np.float64))
    return genes, np.array(idx), np.vstack(scores)


def calibrate(scores):
    n = scores.shape[0]
    mean = (scores.sum(axis=0, keepdims=True) - scores) / (n - 1)
    var = ((scores ** 2).sum(axis=0, keepdims=True) - scores ** 2) / (n - 1) - mean ** 2
    return (scores - mean) / (np.sqrt(np.clip(var, 0, None)) + 1e-12)


def ranks_of(scores, idx):
    out = np.empty(scores.shape[0], dtype=np.float64)
    for i in range(scores.shape[0]):
        true = scores[i, idx[i]]
        greater = int((scores[i] > true).sum())
        equal = int((scores[i] == true).sum())
        out[i] = greater + (equal + 1) / 2.0
    return out


def degree_vector(genes, train_path):
    tr = pd.read_csv(train_path, sep="\t")
    counts = tr.groupby("Gene")["Disease"].nunique()
    return np.array([counts.get(g, 0) for g in genes], dtype=np.int64)


def membership_gap(scores, candidate_degree):
    seen = candidate_degree >= 1
    if not seen.any() or seen.all():
        return float("nan")
    per_row_sd = scores.std(axis=1, ddof=1)
    per_row_sd = np.where(per_row_sd > 0, per_row_sd, np.nan)
    gap = scores[:, seen].mean(axis=1) - scores[:, ~seen].mean(axis=1)
    return float(np.nanmean(gap / per_row_sd))


def summarise(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    return values.mean(), values.std(ddof=1) if values.size > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True,
                    help="score-file template containing {f} for the fold")
    ap.add_argument("--calibrate", action="store_true",
                    help="apply leave-one-out per-gene z-scoring before ranking")
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()

    genes, gene_index = load_pool()
    n_pool = len(genes)

    per_bucket = {label: [] for _, _, label in BUCKETS}
    overall, gaps, seen_unseen = [], [], []
    used = 0

    for fold in range(args.folds):
        path = args.scores.format(f=fold)
        if not os.path.exists(path):
            continue
        row_genes, idx, scores = load_scores(path)
        if scores.shape[1] != n_pool:
            raise ValueError(
                f"{path}: score vectors have {scores.shape[1]} entries but the candidate "
                f"pool from data/gene_diseases.csv has {n_pool}. The score files and the "
                f"pool disagree; check that this file was produced by the current pipeline."
            )
        candidate_degree = degree_vector(genes, f"data/folds/fold_{fold}/train.csv")
        gaps.append(membership_gap(scores, candidate_degree))

        if args.calibrate:
            scores = calibrate(scores)
        ranks = ranks_of(scores, idx)
        overall.append(ranks.mean())

        true_degree = np.array(
            [candidate_degree[gene_index[g]] if g in gene_index else -1 for g in row_genes]
        )
        for low, high, label in BUCKETS:
            mask = (true_degree >= low) & (true_degree <= high)
            if mask.any():
                per_bucket[label].append(
                    (ranks[mask].mean(), (1.0 / ranks[mask]).mean(),
                     (ranks[mask] <= 10).mean(), int(mask.sum()))
                )
        unseen = true_degree == 0
        if unseen.any() and (~unseen).any():
            seen_unseen.append(ranks[unseen].mean() - ranks[~unseen].mean())
        used += 1

    if used == 0:
        raise SystemExit(f"No score files matched {args.scores}")

    setting = "calibrated" if args.calibrate else "uncalibrated"
    print(f"{args.scores}\nsetting: {setting}   folds found: {used}   pool N={n_pool}")
    print(f"random baseline MR = {(n_pool + 1) / 2:.1f}\n")

    print(f"{'training degree':<16}{'cases/fold':>12}{'MR':>18}{'MRR':>10}{'H@10':>9}")
    for _, _, label in BUCKETS:
        rows = per_bucket[label]
        if not rows:
            print(f"{label:<16}{'--':>12}")
            continue
        mr, mr_sd = summarise([r[0] for r in rows])
        mrr, _ = summarise([r[1] for r in rows])
        h10, _ = summarise([r[2] for r in rows])
        cases, _ = summarise([r[3] for r in rows])
        print(f"{label:<16}{cases:>12.1f}{mr:>12.1f}±{mr_sd:<5.1f}{mrr:>10.4f}{h10:>9.4f}")

    mr, mr_sd = summarise(overall)
    print(f"\n{'all':<16}{'':>12}{mr:>12.1f}±{mr_sd:<5.1f}")

    gap_mean, gap_sd = summarise(gaps)
    print(f"\nmembership prior (raw scores, unaffected by --calibrate):")
    print(f"  seen-minus-unseen candidate score gap = {gap_mean:+.3f}±{gap_sd:.3f} "
          f"per-query standard deviations")
    if seen_unseen:
        d_mean, d_sd = summarise(seen_unseen)
        print(f"  unseen-minus-seen mean rank ({setting}) = {d_mean:+.1f}±{d_sd:.1f}")
    if len(gaps) > 2 and len(overall) == len(gaps):
        finite = np.isfinite(gaps)
        if finite.sum() > 2:
            r = np.corrcoef(np.array(gaps)[finite], np.array(overall)[finite])[0, 1]
            print(f"  correlation across folds between the gap and mean rank = {r:+.3f}")


if __name__ == "__main__":
    main()
