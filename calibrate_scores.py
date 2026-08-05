"""Per-gene score calibration for the GDA ranking.

Why this exists
---------------
The trained model encodes a near-binary "does this gene appear in the supervised
association edges" signal, and that signal shifts a gene's score for EVERY query by
roughly the same amount. It is a per-gene offset, not disease-specific evidence. Because
the 302 candidates with no training association are a small fraction of the 4,399-gene
pool but supply about half of the test pairs, the sign of that offset dominates mean rank,
and which sign a run lands on is decided by the random seed.

Calibration removes the offset. For each gene we estimate a baseline -- what this gene
scores for a typical query -- and subtract it, so what remains is how much this disease
raises the gene above its own norm. A gene that scores high for every disease carries no
information about any particular one.

Baselines
---------
--baseline loo   (default) leave-one-out mean over the other queries in the same file.
                 The pair being scored never contributes to its own baseline.
--baseline file  mean over the queries in a separate file, e.g. the validation results.
                 Stricter: nothing from the test set enters the score.

Modes
-----
--mode center    subtract the baseline (removes an additive offset)
--mode zscore    also divide by the gene's spread (removes scale differences too)
--mode both      report both alongside the raw ranks

Ranks use the optimistic convention (1 + number of candidates scoring strictly higher),
matching p_value_per_fold.py.

    python calibrate_scores.py --results data/results/kge_results_....tsv
    python calibrate_scores.py --results ... --train data/folds/fold_0/train.csv   # stratify
    python calibrate_scores.py --results ... --example                             # worked example
"""
import csv
import collections

import click as ck
import numpy as np


def load_scores(path):
    """Return (score matrix, true-gene index per row, gene name per row)."""
    scores, true_idx, genes = [], [], []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            genes.append(parts[0])
            true_idx.append(int(parts[2]))
            scores.append(np.fromiter((float(x) for x in parts[3:]), dtype=np.float64))
    return np.vstack(scores), np.array(true_idx), genes


def baselines(scores, external=None):
    """Per-gene baseline, shaped to broadcast against the score matrix."""
    if external is not None:
        return external.mean(axis=0, keepdims=True), external.std(axis=0, keepdims=True) + 1e-12
    n = scores.shape[0]
    if n < 2:
        raise ValueError("leave-one-out needs at least two queries; use --baseline file")
    # leave-one-out mean and standard deviation, computed without loops
    total = scores.sum(axis=0, keepdims=True)
    loo_mean = (total - scores) / (n - 1)
    total_sq = (scores ** 2).sum(axis=0, keepdims=True)
    loo_var = (total_sq - scores ** 2) / (n - 1) - loo_mean ** 2
    return loo_mean, np.sqrt(np.clip(loo_var, 0, None)) + 1e-12


def metrics(scores, true_idx, mask=None):
    rows = range(scores.shape[0]) if mask is None else np.flatnonzero(mask)
    ranks = np.array([1 + int((scores[i] > scores[i, true_idx[i]]).sum()) for i in rows])
    if ranks.size == 0:
        return None
    return {"n": ranks.size, "mr": ranks.mean(), "mrr": (1 / ranks).mean(),
            "h10": (ranks <= 10).mean(), "h100": (ranks <= 100).mean()}


def show(label, m):
    if m is None:
        return
    print(f"  {label:<26} n={m['n']:<5} MR {m['mr']:8.1f}   MRR {m['mrr']:.4f}   "
          f"H@10 {m['h10']:.4f}   H@100 {m['h100']:.4f}")


@ck.command()
@ck.option("--results", required=True, help="Per-instance score TSV to calibrate")
@ck.option("--baseline", type=ck.Choice(["loo", "file"]), default="loo")
@ck.option("--baseline-file", default=None, help="Score TSV supplying the baseline when --baseline file")
@ck.option("--mode", type=ck.Choice(["center", "zscore", "both"]), default="both")
@ck.option("--train", default=None, help="Fold train.csv; stratifies by whether the gene has a training association")
@ck.option("--example", is_flag=True, help="Show a worked example for one test pair")
def main(results, baseline, baseline_file, mode, train, example):
    scores, true_idx, genes = load_scores(results)
    print(f"{scores.shape[0]} queries x {scores.shape[1]} candidates\n")

    external = None
    if baseline == "file":
        if not baseline_file:
            raise SystemExit("--baseline file requires --baseline-file")
        external, _, _ = load_scores(baseline_file)
        if external.shape[1] != scores.shape[1]:
            raise SystemExit("baseline file has a different candidate pool")
    mean, spread = baselines(scores, external)

    variants = [("raw", scores)]
    if mode in ("center", "both"):
        variants.append(("calibrated (centred)", scores - mean))
    if mode in ("zscore", "both"):
        variants.append(("calibrated (z-scored)", (scores - mean) / spread))

    seen_mask = None
    if train:
        deg = collections.Counter()
        with open(train) as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
            col = header.index("Gene")
            for row in reader:
                if row:
                    deg[row[col]] += 1
        seen_mask = np.array([g in deg for g in genes])
        print(f"true gene has a training association: {seen_mask.sum()} of {len(genes)} queries\n")

    for label, matrix in variants:
        print(label)
        show("all", metrics(matrix, true_idx))
        if seen_mask is not None:
            show("gene seen in training", metrics(matrix, true_idx, seen_mask))
            show("gene unseen in training", metrics(matrix, true_idx, ~seen_mask))
        print()

    if example:
        centred = scores - mean
        row = int(np.argmax([1 + int((scores[i] > scores[i, true_idx[i]]).sum())
                             - (1 + int((centred[i] > centred[i, true_idx[i]]).sum()))
                             for i in range(scores.shape[0])]))
        g = true_idx[row]
        print(f"worked example: query {row}, true gene column {g} ({genes[row]})")
        print(f"  raw score for the true gene        {scores[row, g]:+.4f}")
        print(f"  this gene's baseline over queries  {mean[0, g]:+.4f}")
        print(f"  calibrated score                   {centred[row, g]:+.4f}")
        print(f"  rank raw {1 + int((scores[row] > scores[row, g]).sum()):5d}"
              f"   ->  calibrated {1 + int((centred[row] > centred[row, g]).sum()):5d}")


if __name__ == "__main__":
    main()
