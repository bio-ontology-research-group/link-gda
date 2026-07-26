"""Overlap-stratified ranking for the semantic-similarity baselines (paper's aggregation).

Same per-fold-then-across-fold scheme and overlap definition as
leakage_overlap_perfold.py, applied to the baseline result files in
data/baseline_results/ (format: gene, disease, true-index, then scores).

Run on the workstation, where the baseline_results live:
    python sem_sim_overlap.py
"""
from collections import defaultdict

import numpy as np
import pandas as pd

N_FOLDS = 10
N_POOL = 4399
HITS_K = (1, 3, 10, 100)
BASE = "data/baseline_results"

# display name -> file config stem (files are {stem}_fold{N}_results.txt)
METHODS = {
    "SimGIC":     "resnik_simgic",
    "Resnik-BMA": "resnik_resnik_bma",
    "Resnik-BMM": "resnik_resnik_bmm",
    "Lin-BMA":    "resnik_lin_bma",
    "Lin-BMM":    "resnik_lin_bmm",
}


def per_instance_ranks(stem):
    ranks = {}
    for fold in range(N_FOLDS):
        path = f"{BASE}/{stem}_fold{fold}_results.txt"
        try:
            fh = open(path)
        except FileNotFoundError:
            return None
        with fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                pos = int(p[2])
                sc = np.asarray(p[3:], dtype=float)
                greater = int(np.count_nonzero(sc > sc[pos]))
                equal = int(np.count_nonzero(sc == sc[pos]))  # includes the true gene
                ranks[(fold, p[1], p[0])] = greater + (equal + 1) / 2.0  # average rank
    return ranks


def compute_overlap():
    dp = pd.read_csv("data/disease_phenotypes.csv")
    d2p = dp.groupby("Disease")["Phenotype"].apply(set).to_dict()
    overlap = {}
    for fold in range(N_FOLDS):
        tr = pd.read_csv(f"data/folds/fold_{fold}/train.csv", sep="\t")
        te = pd.read_csv(f"data/folds/fold_{fold}/test.csv", sep="\t")
        cp = defaultdict(set)
        for g, d in zip(tr["Gene"], tr["Disease"]):
            cp[g] |= d2p.get(d, set())
        for g, d in zip(te["Gene"], te["Disease"]):
            P = d2p.get(d, set())
            if not P:
                continue
            overlap[(fold, d, g)] = len(P & cp.get(g, set())) / len(P)
    return overlap


def fold_metrics(ranks, overlap, mask_fn):
    rows = []
    for fold in range(N_FOLDS):
        v = np.array([ranks[k] for k in ranks
                      if k[0] == fold and k in overlap and mask_fn(overlap[k])], dtype=float)
        if len(v) == 0:
            continue
        m = {"MR": v.mean(), "MRR": (1.0 / v).mean(), "AUC": (N_POOL - v.mean()) / (N_POOL - 1)}
        for k in HITS_K:
            m[f"H@{k}"] = (v <= k).mean()
        rows.append(m)
    keys = ["MR", "MRR"] + [f"H@{k}" for k in HITS_K] + ["AUC"]
    return {key: (np.mean([r[key] for r in rows]), np.std([r[key] for r in rows], ddof=1)) for key in keys}


def main():
    overlap = compute_overlap()
    strata = [("zero overlap", lambda o: o == 0),
              ("some overlap", lambda o: o > 0),
              ("all",          lambda o: True)]
    cols = ["MR", "MRR", "H@1", "H@3", "H@10", "H@100", "AUC"]
    for name, stem in METHODS.items():
        r = per_instance_ranks(stem)
        if r is None:
            print(f"{name}: SKIPPED (missing {stem}_fold*_results.txt)")
            continue
        print(f"=== {name} ===")
        print(f"{'subset':<13} " + " ".join(f"{c:>13}" for c in cols))
        for label, fn in strata:
            m = fold_metrics(r, overlap, fn)
            def cell(c):
                mean, sd = m[c]
                return f"{mean:7.2f}+/-{sd:5.2f}" if c == "MR" else f"{mean:7.3f}+/-{sd:5.3f}"
            print(f"{label:<13} " + " ".join(f"{cell(c):>13}" for c in cols))
        print()


if __name__ == "__main__":
    main()
