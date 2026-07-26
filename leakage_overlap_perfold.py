"""Overlap-stratified ranking, aggregated the paper's way (per fold, then across folds).

Extends leakage_overlap.py to match the main tables' aggregation: for each method
and each overlap stratum, compute the pooled mean rank within each fold, then report
the mean and standard deviation across the ten folds (the same two-level scheme the
paper uses for Tables 2 and 3). AUC uses the analytic identity (N - MR)/(N - 1).

Overlap for a test instance (disease d', gene g) is the fraction of d''s phenotypes
already linked to g by a training causes_phenotype edge (from leakage_overlap.py).
Zero overlap = no memorized phenotype match for the true gene, the genuine
generalisation regime.

Ranks use best-rank tie handling (1 + number of candidates scoring strictly higher),
which is deterministic. The main tables break ties with an unseeded permutation, so
the "all" column here reproduces them up to that tie convention (a sub-position
difference).

Run from the repository root where data/ lives (ibex):
    python leakage_overlap_perfold.py
"""
from collections import defaultdict

import numpy as np
import pandas as pd

N_FOLDS = 10
N_POOL = 4399
RESULTS = "data/results"
FILENAME = "kge_results_{arch}_fold_{fold}_seed_0_{config}.tsv"

METHODS = {
    "LinkGDA-p":   ("transd", "dim_400_bs_32768_lr_0.001_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "LinkGDA-pf":  ("transd", "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "LinkGDA-pfs": ("transd", "dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "INDIGENA":    ("transd", "dim_200_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_inductive_bma"),
}


def per_instance_ranks(arch, config):
    """{(fold, disease, gene): best-rank of the true gene}, or None if a file is missing."""
    ranks = {}
    for fold in range(N_FOLDS):
        try:
            fh = open(f"{RESULTS}/" + FILENAME.format(arch=arch, fold=fold, config=config))
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


HITS_K = (1, 3, 10, 100)


def fold_metrics(ranks, overlap, mask_fn):
    """Per-fold pooled metrics over instances passing mask_fn(overlap); mean+/-std over folds."""
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
    return {key: (np.mean([r[key] for r in rows]), np.std([r[key] for r in rows], ddof=1))
            for key in keys}, len(rows)


def main():
    overlap = compute_overlap()
    ov = np.array(list(overlap.values()))
    print(f"instances with a phenotype profile: {len(ov)}")
    print(f"  zero overlap: {int(np.sum(ov == 0))} ({100*np.mean(ov == 0):.1f}%)  "
          f"some overlap: {int(np.sum(ov > 0))} ({100*np.mean(ov > 0):.1f}%)\n")

    strata = [
        ("zero overlap", lambda o: o == 0),
        ("some overlap", lambda o: o > 0),
        ("all",          lambda o: True),
    ]
    cols = ["MR", "MRR", "H@1", "H@3", "H@10", "H@100", "AUC"]
    for name, (arch, cfg) in METHODS.items():
        r = per_instance_ranks(arch, cfg)
        if r is None:
            print(f"{name}: SKIPPED (missing result files)")
            continue
        print(f"=== {name} ===")
        print(f"{'subset':<13} " + " ".join(f"{c:>13}" for c in cols))
        for label, fn in strata:
            m, nf = fold_metrics(r, overlap, fn)
            def cell(c):
                mean, sd = m[c]
                return f"{mean:7.2f}+/-{sd:5.2f}" if c == "MR" else f"{mean:7.3f}+/-{sd:5.3f}"
            print(f"{label:<13} " + " ".join(f"{cell(c):>13}" for c in cols))
        print()


if __name__ == "__main__":
    main()
