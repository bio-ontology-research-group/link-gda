"""Separate the phenotype-overlap effect from the gene-degree effect.

leakage_overlap.py showed LinkGDA ranks BETTER on zero-overlap instances while
INDIGENA ranks far worse. Before attributing that to overlap, note that a gene
with training degree 0 has no causes_phenotype edges at all, so it is zero-overlap
BY CONSTRUCTION. The zero-overlap set is therefore contaminated with the degree-0
set, and "overlap" and "degree" are confounded.

This splits three ways:
    A. degree == 0                (no training diseases; overlap 0 by construction)
    B. degree >= 1, overlap == 0  (has training diseases, none share phenotypes)
    C. degree >= 1, overlap > 0   (has phenotypically related training diseases)

A vs B isolates the degree effect at constant (zero) overlap.
B vs C isolates the overlap effect at constant (non-zero) degree.

Also reports the disease phenotype-set size and the gene's MP annotation count per
group, since either could confound a similarity-based method.

Run from the repository root on ibex:
    python leakage_overlap_verify.py
"""
from collections import defaultdict

import numpy as np
import pandas as pd

N_FOLDS = 10
RESULTS = "data/results"
FILENAME = "kge_results_{arch}_fold_{fold}_seed_0_{config}.tsv"
N_POOL = 4399

METHODS = {
    "LinkGDA-pfs": ("transd", "dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "LinkGDA-pf":  ("transd", "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "INDIGENA":    ("transd", "dim_200_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_inductive_bma"),
}


def per_instance_ranks(arch, config):
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
                ranks[(fold, p[1], p[0])] = 1 + int(np.count_nonzero(sc > sc[pos]))
    return ranks


def main():
    dp = pd.read_csv("data/disease_phenotypes.csv")
    d2p = dp.groupby("Disease")["Phenotype"].apply(set).to_dict()
    gp = pd.read_csv("data/gene_phenotypes.csv")
    g2mp = gp.groupby("Gene")["Phenotype"].nunique().to_dict()

    group, dsize, gmp = {}, {}, {}
    for fold in range(N_FOLDS):
        tr = pd.read_csv(f"data/folds/fold_{fold}/train.csv", sep="\t")
        te = pd.read_csv(f"data/folds/fold_{fold}/test.csv", sep="\t")
        cp = defaultdict(set)
        deg = defaultdict(int)
        for g, d in zip(tr["Gene"], tr["Disease"]):
            cp[g] |= d2p.get(d, set())
        for g, d in tr.groupby(["Gene"])["Disease"].nunique().items():
            deg[g] = d
        for g, d in zip(te["Gene"], te["Disease"]):
            P = d2p.get(d, set())
            if not P:
                continue
            k = (fold, d, g)
            ov = len(P & cp.get(g, set())) / len(P)
            if deg.get(g, 0) == 0:
                group[k] = "A: degree==0"
            elif ov == 0:
                group[k] = "B: deg>=1, ov==0"
            else:
                group[k] = "C: deg>=1, ov>0"
            dsize[k] = len(P)
            gmp[k] = g2mp.get(g, 0)

    labels = ["A: degree==0", "B: deg>=1, ov==0", "C: deg>=1, ov>0"]
    print("=== group sizes and possible confounds ===")
    print(f"{'group':<18} {'n':>6} {'mean |P_d|':>11} {'mean gene MP':>13}")
    for L in labels:
        ks = [k for k in group if group[k] == L]
        print(f"{L:<18} {len(ks):6d} {np.mean([dsize[k] for k in ks]):11.1f} {np.mean([gmp[k] for k in ks]):13.1f}")
    print()

    print("=== ranking per group ===")
    print(f"{'method':<12} {'group':<18} {'n':>6} {'MR':>9} {'AUC':>6}")
    for name, (arch, cfg) in METHODS.items():
        r = per_instance_ranks(arch, cfg)
        if r is None:
            print(f"{name:<12} SKIPPED (missing files)")
            continue
        for L in labels:
            ks = [k for k in group if group[k] == L and k in r]
            sub = np.array([r[k] for k in ks], dtype=float)
            mr = sub.mean()
            print(f"{name:<12} {L:<18} {len(sub):6d} {mr:9.2f} {(N_POOL-mr)/(N_POOL-1):6.3f}")
        print()


if __name__ == "__main__":
    main()
