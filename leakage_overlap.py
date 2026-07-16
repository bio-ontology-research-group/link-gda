"""Quantify phenotype overlap between a test instance and the true gene's training edges.

The folds are disease-disjoint, but genes are shared with training. So for a test
instance (disease d', gene g), some of d''s phenotypes may ALREADY be linked to g
in the training graph, because g is associated with other training diseases that
share those phenotypes. Those materialized causes_phenotype(g, p) edges are exactly
what the model scores at inference, so a referee will ask how much of the ranking
they explain.

check_data_leakage.py does not measure this: it tests EXACT phenotype-profile
equality between a test and a train disease, so two diseases sharing 90% of their
phenotypes are not flagged. (It also writes test_no_leakage.csv using a
profile+gene criterion that never fires, so that file is identical to test.csv.)

For each test instance we compute
    overlap = |{p in P_d' : (g, p) in training causes_phenotype}| / |P_d'|
and then report each method's ranking separately on the zero-overlap instances and
the overlapping ones. If the conclusions hold on the zero-overlap subset, the
concern is answered.

Run from the repository root on ibex (where data/ lives):
    python leakage_overlap.py
"""
from collections import defaultdict

import numpy as np
import pandas as pd

N_FOLDS = 10
RESULTS = "data/results"
FILENAME = "kge_results_{arch}_fold_{fold}_seed_0_{config}.tsv"

METHODS = {
    "LinkGDA-p":   ("transd", "dim_400_bs_32768_lr_0.001_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "LinkGDA-pf":  ("transd", "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "LinkGDA-pfs": ("transd", "dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    # INDIGENA under OWL2Vec*, matching Table 1 (see p_value_per_fold.py)
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

    overlap = {}
    for fold in range(N_FOLDS):
        tr = pd.read_csv(f"data/folds/fold_{fold}/train.csv", sep="\t")
        te = pd.read_csv(f"data/folds/fold_{fold}/test.csv", sep="\t")
        # materialized causes_phenotype edges available in training, per gene
        cp = defaultdict(set)
        for g, d in zip(tr["Gene"], tr["Disease"]):
            cp[g] |= d2p.get(d, set())
        for g, d in zip(te["Gene"], te["Disease"]):
            P = d2p.get(d, set())
            if not P:
                continue
            overlap[(fold, d, g)] = len(P & cp.get(g, set())) / len(P)

    ov = np.array(list(overlap.values()))
    n = len(ov)
    print(f"test instances with a phenotype profile: {n}")
    print(f"  zero overlap  : {int(np.sum(ov == 0)):5d}  ({100*np.mean(ov == 0):.1f}%)")
    print(f"  some overlap  : {int(np.sum(ov > 0)):5d}  ({100*np.mean(ov > 0):.1f}%)")
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0000001)):
        m = (ov > lo) & (ov <= hi) if lo > 0 else (ov > 0) & (ov <= hi)
        print(f"    overlap in ({lo:.2f},{hi:.2f}]: {int(m.sum()):5d}")
    print(f"  mean overlap over instances with any: {ov[ov > 0].mean():.3f}")
    print()

    N_POOL = 4399
    print(f"{'method':<12} {'subset':<14} {'n':>6} {'MR':>9} {'AUC':>6}")
    for name, (arch, cfg) in METHODS.items():
        r = per_instance_ranks(arch, cfg)
        if r is None:
            print(f"{name:<12} SKIPPED (missing result files)")
            continue
        keys = [k for k in r if k in overlap]
        rr = np.array([r[k] for k in keys], dtype=float)
        oo = np.array([overlap[k] for k in keys])
        for label, mask in (("zero overlap", oo == 0), ("some overlap", oo > 0), ("all", np.ones(len(oo), bool))):
            sub = rr[mask]
            mr = sub.mean()
            print(f"{name:<12} {label:<14} {len(sub):6d} {mr:9.2f} {(N_POOL-mr)/(N_POOL-1):6.3f}")
        print()


if __name__ == "__main__":
    main()
