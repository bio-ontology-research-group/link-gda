"""Median rank and rank-CDF per method, pooled over all instances (paper's saved ranks).

Reads the same per-instance average ranks the overlap scripts use, pools across the
ten folds, and reports (a) median rank overall and by phenotype-overlap stratum and
(b) the empirical CDF P(rank <= x) on a fixed grid, for a rank-CDF figure.

Two callers, since the result files live on different hosts:
  ibex        : kge methods  (LinkGDA-p/pf/pfs, INDIGENA)   -> pass mode="kge"
  workstation : sem-sim      (SimGIC, Resnik/Lin x BMA/BMM) -> pass mode="semsim"
"""
import sys
from collections import defaultdict
import numpy as np
import pandas as pd

N_FOLDS = 10
GRID = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4399]

KGE = {
    "LinkGDA-p":   ("transd", "dim_400_bs_32768_lr_0.001_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "LinkGDA-pf":  ("transd", "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "LinkGDA-pfs": ("transd", "dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "INDIGENA":    ("transd", "dim_200_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_inductive_bma"),
}
SEMSIM = {"SimGIC":"resnik_simgic","Resnik-BMA":"resnik_resnik_bma","Resnik-BMM":"resnik_resnik_bmm",
          "Lin-BMA":"resnik_lin_bma","Lin-BMM":"resnik_lin_bmm"}


def kge_ranks(arch, config):
    ranks = {}
    for fold in range(N_FOLDS):
        try:
            fh = open(f"data/results/kge_results_{arch}_fold_{fold}_seed_0_{config}.tsv")
        except FileNotFoundError:
            return None
        with fh:
            for line in fh:
                p = line.rstrip("\n").split("\t"); pos = int(p[2])
                sc = np.asarray(p[3:], dtype=float)
                g = int(np.count_nonzero(sc > sc[pos])); e = int(np.count_nonzero(sc == sc[pos]))
                ranks[(fold, p[1], p[0])] = g + (e + 1) / 2.0
    return ranks


def semsim_ranks(stem):
    ranks = {}
    for fold in range(N_FOLDS):
        try:
            fh = open(f"data/baseline_results/{stem}_fold{fold}_results.txt")
        except FileNotFoundError:
            return None
        with fh:
            for line in fh:
                p = line.rstrip("\n").split("\t"); pos = int(p[2])
                sc = np.asarray(p[3:], dtype=float)
                g = int(np.count_nonzero(sc > sc[pos])); e = int(np.count_nonzero(sc == sc[pos]))
                ranks[(fold, p[1], p[0])] = g + (e + 1) / 2.0
    return ranks


def overlap_map():
    dp = pd.read_csv("data/disease_phenotypes.csv")
    d2p = dp.groupby("Disease")["Phenotype"].apply(set).to_dict()
    ov = {}
    for fold in range(N_FOLDS):
        tr = pd.read_csv(f"data/folds/fold_{fold}/train.csv", sep="\t")
        te = pd.read_csv(f"data/folds/fold_{fold}/test.csv", sep="\t")
        cp = defaultdict(set)
        for g, d in zip(tr["Gene"], tr["Disease"]): cp[g] |= d2p.get(d, set())
        for g, d in zip(te["Gene"], te["Disease"]):
            P = d2p.get(d, set())
            if P: ov[(fold, d, g)] = len(P & cp.get(g, set())) / len(P)
    return ov


def report(name, ranks, ov):
    keys = [k for k in ranks if k in ov]
    allv = np.array([ranks[k] for k in keys])
    zero = np.array([ranks[k] for k in keys if ov[k] == 0])
    some = np.array([ranks[k] for k in keys if ov[k] > 0])
    print(f"=== {name} (n={len(allv)}) ===")
    print(f"median  all={np.median(allv):.1f}  zero={np.median(zero):.1f}  some={np.median(some):.1f}")
    for lab, v in (("all", allv), ("zero", zero), ("some", some)):
        cdf = " ".join(f"({x},{100*np.mean(v <= x):.1f})" for x in GRID)
        print(f"CDF-{lab:<4} {cdf}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "kge"
    ov = overlap_map()
    src = KGE if mode == "kge" else SEMSIM
    for name, spec in src.items():
        r = kge_ranks(*spec) if mode == "kge" else semsim_ranks(spec)
        if r is None:
            print(f"{name}: SKIPPED"); continue
        report(name, r, ov)


if __name__ == "__main__":
    main()
