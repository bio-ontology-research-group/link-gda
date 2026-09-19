"""Overlap-stratified ranking, aggregated the paper's way (per fold, then across folds).

Extends leakage_overlap.py to match the main tables' aggregation: for each method
and each overlap stratum, compute the pooled mean rank within each fold, then report
the mean and standard deviation across the ten folds (the same two-level scheme the
paper uses for Tables 2 and 3). AUC uses the analytic identity (N - MR)/(N - 1).

Overlap for a test instance (disease d', gene g) is the fraction of d''s phenotypes
already linked to g by a training causes_phenotype edge. The definition lives in
leakage_overlap.py and is imported, not restated, so the two scripts cannot drift.
Its default counts only the 90% training split kge_transd.py fits on;
--overlap-source full-train restores the every-pair-in-train.csv variant the
published rows were computed with. Zero overlap = no memorised phenotype match for
the true gene, the genuine generalisation regime.

Ranks use best-rank tie handling (1 + number of candidates scoring strictly higher),
which is deterministic. The main tables break ties with an unseeded permutation, so
the "all" column here reproduces them up to that tie convention (a sub-position
difference).

Run from the repository root where data/ lives (ibex):
    python leakage_overlap_perfold.py
"""
from collections import defaultdict

import click as ck
import numpy as np
import pandas as pd

from leakage_overlap import OVERLAP_SOURCES, disease_phenotypes, instance_overlap

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


def _calibrate(scores):
    """Leave-one-out per-gene z-scoring, the calibrated setting used throughout."""
    n = scores.shape[0]
    if n < 3:
        return scores
    mean = (scores.sum(axis=0, keepdims=True) - scores) / (n - 1)
    var = ((scores ** 2).sum(axis=0, keepdims=True) - scores ** 2) / (n - 1) - mean ** 2
    return (scores - mean) / (np.sqrt(np.clip(var, 0, None)) + 1e-12)


def per_instance_ranks(arch, config, calibrate=False, results=None):
    """{(fold, disease, gene): average rank of the true gene}, or None if a file is missing.

    calibrate applies the leave-one-out per-gene z-scoring within each fold before
    ranking, which is what the calibrated setting means everywhere else in this project.
    Left off, this function is byte-identical in behaviour to the original.
    """
    results = results or RESULTS
    ranks = {}
    for fold in range(N_FOLDS):
        try:
            fh = open(f"{results}/" + FILENAME.format(arch=arch, fold=fold, config=config))
        except FileNotFoundError:
            return None
        keys, rows = [], []
        with fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                keys.append((fold, p[1], p[0], int(p[2])))
                rows.append(np.asarray(p[3:], dtype=float))
        matrix = np.vstack(rows)
        if calibrate:
            matrix = _calibrate(matrix)
        for i, (f, disease, gene, pos) in enumerate(keys):
            sc = matrix[i]
            greater = int(np.count_nonzero(sc > sc[pos]))
            equal = int(np.count_nonzero(sc == sc[pos]))
            ranks[(f, disease, gene)] = greater + (equal + 1) / 2.0
    return ranks


def _causes_phenotype_from_dump(path):
    """Gene -> phenotype set, read from the causes_phenotype edges actually in a graph dump."""
    cp = defaultdict(set)
    with open(path) as handle:
        for line in handle:
            src, rel, dst = line.rstrip("\n").split("\t")
            if rel == "causes_phenotype":
                cp[src].add(dst)
    return cp


def compute_overlap(dump_template=None, overlap_source="train-split"):
    """Fraction of a test disease's phenotypes already linked to the true gene in training.

    Without dump_template the definition is leakage_overlap.instance_overlap, which is
    the one the rest of the project uses. dump_template instead reads the edges from an
    exported graph; that path exists to verify the reconstruction, since a dump is the
    graph the model was handed rather than a reconstruction of it, and the two must agree.
    """
    if not dump_template:
        return instance_overlap(overlap_source, N_FOLDS)
    d2p = disease_phenotypes()
    overlap = {}
    for fold in range(N_FOLDS):
        cp = _causes_phenotype_from_dump(dump_template.format(fold=fold))
        te = pd.read_csv(f"data/folds/fold_{fold}/test.csv", sep="\t")
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


@ck.command()
@ck.option("--extra", multiple=True, help="Additional method as name=arch:config, repeatable")
@ck.option("--calibrate", is_flag=True, help="Rank calibrated scores instead of raw ones")
@ck.option("--only-extra", is_flag=True, help="Report only the --extra methods")
@ck.option("--dump-template", default=None, help="Graph dump path with {fold}; overlap is then read from the exported causes_phenotype edges, as a check on the reconstruction")
@ck.option("--overlap-source", type=ck.Choice(OVERLAP_SOURCES), default="train-split", show_default=True, help="full-train counts every pair in train.csv, which is what the published rows used")
@ck.option("--results", default=RESULTS, show_default=True)
@ck.option("--out", default=None, help="Write the strata table to this TSV as well")
def main(extra, calibrate, only_extra, dump_template, overlap_source, results, out):
    methods = {} if only_extra else dict(METHODS)
    for item in extra:
        name, spec = item.split("=", 1)
        arch, config = spec.split(":", 1)
        methods[name] = (arch, config)

    overlap = compute_overlap(dump_template, overlap_source)
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
    lines = ["\t".join(["method", "setting", "stratum", "folds"]
                       + [f"{c}_{s}" for c in cols for s in ("mean", "sd")])]
    setting = "calibrated" if calibrate else "raw"
    for name, (arch, cfg) in methods.items():
        r = per_instance_ranks(arch, cfg, calibrate=calibrate, results=results)
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
            lines.append("\t".join([name, setting, label, str(nf)]
                                   + [f"{v:.6f}" for c in cols for v in m[c]]))
        print()

    if out:
        with open(out, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
