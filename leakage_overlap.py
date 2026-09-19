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

Which training pairs count
--------------------------
The default is the pairs the model is actually fitted on, not every pair in
train.csv. kge_transd.py splits train.csv with create_train_val_split(df,
val_ratio=0.1, random_seed=0) and materialises causes_phenotype edges from the
training half only; the validation diseases' edges never enter the graph. Counting
them anyway marks roughly 760 instances as overlapping when the model held no such
edge, which dilutes the overlapping stratum and understates the contrast the
stratification exists to measure. That mislabelling applies to every method in the
table, including the symbolic baselines, so it is not a per-method correction.

A second, smaller correction goes with it. kge_transd.py drops a disease-phenotype
annotation whose phenotype is not a node of the projected ontology (about 16k of
164k), so those edges are not in the graph either. Reconstructing without that
filter leaves 9,469 phantom edges per fold over 76 phenotypes, enough to move 591
instances out of the zero-overlap stratum. train-split therefore intersects with the
phenotype vocabulary of the uPheno edge list.

--overlap-source full-train restores the earlier behaviour, unfiltered and over every
training pair, which is what the already-published rows were computed with.
"""
from collections import defaultdict

import click as ck
import numpy as np
import pandas as pd

from data import create_train_val_split

N_FOLDS = 10
OVERLAP_SOURCES = ("train-split", "full-train")
UPHENO_EDGES = "data/upheno_edges.tsv"
RESULTS = "data/results"
FILENAME = "kge_results_{arch}_fold_{fold}_seed_0_{config}.tsv"

METHODS = {
    "LinkGDA-p":   ("transd", "dim_400_bs_32768_lr_0.001_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "LinkGDA-pf":  ("transd", "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    "LinkGDA-pfs": ("transd", "dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
    # INDIGENA under OWL2Vec*, matching Table 1 (see p_value_per_fold.py)
    "INDIGENA":    ("transd", "dim_200_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_inductive_bma"),
}


def disease_phenotypes():
    """Disease -> phenotype set, the denominator of the overlap fraction."""
    dp = pd.read_csv("data/disease_phenotypes.csv")
    return dp.groupby("Disease")["Phenotype"].apply(set).to_dict()


def graph_phenotypes(edges_file=UPHENO_EDGES):
    """Phenotype terms the projected ontology contributes as graph nodes.

    A phenotype reaches the graph only through the uPheno projection, so the node set of
    that edge list is exactly the vocabulary kge_transd.py keeps when it builds
    disease2pheno. The two projections' edge lists agree on every term that matters here,
    which is why the labels come out the same under either.
    """
    nodes = set()
    with open(edges_file) as handle:
        for line in handle:
            src, _, dst = line.rstrip("\n").split("\t")
            nodes.add(src)
            nodes.add(dst)
    return nodes


def training_phenotype_edges(fold, d2p, source="train-split", vocabulary=None):
    """Gene -> phenotypes reachable through that gene's training causes_phenotype edges.

    train-split uses create_train_val_split with the defaults kge_transd.py runs with,
    so the edges are the ones the training graph actually carries. full-train uses every
    pair in train.csv, which is what the published numbers were computed from. vocabulary
    restricts the phenotypes to those the graph holds, matching the filter kge_transd.py
    applies when it materialises the edges.
    """
    if source not in OVERLAP_SOURCES:
        raise ValueError(f"overlap source must be one of {OVERLAP_SOURCES}")
    train = pd.read_csv(f"data/folds/fold_{fold}/train.csv", sep="\t")
    if source == "train-split":
        train, _ = create_train_val_split(train, val_ratio=0.1, random_seed=0)
    edges = defaultdict(set)
    for gene, disease in zip(train["Gene"], train["Disease"]):
        edges[gene] |= d2p.get(disease, set())
    if vocabulary is not None:
        for gene in edges:
            edges[gene] &= vocabulary
    return edges


def instance_overlap(source="train-split", folds=N_FOLDS, edges_file=UPHENO_EDGES):
    """{(fold, disease, gene): overlap fraction} for every test pair with a profile."""
    d2p = disease_phenotypes()
    vocabulary = graph_phenotypes(edges_file) if source == "train-split" else None
    overlap = {}
    for fold in range(folds):
        edges = training_phenotype_edges(fold, d2p, source, vocabulary)
        test = pd.read_csv(f"data/folds/fold_{fold}/test.csv", sep="\t")
        for gene, disease in zip(test["Gene"], test["Disease"]):
            phenos = d2p.get(disease, set())
            if not phenos:
                continue
            overlap[(fold, disease, gene)] = len(phenos & edges.get(gene, set())) / len(phenos)
    return overlap


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


@ck.command()
@ck.option("--overlap-source", type=ck.Choice(OVERLAP_SOURCES), default="train-split", show_default=True)
def main(overlap_source):
    overlap = instance_overlap(overlap_source)

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
