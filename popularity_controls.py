"""Query-independent annotation-count controls for the GDA benchmark.

Purpose: a referee will ask whether LinkGDA's ranking is driven by a popularity
prior rather than by phenotype/function reasoning. Genes that appear in many
training gene-disease associations receive more materialized causes_phenotype
edges, and genes with many annotations of any kind receive better-trained
embeddings and (for best-match aggregation) more chances to match an arbitrary
query. These controls quantify that floor.

Each control ranks the SAME candidate-gene pool by a score that ignores the
query disease entirely, so any method that reasons about the query must beat
them. Four controls:

  1. Gene degree      -- number of distinct TRAINING diseases a gene is
                         associated with (fold-dependent; never uses test
                         associations). Probes the causes_phenotype prior.
  2. Phenotype count  -- number of distinct MP phenotype annotations.
  3. Function count   -- number of distinct GO function annotations. Probes
                         ascertainment bias: GO annotation count is a proxy for
                         how well studied a gene is, and well studied genes are
                         likelier to be known disease genes.
  4. Expression-site count -- number of distinct UBERON sites.

Controls 2-4 are static (annotations do not depend on the fold), so their
ranking is fixed across folds; only the evaluated test genes vary.

Tie handling: scores are integer counts, so ties are pervasive (most genes have
degree 1). We use the AVERAGE rank within a tie group, which is the unbiased
convention. The pipeline's own convention (1 + #strictly-greater) would give the
best-case rank inside a tie group and would flatter these controls; it is
reported by --optimistic for transparency.

Run from the repository root, where data/ lives (i.e. on ibex):
    python popularity_controls.py
"""
import argparse

import numpy as np
import pandas as pd

N_FOLDS = 10


def load_pool():
    gd = pd.read_csv("data/gene_diseases.csv")
    genes = sorted(set(gd["Gene"]))
    return genes, {g: i for i, g in enumerate(genes)}


def static_scores(genes, path, col):
    df = pd.read_csv(path)
    cnt = df.groupby("Gene")[col].nunique()
    s = np.array([cnt.get(g, 0) for g in genes], dtype=float)
    return [s] * N_FOLDS


def degree_scores(genes, trains):
    out = []
    for tr in trains:
        pop = tr.groupby("Gene")["Disease"].nunique()
        out.append(np.array([pop.get(g, 0) for g in genes], dtype=float))
    return out


def rank_of(score_vec, idx, optimistic):
    s = score_vec[idx]
    n_greater = int(np.count_nonzero(score_vec > s))
    if optimistic:
        return n_greater + 1
    n_equal = int(np.count_nonzero(score_vec == s))
    return n_greater + (n_equal + 1) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimistic", action="store_true",
                    help="use the pipeline's best-case-within-ties convention")
    args = ap.parse_args()

    genes, gi = load_pool()
    n = len(genes)
    tests = [pd.read_csv(f"data/folds/fold_{f}/test.csv", sep="\t") for f in range(N_FOLDS)]
    trains = [pd.read_csv(f"data/folds/fold_{f}/train.csv", sep="\t") for f in range(N_FOLDS)]

    controls = [
        ("Gene degree (training GDAs)", degree_scores(genes, trains)),
        ("Phenotype count", static_scores(genes, "data/gene_phenotypes.csv", "Phenotype")),
        ("Function count", static_scores(genes, "data/gene_functions.csv", "Function")),
        ("Expression-site count", static_scores(genes, "data/gene_site.csv", "Tissue")),
    ]

    tie = "optimistic (best-case within ties)" if args.optimistic else "average rank within ties"
    print(f"Candidate pool N={n}; tie handling: {tie}")
    print(f"Random baseline: MR={(n + 1) / 2:.2f}, AUC=0.500\n")
    print("LaTeX rows (mean +/- std over the 10 folds):")

    for tag, score_vecs in controls:
        acc = {k: [] for k in ("mr", "mrr", "h1", "h3", "h10", "h100", "auc")}
        for te, sc in zip(tests, score_vecs):
            rr = [rank_of(sc, gi[g], args.optimistic) for g in te["Gene"] if g in gi]
            r = np.array(rr, dtype=float)
            mr = r.mean()
            acc["mr"].append(mr)
            acc["mrr"].append(float(np.mean(1.0 / r)))
            acc["h1"].append(float(np.mean(r <= 1)))
            acc["h3"].append(float(np.mean(r <= 3)))
            acc["h10"].append(float(np.mean(r <= 10)))
            acc["h100"].append(float(np.mean(r <= 100)))
            acc["auc"].append((n - mr) / (n - 1))

        cells = []
        for key, dec in (("mr", 2), ("mrr", 2), ("h1", 2), ("h3", 2),
                         ("h10", 2), ("h100", 2), ("auc", 2)):
            m = np.mean(acc[key])
            s = np.std(acc[key])
            cells.append("%.*f\\std{%.*f}" % (dec, m, dec, s))
        print("  %s & %s \\\\" % (tag, " & ".join(cells)))


if __name__ == "__main__":
    main()
