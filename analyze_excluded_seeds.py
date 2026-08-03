"""Aggregate the ten-seed excluded-gene benchmark runs.

Recomputes every metric from the per-instance rank files on disk rather than reading
any summary the training run printed, and reports mean and standard deviation across
seeds. The seed spread is the point: the main benchmark reports across-fold deviations
from a single seed, so it cannot say whether an effect exceeds initialization variance.

Rank convention matches p_value_per_fold.py: a row is
    gene <TAB> disease <TAB> true_index <TAB> score_0 ... score_{N-1}
and the rank is one plus the number of candidates scoring strictly higher than the
true gene, so ties take the best rank. That is the optimistic convention used
elsewhere in this work.

The result filename does not encode --val_seed or --tolerance, so runs that differ only
in those settings collide. Each arm therefore gets its own working directory and is read
by pointing --results at it. The reported numbers come from the tolerance-15 arm; the
tolerance-5 arm is kept for comparison.

Usage, from the excluded-benchmark working directory:
    python analyze_excluded_seeds.py --results tol15/data/results --seeds 10
    python analyze_excluded_seeds.py --results tol5/data/results  --seeds 10
"""
import glob
import os
import re

import click as ck
import numpy as np

CONFIG = "dim_100_bs_16384_lr_0.001"
VARIANTS = {"f": "func", "fs": "func_expr"}
HITS_AT = (1, 3, 10, 100)


def ranks_from_file(path):
    """Rank of the true gene for every test instance in one result file."""
    ranks = []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            true_index = int(parts[2])
            scores = np.asarray(parts[3:], dtype=float)
            ranks.append(1 + int(np.count_nonzero(scores > scores[true_index])))
    return np.asarray(ranks, dtype=float)


def metrics(ranks, pool_size):
    """Mean rank, MRR, hits at k, and the AUC implied by mean rank."""
    out = {
        "n": len(ranks),
        "MR": ranks.mean(),
        "MRR": float((1.0 / ranks).mean()),
    }
    for k in HITS_AT:
        out[f"H@{k}"] = float((ranks <= k).mean())
    out["AUC"] = (pool_size - ranks.mean()) / (pool_size - 1)
    return out


@ck.command()
@ck.option("--results", default="data/results", help="Directory holding the result TSVs")
@ck.option("--seeds", default=10, help="Number of seeds expected per variant")
@ck.option("--aggregation", default="bma", type=ck.Choice(["bma", "bmm"]))
@ck.option("--pool-file", default="data/gene_diseases.csv", help="Candidate pool, for AUC")
def main(results, seeds, aggregation, pool_file):
    import pandas as pd
    pool = pd.read_csv(pool_file)
    pool_size = len({str(g).split("/")[-1] for g in pool["Gene"]})
    print(f"candidate pool: {pool_size:,} genes   aggregation: {aggregation}\n")

    for variant, source in VARIANTS.items():
        rows, missing = [], []
        for seed in range(seeds):
            pattern = (f"{results}/kge_results_transd_fold_0_seed_{seed}_{CONFIG}_"
                       f"{source}_proj_owl2vecstar_gda_use_graph_True_by_graph_{aggregation}.tsv")
            hits = glob.glob(pattern)
            if not hits:
                missing.append(seed)
                continue
            rows.append(metrics(ranks_from_file(hits[0]), pool_size))

        label = f"LinkGDA-{variant}"
        if missing:
            print(f"{label}: MISSING seeds {missing}")
        if not rows:
            print(f"{label}: no result files found\n")
            continue

        n_instances = {r["n"] for r in rows}
        print(f"{label}  ({len(rows)} seeds, {n_instances.pop() if len(n_instances)==1 else n_instances} test instances)")
        for key in ["MR", "MRR"] + [f"H@{k}" for k in HITS_AT] + ["AUC"]:
            values = np.array([r[key] for r in rows])
            fmt = "8.2f" if key == "MR" else "8.4f"
            print(f"   {key:<6} {values.mean():{fmt}} +/- {values.std(ddof=1):{fmt}}"
                  f"   [min {values.min():{fmt}}, max {values.max():{fmt}}]")
        print()


if __name__ == "__main__":
    main()
