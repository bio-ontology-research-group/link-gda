"""Ask whether the seed-to-seed spread in mean rank comes from the models or the metric.

The fixed-split arms hold the data constant: same training diseases, same validation
diseases, same test pairs for every seed. Any remaining spread therefore comes from
initialization and negative sampling. This script asks a further question: does that
spread reflect models that differ broadly, or models that differ on a few test
instances whose ranks the mean amplifies?

Three views:

  1. Mean versus median rank across seeds. The mean is tail-sensitive and the median
     is not, so a large mean spread beside a small median spread points at the tail
     rather than at broad disagreement.
  2. Per-instance rank spread. For each test instance, the standard deviation of its
     rank across seeds. If a small subset carries most of the variance, the metric is
     amplifying a local disagreement.
  3. Leave-the-worst-out. Recompute the across-seed spread in mean rank after dropping
     the k test instances with the largest per-instance spread. A steep drop means a
     handful of instances drive the reported variance.

Usage:
    python variance_decomposition.py --results tol15/data/results --seeds 10
"""
import os

import click as ck
import numpy as np

CONFIG = "dim_100_bs_16384_lr_0.001"
VARIANTS = {"f": "func", "fs": "func_expr"}


def ranks_by_instance(path):
    """Rank per test instance, keyed by (disease, gene) so seeds align row-wise."""
    out = {}
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            gene, disease, idx = parts[0], parts[1], int(parts[2])
            scores = np.asarray(parts[3:], dtype=float)
            out[(disease, gene)] = 1 + int(np.count_nonzero(scores > scores[idx]))
    return out


@ck.command()
@ck.option("--results", default="tol15/data/results")
@ck.option("--seeds", default=10)
@ck.option("--aggregation", default="bma")
def main(results, seeds, aggregation):
    for variant, source in VARIANTS.items():
        per_seed = []
        for seed in range(seeds):
            path = (f"{results}/kge_results_transd_fold_0_seed_{seed}_{CONFIG}_"
                    f"{source}_proj_owl2vecstar_gda_use_graph_True_by_graph_"
                    f"{aggregation}.tsv")
            if os.path.exists(path):
                per_seed.append(ranks_by_instance(path))

        print(f"{'=' * 70}\nLinkGDA-{variant}  ({len(per_seed)} seeds)\n{'=' * 70}")
        if len(per_seed) < 3:
            print("  too few seeds\n")
            continue

        keys = sorted(set.intersection(*[set(d) for d in per_seed]))
        matrix = np.array([[d[k] for k in keys] for d in per_seed], dtype=float)  # seeds x instances

        means = matrix.mean(axis=1)
        medians = np.median(matrix, axis=1)
        print(f"  instances compared                : {len(keys)}")
        print(f"  MEAN   rank across seeds          : {means.mean():8.2f} +/- {means.std(ddof=1):7.2f}"
              f"   (cv {means.std(ddof=1)/means.mean():.1%})")
        print(f"  MEDIAN rank across seeds          : {medians.mean():8.2f} +/- {medians.std(ddof=1):7.2f}"
              f"   (cv {medians.std(ddof=1)/medians.mean():.1%})")

        inst_sd = matrix.std(axis=0, ddof=1)
        order = np.argsort(inst_sd)[::-1]
        print(f"\n  per-instance rank sd across seeds : median {np.median(inst_sd):.1f}, "
              f"p90 {np.percentile(inst_sd, 90):.1f}, max {inst_sd.max():.1f}")
        total_var = float(np.sum(inst_sd ** 2))
        for frac in (0.01, 0.05, 0.10):
            k = max(1, int(len(keys) * frac))
            share = float(np.sum(inst_sd[order[:k]] ** 2)) / total_var
            print(f"    top {frac:>4.0%} most variable instances ({k:>3}) hold {share:5.1%} of instance variance")

        print("\n  leave-the-worst-out: spread in mean rank after dropping k noisiest instances")
        for k in (0, 5, 10, 20, 40):
            if k >= len(keys):
                continue
            keep = order[k:]
            m = matrix[:, keep].mean(axis=1)
            print(f"    drop {k:>3}  ({k/len(keys):>4.1%})  mean {m.mean():8.2f} +/- {m.std(ddof=1):7.2f}")
        print()


if __name__ == "__main__":
    main()
