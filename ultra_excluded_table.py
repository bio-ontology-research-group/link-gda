"""Metrics for the zero-shot ULTRA baseline on the excluded-gene benchmark.

The excluded benchmark holds out gene-disease pairs whose gene has no MGI phenotype
annotation but does have GO functions, with the query disease removed from training by
both identifier and phenotype-profile overlap. Its 350 test genes therefore carry no
supervised association at all, while their function and expression annotations stay in
the graph. Metrics are reported on two candidate pools, as in excluded_table.py: the
full pool of 4,749 genes, and those 350 genes alone.

Loading, leave-one-out per-gene calibration and average-rank metrics are imported from
rq1_table so the conventions match the published excluded table rather than merely
resembling it. That table's AUC is the closed form (n - MR) / (n - 1), which is why this
script does not reuse ultra_metrics.py: the two are not interchangeable.

The pool definition is also the benchmark's leakage condition, and is asserted rather
than assumed: every test gene must be absent from the training pairs, so the unseen pool
must contain every true gene. A pass here means the 350-gene pool really is a pool of
genes the graph carries no association for.

Zero-shot inference is deterministic, so each configuration is one pass and the columns
carry no standard deviation.
"""
import csv
import glob
import os

import click as ck
import numpy as np

from rq1_table import calibrate, load, metrics


KEYS = ["mr", "mrr", "h1", "h3", "h10", "h100", "auc"]


def unseen_mask(benchmark):
    """Boolean over the candidate pool: genes with no training association.

    The pool order is the sorted unique Gene column of gene_diseases.csv, which is what
    kge_transd.py uses for eval_genes and therefore the column order of every score file.
    """
    pool = sorted({r["Gene"] for r in csv.DictReader(open(os.path.join(benchmark, "gene_diseases.csv")))})
    train = {r["Gene"] for r in csv.DictReader(
        open(os.path.join(benchmark, "folds", "fold_0", "train.csv")), delimiter="\t")}
    return pool, np.array([g not in train for g in pool])


def restrict(scores, idx, mask):
    position = np.cumsum(mask) - 1
    if not mask[idx].all():
        raise SystemExit("a test pair's true gene has a training association; "
                         "the excluded benchmark's holdout condition is violated")
    return scores[:, mask], position[idx]


@ck.command()
@ck.option("--results", default="data/results", show_default=True)
@ck.option("--benchmark", required=True, help="Excluded-benchmark data directory")
@ck.option("--pattern", default="kge_results_ultra4g_zeroshot_excluded_fold_0_seed_0_*_by_graph_*.tsv")
@ck.option("--out", default=None)
def main(results, benchmark, pattern, out):
    out = out or os.path.join(results, "ultra_excluded_table.tsv")
    pool, mask = unseen_mask(benchmark)
    print(f"candidate pool {len(pool)}, of which {int(mask.sum())} carry no training association")

    paths = sorted(glob.glob(os.path.join(results, pattern)))
    if not paths:
        raise SystemExit(f"no score files matched {pattern} under {results}")

    header = ["modality", "projection", "aggregation", "setting", "pool", "pool_size", "pairs"] + KEYS
    lines = ["\t".join(header)]
    for path in paths:
        stem = os.path.basename(path)[len("kge_results_"):-len(".tsv")]
        identifier, aggregation = stem.rsplit("_by_graph_", 1)
        head, projection = identifier.rsplit("_proj_", 1)
        projection = projection.split("_use_graph")[0]
        modality = head.split("_seed_")[1].split("_", 1)[1]

        scores, idx = load(path)
        for setting, prepared in (("raw", scores), ("calibrated", calibrate(scores))):
            for label, m in (("full", None), ("unseen", mask)):
                s, i = (prepared, idx) if m is None else restrict(prepared, idx, m)
                stats = metrics(s, i)
                lines.append("\t".join(
                    [modality, projection, aggregation, setting, label, str(stats["n"]), str(s.shape[0])]
                    + [f"{stats[k]:.6f}" for k in KEYS]))
        print(f"{stem}: done")

    with open(out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines) - 1} rows)")
    print(f"random-ranking MR: full pool {(len(pool) + 1) / 2:.1f}, "
          f"unseen pool {(int(mask.sum()) + 1) / 2:.1f}")


if __name__ == "__main__":
    main()
