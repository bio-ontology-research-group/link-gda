"""Per-fold and aggregate metrics for the ULTRA zero-shot baseline.

Metrics are computed by the project's own functions rather than reimplemented:
compute_metrics_from_rows for the ranking metrics, including its deterministic
average-rank tie handling, and _calibrated_rows for the leave-one-out per-gene
z-scoring that defines the calibrated setting. A baseline that used a different
rank convention or a different calibration would not be comparable with the
methods it is meant to sit beside in the results table.

Zero-shot ULTRA has no checkpoint selection, so unlike the trained methods the
two settings here share one set of scores: the calibrated column is the same
score file with the per-gene baseline removed. There is no _calsel arm to report.

Output is one TSV row per (modality, projection, aggregation, setting, fold), plus mean
and sample standard deviation over folds, written next to the score files. The modality
token is kge_transd.py's source_str, so a row lines up with the LinkGDA variant it is a
baseline for: pheno_func_expr is -pfs, func is -f, expr is -s, func_expr is -fs.
"""
import os
import glob
import json
import statistics

import click as ck

from evaluate_sem_sim import compute_metrics_from_rows
from evaluation import _calibrated_rows


METRICS = ["mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@100", "auc"]


def read_rows(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[0], parts[1], parts[2], [float(x) for x in parts[3:]]))
    return rows


@ck.command()
@ck.option("--results_dir", default="data/results", help="Directory holding the score files")
@ck.option("--pattern", default="kge_results_ultra4g_zeroshot_fold_*_by_graph_*.tsv")
@ck.option("--out", default="data/results/ultra_zeroshot_metrics.tsv")
def main(results_dir, pattern, out):
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not paths:
        raise SystemExit(f"no score files matched {pattern} under {results_dir}")

    records = []
    for path in paths:
        stem = os.path.basename(path)[len("kge_results_"):-len(".tsv")]
        identifier, aggregation = stem.rsplit("_by_graph_", 1)
        fold = int(identifier.split("_fold_")[1].split("_")[0])
        head, projection = identifier.rsplit("_proj_", 1)
        projection = projection.split("_use_graph")[0]
        modality = head.split("_seed_")[1].split("_", 1)[1]

        rows = read_rows(path)
        for setting, prepared in (("raw", rows), ("calibrated", _calibrated_rows(rows))):
            _, macro = compute_metrics_from_rows(prepared)
            records.append({
                "modality": modality, "projection": projection, "aggregation": aggregation,
                "setting": setting, "fold": fold,
                **{m: macro[m] for m in METRICS},
            })
        print(f"{stem}: done")

    header = ["modality", "projection", "aggregation", "setting", "fold"] + METRICS
    with open(out, "w") as handle:
        handle.write("\t".join(header) + "\n")
        for record in sorted(records, key=lambda r: (r["modality"], r["projection"], r["aggregation"], r["setting"], r["fold"])):
            handle.write("\t".join(str(record[c]) for c in header) + "\n")

        groups = {}
        for record in records:
            groups.setdefault((record["modality"], record["projection"], record["aggregation"], record["setting"]), []).append(record)
        for key, group in sorted(groups.items()):
            for label, fn in (("mean", statistics.fmean),
                              ("sd", lambda v: statistics.stdev(v) if len(v) > 1 else 0.0)):
                handle.write("\t".join(list(key) + [label] +
                                       [str(fn([r[m] for r in group])) for m in METRICS]) + "\n")

    print(f"wrote {out} ({len(records)} rows)")


if __name__ == "__main__":
    main()
