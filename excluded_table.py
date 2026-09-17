"""Metrics on the excluded-gene benchmark from the per-seed prediction files.

Reads the raw score files for LinkGDA-f and LinkGDA-fs under both projections and both
arms, ten seeds each, and reports mean and sample standard deviation over seeds in the
uncalibrated and calibrated settings, on two candidate pools: the full pool of 4,749
genes and the 350 benchmark genes alone, none of which appears in any training
association. Loading, leave-one-out per-gene calibration and average-rank metrics are
shared with rq1_table.py so the conventions match the main benchmark tables. Writes a
full-precision TSV and prints the rows of the main-text table in LaTeX.
"""
import csv
import os

import click as ck
import numpy as np

from rq1_table import calibrate, load, metrics, summarise

KEYS = ["mr", "mrr", "h1", "h3", "h10", "h100", "auc"]
VARIANTS = [("LinkGDA-f", "func"), ("LinkGDA-fs", "func_expr")]
PROJECTIONS = [("owl2vecstar", "OWL2Vec*", r"\projection{}"), ("owl2vecstar_gda", "GDAProjector", r"\projectionextended{}")]
CELLS = {
    ("owl2vecstar", "uncalibrated"): ("dim_100_bs_16384_lr_0.001", ""),
    ("owl2vecstar_gda", "uncalibrated"): ("dim_800_bs_131072_lr_0.01", ""),
    ("owl2vecstar", "calibrated"): ("dim_200_bs_65536_lr_0.001", "_calsel"),
    ("owl2vecstar_gda", "calibrated"): ("dim_200_bs_65536_lr_0.001", "_calsel"),
}
TABLE_KEYS = ["mr", "h10", "auc"]


def template(results, source, projection, cell):
    hps, tag = cell
    return os.path.join(results, "kge_results_transd_fold_0_seed_{s}_" + hps + "_" + source
                        + "_proj_" + projection + "_use_graph_True_tol_15" + tag + "_by_graph_bma.tsv")


def unseen_mask(benchmark):
    pool = sorted({r["Gene"] for r in csv.DictReader(open(os.path.join(benchmark, "gene_diseases.csv")))})
    train = {r["Gene"] for r in csv.DictReader(open(os.path.join(benchmark, "folds", "fold_0", "train.csv")), delimiter="\t")}
    return np.array([g not in train for g in pool])


def restrict(scores, idx, mask):
    position = np.cumsum(mask) - 1
    assert mask[idx].all()
    return scores[:, mask], position[idx]


def fmt(key, mean, sd):
    digits = 2
    return f"{mean:.{digits}f}\\std{{{sd:.{digits}f}}}"


@ck.command()
@ck.option("--results", default="data/results_excluded", show_default=True)
@ck.option("--benchmark", default="../link-gda-excluded/data", show_default=True, help="Excluded-benchmark data directory, for the candidate pool and training pairs")
@ck.option("--seeds", default=10, show_default=True)
@ck.option("--out", default=None, help="TSV path; defaults to <results>/excluded_table.tsv")
def main(results, benchmark, seeds, out):
    out = out or os.path.join(results, "excluded_table.tsv")
    mask = unseen_mask(benchmark)
    pools = [("unseen", mask), ("full", None)]
    header = ["method", "projection", "setting", "pool", "pool_size", "seeds"] + [f"{k}_{s}" for k in KEYS for s in ("mean", "sd")]
    lines = ["\t".join(header)]
    table = {}
    for label, source in VARIANTS:
        for projection, pname, pmacro in PROJECTIONS:
            for setting in ("uncalibrated", "calibrated"):
                path = template(results, source, projection, CELLS[(projection, setting)])
                rows = {p: [] for p, _ in pools}
                for seed in range(seeds):
                    p = path.format(s=seed)
                    if not os.path.exists(p):
                        print(f"missing {p}")
                        continue
                    scores, idx = load(p)
                    scored = calibrate(scores) if setting == "calibrated" else scores
                    for pool, m in pools:
                        s, i = (scored, idx) if m is None else restrict(scored, idx, m)
                        rows[pool].append(metrics(s, i))
                for pool, m in pools:
                    stats = {k: summarise(rows[pool], k) for k in KEYS}
                    size = rows[pool][0]["n"] if rows[pool] else 0
                    lines.append("\t".join([label, pname, setting, pool, str(size), str(len(rows[pool]))] + [f"{v:.6f}" for k in KEYS for v in stats[k]]))
                    table[(label, pmacro, setting, pool)] = stats
    with open(out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {out}\n")
    for label, _ in VARIANTS:
        print(f"    \\multirow{{4}}{{*}}{{\\method{{}}-{label.split('-')[1]}}}")
        for _, _, pmacro in PROJECTIONS:
            print(f"      & \\multirow{{2}}{{*}}{{{pmacro}}}")
            for setting in ("uncalibrated", "calibrated"):
                cells = []
                for pool in ("unseen", "full"):
                    st = table[(label, pmacro, setting, pool)]
                    cells += [fmt(k, *st[k]) for k in TABLE_KEYS]
                lead = "        & " if setting == "uncalibrated" else "        & & "
                print(f"{lead}{setting} & " + " & ".join(cells) + r" \\")
        print(r"    \midrule")


if __name__ == "__main__":
    main()
