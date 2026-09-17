"""Metrics on the excluded-gene benchmark from the per-seed prediction files.

Reads the raw score files for LinkGDA-f and LinkGDA-fs under both projections and both
arms, ten seeds each, and reports mean and sample standard deviation over seeds in the
uncalibrated and calibrated settings. Loading, leave-one-out per-gene calibration and
average-rank metrics are shared with rq1_table.py so the conventions match the main
benchmark tables. Writes a full-precision TSV beside the printed table.
"""
import os

import click as ck

from rq1_table import calibrate, load, metrics, summarise

KEYS = ["mr", "mrr", "h1", "h3", "h10", "h100", "auc"]
VARIANTS = [("LinkGDA-f", "func"), ("LinkGDA-fs", "func_expr")]
PROJECTIONS = [("owl2vecstar", "OWL2Vec*"), ("owl2vecstar_gda", "GDAProjector")]
CELLS = {
    ("owl2vecstar", "uncalibrated"): ("dim_100_bs_16384_lr_0.001", ""),
    ("owl2vecstar_gda", "uncalibrated"): ("dim_800_bs_131072_lr_0.01", ""),
    ("owl2vecstar", "calibrated"): ("dim_200_bs_65536_lr_0.001", "_calsel"),
    ("owl2vecstar_gda", "calibrated"): ("dim_200_bs_65536_lr_0.001", "_calsel"),
}


def template(results, source, projection, cell):
    hps, tag = cell
    return os.path.join(results, "kge_results_transd_fold_0_seed_{s}_" + hps + "_" + source
                        + "_proj_" + projection + "_use_graph_True_tol_15" + tag + "_by_graph_bma.tsv")


@ck.command()
@ck.option("--results", default="data/results_excluded", show_default=True)
@ck.option("--seeds", default=10, show_default=True)
@ck.option("--out", default=None, help="TSV path; defaults to <results>/excluded_table.tsv")
def main(results, seeds, out):
    out = out or os.path.join(results, "excluded_table.tsv")
    lines = ["\t".join(["method", "projection", "setting", "seeds"] + [f"{k}_{s}" for k in KEYS for s in ("mean", "sd")])]
    print(f"{'method':11} {'projection':13} {'setting':13} {'n':>2} {'MR':>18} {'MRR':>11} {'H@1':>11} {'H@3':>11} {'H@10':>11} {'H@100':>11} {'AUC':>11}")
    for label, source in VARIANTS:
        for projection, pname in PROJECTIONS:
            for setting in ("uncalibrated", "calibrated"):
                path = template(results, source, projection, CELLS[(projection, setting)])
                rows = []
                for seed in range(seeds):
                    p = path.format(s=seed)
                    if not os.path.exists(p):
                        print(f"missing {p}")
                        continue
                    scores, idx = load(p)
                    rows.append(metrics(calibrate(scores) if setting == "calibrated" else scores, idx))
                stats = {k: summarise(rows, k) for k in KEYS}
                lines.append("\t".join([label, pname, setting, str(len(rows))] + [f"{v:.6f}" for k in KEYS for v in stats[k]]))
                cells = [f"{stats['mr'][0]:8.2f} ± {stats['mr'][1]:6.2f}"] + [f"{stats[k][0]:.2f} ± {stats[k][1]:.2f}" for k in KEYS[1:]]
                print(f"{label:11} {pname:13} {setting:13} {len(rows):>2} {cells[0]:>18} " + " ".join(f"{c:>11}" for c in cells[1:]))
    with open(out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
