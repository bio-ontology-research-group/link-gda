"""W6 severity check: headline mean ranks on all 10 folds vs the clean folds 3-9.

Hyperparameters were selected by minimizing test mean rank on folds 0, 1, 2
(sweeps minimize test_imac_bma_mr, which is the test-set metric). Folds 3-9 were
not used in selection, so comparing all-10-fold means to folds-3-9 means shows
how much the selection inflated the reported numbers.

Reads saved per-instance ranks; no training. Seed 0, BMA.
"""
import numpy as np

RES = "data/results"
SUFFIX = "by_graph_bma"

METHODS = {
    "LinkGDA-p":   "dim_400_bs_32768_lr_0.001_pheno_proj_owl2vecstar_gda_use_graph_True",
    "LinkGDA-pf":  "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True",
    "LinkGDA-pfs": "dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True",
    "LinkGDA-f":   "dim_100_bs_16384_lr_0.001_func_proj_owl2vecstar_gda_use_graph_True",
    "INDIGENA":    "dim_200_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_inductive",
}


def fold_mean_rank(config, fold):
    # INDIGENA files use the "_inductive_bma" tail; graph methods use "_by_graph_bma".
    tail = "bma" if config.endswith("inductive") else SUFFIX
    path = f"{RES}/kge_results_transd_fold_{fold}_seed_0_{config}_{tail}.tsv"
    ranks = []
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            idx = int(p[2])
            sc = np.asarray(p[3:], dtype=float)
            ranks.append(1 + int(np.count_nonzero(sc > sc[idx])))
    return float(np.mean(ranks))


print(f"{'method':13} {'all-10 MR':>10} {'folds3-9 MR':>12} {'folds0-2 MR':>12} {'inflation':>10}")
for name, cfg in METHODS.items():
    try:
        fm = [fold_mean_rank(cfg, f) for f in range(10)]
    except FileNotFoundError as e:
        print(f"{name:13} MISSING: {e.filename.split('/')[-1][:50]}")
        continue
    all10 = np.mean(fm)
    clean = np.mean(fm[3:])
    tuned = np.mean(fm[:3])
    print(f"{name:13} {all10:>10.2f} {clean:>12.2f} {tuned:>12.2f} {clean - all10:>+10.2f}")
