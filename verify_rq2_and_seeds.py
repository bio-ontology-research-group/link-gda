"""Verify the RQ2 fold-level claim from regenerated LinkGDA-f, and report seed variance.

Part 1: reproduce the RQ2 headline. Per-fold mean rank for LinkGDA-f and LinkGDA-p at
seed 0 over the ten folds, and the one-sided paired t-test (f < p) the paper reports.

Part 2: seed variance on fold 0. Mean rank per seed for each variant, its spread, and
the fold-0 gap between them beside the per-seed noise.
"""
import numpy as np
from scipy import stats

RES = "data/results"
FCFG = "dim_100_bs_16384_lr_0.001_func"
PCFG = "dim_400_bs_32768_lr_0.001_pheno"
SUFFIX = "proj_owl2vecstar_gda_use_graph_True_by_graph_bma"


def mean_rank(path):
    r = []
    for line in open(path):
        q = line.rstrip("\n").split("\t")
        if len(q) < 4:
            continue
        i = int(q[2])
        s = np.asarray(q[3:], dtype=float)
        r.append(1 + int(np.count_nonzero(s > s[i])))
    return float(np.mean(r))


def fold_means(cfg):
    return np.array([mean_rank(f"{RES}/kge_results_transd_fold_{fold}_seed_0_{cfg}_{SUFFIX}.tsv")
                     for fold in range(10)])


def seed_means(cfg):
    return np.array([mean_rank(f"{RES}/kge_results_transd_fold_0_seed_{seed}_{cfg}_{SUFFIX}.tsv")
                     for seed in range(10)])


print("=" * 68)
print("PART 1  RQ2 reproduction: LinkGDA-f vs LinkGDA-p, per fold, seed 0")
print("=" * 68)
f, p = fold_means(FCFG), fold_means(PCFG)
diff = f - p
print("  -f folds:", " ".join(f"{x:7.1f}" for x in f))
print("  -p folds:", " ".join(f"{x:7.1f}" for x in p))
print(f"  -f attains lower mean rank in {int((f < p).sum())}/10 folds")
print(f"  mean gap (f - p): {diff.mean():+.2f} +/- {diff.std(ddof=1):.2f}")
_, pv = stats.ttest_rel(f, p, alternative="less")
print(f"  one-sided paired t (f < p): p = {pv:.2e}")
print("  paper states: 10/10 folds, ~82-position gap, p = 2.2e-5")

print("\n" + "=" * 68)
print("PART 2  Seed variance on fold 0 (val_seed fixed at 0)")
print("=" * 68)
for name, cfg in (("LinkGDA-f", FCFG), ("LinkGDA-p", PCFG)):
    s = seed_means(cfg)
    print(f"  {name:12s} mean {s.mean():8.2f} +/- {s.std(ddof=1):6.2f}   [min {s.min():.1f}, max {s.max():.1f}]")
sf, sp = seed_means(FCFG), seed_means(PCFG)
print(f"\n  fold-0 gap (f - p) at seed 0        : {sf[0] - sp[0]:+.2f}")
print(f"  seed sd of that gap across 10 seeds : {(sf - sp).std(ddof=1):.2f}")
print(f"  does the gap clear its own seed noise? gap {abs((sf-sp).mean()):.0f} vs sd {(sf-sp).std(ddof=1):.0f}")
