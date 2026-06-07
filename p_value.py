"""
Paired significance tests for ranking methods on the multihop-gda GDA benchmark.
Adapted from the INDIGENA p_value.py.

Each result file
    data/results/kge_results_transd_fold_{fold}_seed_0_{CONFIG}.tsv
has rows:  gene <TAB> disease <TAB> gene_index <TAB> score_1 ... score_4399
The rank of the true gene = 1 + (#candidates scoring strictly higher).

Test instances (fold, disease, gene) are paired across two methods and pooled over
the 10 folds. For each comparison we report mean/median rank, per-instance win rate,
a paired t-test (mean) and a Wilcoxon signed-rank test (robust), as in INDIGENA.
CONFIG = the filename part between "seed_0_" and ".tsv".
"""
import numpy as np
from scipy import stats

RESULTS_DIR = "data/results"; N_FOLDS = 10

METHODS = {
 "MultiHopGDA-p":  "dim_400_bs_32768_lr_0.001_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma",
 "MultiHopGDA-pf": "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma",
 "MultiHopGDA-pfs":"dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma",
 "INDIGENA":       "dim_200_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_False_inductive_bma",
 # RQ2 (generate -f result files first):
 # "MultiHopGDA-f":"dim_100_bs_32768_lr_0.001_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma",
}
# (method_A, method_B): tests whether A has LOWER (better) ranks than B
COMPARISONS = [("MultiHopGDA-p","INDIGENA"),
               ("MultiHopGDA-pf","INDIGENA"),
               ("MultiHopGDA-pfs","INDIGENA")]

def per_instance_ranks(config):
    ranks = {}
    for fold in range(N_FOLDS):
        with open(f"{RESULTS_DIR}/kge_results_transd_fold_{fold}_seed_0_{config}.tsv") as f:
            for line in f:
                p = line.rstrip("\n").split("\t"); pos = int(p[2])
                sc = np.asarray(p[3:], dtype=float)
                ranks[(fold, p[1], p[0])] = 1 + int(np.count_nonzero(sc > sc[pos]))
    return ranks

cache = {m: per_instance_ranks(c) for m, c in METHODS.items() if m in {x for cmp in COMPARISONS for x in cmp}}
for A, B in COMPARISONS:
    a, b = cache[A], cache[B]; keys = sorted(set(a) & set(b))
    ra = np.array([a[k] for k in keys]); rb = np.array([b[k] for k in keys]); d = ra - rb
    print(f"=== {A} vs {B}  (n={len(keys)}) ===")
    print(f"  mean rank  {ra.mean():8.2f} vs {rb.mean():8.2f}   median {np.median(ra):.0f} vs {np.median(rb):.0f}")
    print(f"  {A} better on {100*np.mean(d<0):.1f}% of instances")
    t, pt = stats.ttest_rel(ra, rb, alternative='less')
    print(f"  paired t-test ({A}<{B}): t={t:.3f}  p={pt:.3e}")
    wl, pl = stats.wilcoxon(ra, rb, alternative='less')
    w2, p2 = stats.wilcoxon(ra, rb, alternative='two-sided')
    print(f"  Wilcoxon less p={pl:.3e}   two-sided p={p2:.3e}")
