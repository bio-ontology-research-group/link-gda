"""
Paired significance tests for ranking methods on the multihop-gda GDA benchmark.
Adapted from the INDIGENA p_value.py.

Each result file
    data/results/kge_results_{ARCH}_fold_{fold}_seed_0_{CONFIG}.tsv
where ARCH is "transd" or "convkbd", has rows:
    gene <TAB> disease <TAB> gene_index <TAB> score_1 ... score_4399
The rank of the true gene = 1 + (#candidates scoring strictly higher).

Test instances (fold, disease, gene) are paired across two methods and pooled over
the 10 folds. For each comparison we report mean/median rank, per-instance win rate,
a paired t-test (mean) and a Wilcoxon signed-rank test (robust), as in INDIGENA.
CONFIG = the filename part between "seed_0_" and ".tsv".
"""
import numpy as np
from scipy import stats

RESULTS_DIR = "data/results"; N_FOLDS = 10
FILENAME = "kge_results_{arch}_fold_{fold}_seed_0_{config}.tsv"

METHODS = {
 "LinkGDA-p":          ("transd",  "dim_400_bs_32768_lr_0.001_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "LinkGDA-pf":         ("transd",  "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "LinkGDA-pfs":        ("transd",  "dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "LinkGDA-f":          ("transd",  "dim_100_bs_16384_lr_0.001_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 # FIXED: this previously pointed at the owl2vecstar_gda INDIGENA run (mean rank
 # 902.13), but Table 1 of the paper reports the owl2vecstar (non-GDA) INDIGENA
 # (recomputed 876.58 +/- 63.53, matching the tabulated 876.78 +/- 63.56). The
 # significance test was therefore comparing against a weaker baseline than the
 # one in the table. Now points at the owl2vecstar run so the two agree.
 "INDIGENA":               ("transd",  "dim_200_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_inductive_bma"),
 "LinkGDA-p-convkbd":  ("convkbd", "dim_400_bs_16384_lr_1e-05_hdr_0.0_nf_200_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "LinkGDA-f-convkbd":  ("convkbd", "dim_100_bs_32768_lr_0.0001_hdr_0.0_nf_200_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
}
# (method_A, method_B): tests whether A has LOWER (better) ranks than B.
# RQ1: each phenotype-bearing variant against the INDIGENA baseline.
# RQ2: -f against -p, run under both architectures to check the result is not
#      an artefact of the embedding model.
COMPARISONS = [("LinkGDA-p",         "INDIGENA"),
               ("LinkGDA-pf",        "INDIGENA"),
               ("LinkGDA-pfs",       "INDIGENA"),
               ("LinkGDA-f",         "LinkGDA-p"),
               ("LinkGDA-f-convkbd", "LinkGDA-p-convkbd")]

def per_instance_ranks(arch, config):
    """Ranks for all folds, or None if any fold's result file has not been generated yet."""
    ranks = {}
    for fold in range(N_FOLDS):
        fname = FILENAME.format(arch=arch, fold=fold, config=config)
        try:
            f = open(f"{RESULTS_DIR}/{fname}")
        except FileNotFoundError:
            return None
        with f:
            for line in f:
                p = line.rstrip("\n").split("\t"); pos = int(p[2])
                sc = np.asarray(p[3:], dtype=float)
                ranks[(fold, p[1], p[0])] = 1 + int(np.count_nonzero(sc > sc[pos]))
    return ranks

cache = {m: per_instance_ranks(arch, c) for m, (arch, c) in METHODS.items() if m in {x for cmp in COMPARISONS for x in cmp}}
for A, B in COMPARISONS:
    absent = [m for m in (A, B) if cache[m] is None]
    if absent:
        print(f"=== {A} vs {B} ===")
        print(f"  SKIPPED: no result files for {', '.join(absent)}\n")
        continue
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
