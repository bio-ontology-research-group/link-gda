"""
Fold-level paired significance tests for ranking methods on the GDA benchmark.

Why this exists (vs. p_value.py):
    p_value.py pools all ~6,571 (fold, disease, gene) test instances and treats
    them as independent paired samples for a t-test / Wilcoxon. They are NOT
    independent: the same gene recurs across many disease pairs, and every
    instance within a fold is scored by the SAME trained model. That
    pseudo-replication inflates the effective sample size and makes p-values look
    far stronger than the evidence warrants (a 52.7% win rate reaching p<1e-14).

    This script instead uses the 10 cross-validation folds as the unit of
    (approximate) independence. Each fold is collapsed to ONE number -- the mean
    rank of each method in that fold -- giving 10 paired observations. A single
    test is then run over those 10 folds. This is the honest, conservative
    version: with only 10 points, a genuinely tiny effect may not reach
    significance, which is the correct outcome.

We report, per comparison:
    - the OLD pooled numbers (for reference / contrast). The pooled t-test and
      pooled Wilcoxon disagree for some variants, and that disagreement is
      informative rather than a contradiction: the t-test weighs the SIZE of each
      difference, Wilcoxon only counts how OFTEN each method wins. A method that
      loses most instances but wins hugely when it wins (tail compression) passes
      the t-test and fails Wilcoxon.
    - per-fold mean-rank difference (A - B), mean +/- std over the 10 folds,
    - how many of the 10 folds A beats B (sign test, binomial),
    - a paired t-test (PRIMARY) and Wilcoxon signed-rank test over the 10 fold
      means. The paired t-test is the primary test: each fold value is a mean over
      ~657 instances, so the CLT makes normality reasonable, whereas Wilcoxon
      discards magnitudes and floors at p ~ 1/2^10 = 0.001 with 10 folds.

All tests here are for PAIRED data (both methods rank the same diseases in the
same folds). Mann-Whitney U / Wilcoxon rank-sum would be WRONG here; INDIGENA's
p_value.r runs one with its own "not appropriate if same diseases" warning.

Limitation, stated rather than corrected: the 10 folds are not perfectly
independent, since any two training sets share ~80% of the data, so these
p-values are mildly optimistic. Corrections for this correlation exist, but they
are acknowledged approximations rather than exact tests, they are absent from
the standard statistical libraries (scipy/statsmodels/base R), and they change
no conclusion here, so we report the uncorrected test and note the caveat.

Each result file
    data/results/kge_results_{ARCH}_fold_{fold}_seed_0_{CONFIG}.tsv
has rows:  gene <TAB> disease <TAB> gene_index <TAB> score_1 ... score_4399
The rank of the true gene = 1 + (#candidates scoring strictly higher).
"""
import numpy as np
from scipy import stats

RESULTS_DIR = "data/results"; N_FOLDS = 10
FILENAME = "kge_results_{arch}_fold_{fold}_seed_0_{config}.tsv"

METHODS = {
 "MultiHopGDA-p":          ("transd",  "dim_400_bs_32768_lr_0.001_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "MultiHopGDA-pf":         ("transd",  "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "MultiHopGDA-pfs":        ("transd",  "dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "MultiHopGDA-f":          ("transd",  "dim_100_bs_16384_lr_0.001_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 # NOTE: Table 1 reports INDIGENA at MR 876.78, which is the OWL2Vec* (non-GDA)
 # projector run (verified: 876.58 +/- 63.53), NOT the GDAProjector run
 # (902.13 +/- 70.58) that the original p_value.py pointed to. We compare against
 # the OWL2Vec* INDIGENA so the significance test matches the baseline in the table.
 "INDIGENA":               ("transd",  "dim_200_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_inductive_bma"),
 "MultiHopGDA-p-convkbd":  ("convkbd", "dim_400_bs_16384_lr_1e-05_hdr_0.0_nf_200_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "MultiHopGDA-f-convkbd":  ("convkbd", "dim_100_bs_32768_lr_0.0001_hdr_0.0_nf_200_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
}
# (method_A, method_B): tests whether A has LOWER (better) ranks than B.
COMPARISONS = [("MultiHopGDA-p",         "INDIGENA"),
               ("MultiHopGDA-pf",        "INDIGENA"),
               ("MultiHopGDA-pfs",       "INDIGENA"),
               ("MultiHopGDA-f",         "MultiHopGDA-p"),
               ("MultiHopGDA-f-convkbd", "MultiHopGDA-p-convkbd")]

def per_instance_ranks(arch, config):
    """Ranks keyed by (fold, disease, gene), or None if any fold file is missing."""
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


needed = {m for cmp in COMPARISONS for m in cmp}
cache = {m: per_instance_ranks(arch, c) for m, (arch, c) in METHODS.items() if m in needed}

for A, B in COMPARISONS:
    absent = [m for m in (A, B) if cache.get(m) is None]
    print(f"=== {A} vs {B} ===")
    if absent:
        print(f"  SKIPPED: no result files for {', '.join(absent)}\n")
        continue
    a, b = cache[A], cache[B]
    keys = sorted(set(a) & set(b))
    ra = np.array([a[k] for k in keys]); rb = np.array([b[k] for k in keys])

    # --- OLD pooled view (for reference; the overpowered one) ---
    d = ra - rb
    _, pt_pool = stats.ttest_rel(ra, rb, alternative='less')
    _, pw_pool = stats.wilcoxon(ra, rb, alternative='less')
    print(f"  [pooled, n={len(keys)}]  mean rank {ra.mean():8.2f} vs {rb.mean():8.2f}"
          f"   {A} better on {100*np.mean(d<0):.1f}% of instances")
    print(f"  [pooled]  paired t p={pt_pool:.2e}   Wilcoxon(less) p={pw_pool:.2e}"
          f"   <-- inflated by pseudo-replication")

    # --- NEW fold-level view (the honest one) ---
    folds = sorted({k[0] for k in keys})
    mean_a, mean_b, winrate = [], [], []
    for fk in folds:
        idx = [i for i, k in enumerate(keys) if k[0] == fk]
        fa, fb = ra[idx], rb[idx]
        mean_a.append(fa.mean()); mean_b.append(fb.mean())
        winrate.append(np.mean(fa < fb))
    mean_a = np.array(mean_a); mean_b = np.array(mean_b)
    fold_diff = mean_a - mean_b                       # negative = A better that fold
    n_better = int(np.sum(fold_diff < 0))

    print(f"  [per-fold, n={len(folds)} folds]  mean rank {mean_a.mean():8.2f} vs {mean_b.mean():8.2f}")
    print(f"  per-fold diff (A-B): mean {fold_diff.mean():+.2f} +/- {fold_diff.std(ddof=1):.2f}"
          f"   (per-fold win rate {100*np.mean(winrate):.1f}%)")
    print(f"  A better in {n_better}/{len(folds)} folds")

    sign = stats.binomtest(n_better, len(folds), 0.5, alternative='greater')
    print(f"  sign test (A better in >half of folds): p={sign.pvalue:.3f}")

    t_rel, p_rel = stats.ttest_rel(mean_a, mean_b, alternative='less')
    print(f"  PRIMARY paired t over folds (A<B): t={t_rel:.3f}  p={p_rel:.2e}  (df={len(folds)-1})")

    try:
        _, p_wil = stats.wilcoxon(mean_a, mean_b, alternative='less')
        print(f"  Wilcoxon over folds (A<B):        p={p_wil:.3f}  (floor ~0.001 at n=10)")
    except ValueError as e:
        print(f"  Wilcoxon over folds:              n/a ({e})")
    print()
