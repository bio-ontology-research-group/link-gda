"""
Dump per-fold mean-rank vectors for the significance-test methods, so the
Nadeau-Bengio corrected paired t-test (correctR::kfold_ttest in R) can be run on
them off-cluster. Reads the large per-instance rank TSVs on ibex; writes ONLY the
tiny per-fold aggregates (10 floats per method) to JSON.

Method configs and comparison structure are copied from p_value_per_fold.py so
the vectors match exactly what the paper's fold-level test consumes.
"""
import json
import numpy as np

RESULTS_DIR = "data/results"; N_FOLDS = 10
FILENAME = "kge_results_{arch}_fold_{fold}_seed_0_{config}.tsv"

METHODS = {
 "LinkGDA-p":   ("transd",  "dim_400_bs_32768_lr_0.001_pheno_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "LinkGDA-pf":  ("transd",  "dim_200_bs_32768_lr_0.001_pheno_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "LinkGDA-pfs": ("transd",  "dim_100_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "LinkGDA-f":   ("transd",  "dim_100_bs_16384_lr_0.001_func_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "INDIGENA":    ("transd",  "dim_200_bs_32768_lr_0.001_pheno_func_expr_proj_owl2vecstar_use_graph_False_inductive_bma"),
 # RQ3 projector ablation pairs (GDAProjector vs OWL2Vec*)
 "LinkGDA-f-owl":  ("transd", "dim_100_bs_16384_lr_0.001_func_proj_owl2vecstar_use_graph_True_by_graph_bma"),
 "LinkGDA-s":      ("transd", "dim_400_bs_32768_lr_0.0001_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "LinkGDA-s-owl":  ("transd", "dim_400_bs_16384_lr_0.0001_expr_proj_owl2vecstar_use_graph_True_by_graph_bma"),
 "LinkGDA-fs":     ("transd", "dim_100_bs_32768_lr_0.001_func_expr_proj_owl2vecstar_gda_use_graph_True_by_graph_bma"),
 "LinkGDA-fs-owl": ("transd", "dim_100_bs_32768_lr_0.001_func_expr_proj_owl2vecstar_use_graph_True_by_graph_bma"),
}

# For each comparison we will need the paired per-fold vectors of A and B.
COMPARISONS = [
 ("LinkGDA-p",   "INDIGENA",      "less"),        # RQ1
 ("LinkGDA-pf",  "INDIGENA",      "less"),        # RQ1
 ("LinkGDA-pfs", "INDIGENA",      "less"),        # RQ1
 ("LinkGDA-f",   "LinkGDA-f-owl",  "two-sided"),  # RQ3 projector
 ("LinkGDA-s",   "LinkGDA-s-owl",  "two-sided"),  # RQ3 projector
 ("LinkGDA-fs",  "LinkGDA-fs-owl", "two-sided"),  # RQ3 projector
]

def per_fold_mean_ranks(arch, config):
    """Return (per_fold_mean[list of N_FOLDS], keys_per_fold) or None if missing.
    Rank of true gene = 1 + (#candidates scoring strictly higher)."""
    per_fold = {}
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
                r = 1 + int(np.count_nonzero(sc > sc[pos]))
                per_fold.setdefault(fold, {})[(p[1], p[0])] = r
    return per_fold

needed = {m for cmp in COMPARISONS for m in cmp[:2]}
cache = {}
for m in needed:
    arch, cfg = METHODS[m]
    pf = per_fold_mean_ranks(arch, cfg)
    if pf is None:
        print(f"WARNING: missing files for {m}")
    cache[m] = pf

out = {"n_folds": N_FOLDS, "comparisons": []}
for A, B, alt in COMPARISONS:
    a, b = cache.get(A), cache.get(B)
    if a is None or b is None:
        out["comparisons"].append({"A": A, "B": B, "alt": alt, "skipped": True})
        continue
    vec_a, vec_b, n_test = [], [], []
    for fold in range(N_FOLDS):
        # pair on shared (disease, gene) keys within the fold
        keys = sorted(set(a[fold]) & set(b[fold]))
        ra = np.array([a[fold][k] for k in keys], dtype=float)
        rb = np.array([b[fold][k] for k in keys], dtype=float)
        vec_a.append(float(ra.mean())); vec_b.append(float(rb.mean()))
        n_test.append(len(keys))
    out["comparisons"].append({
        "A": A, "B": B, "alt": alt,
        "mean_rank_A": vec_a, "mean_rank_B": vec_b,
        "n_test_per_fold": n_test,
    })
    print(f"{A} vs {B}: A folds mean {np.mean(vec_a):.2f}, B folds mean {np.mean(vec_b):.2f}, "
          f"A better in {int(np.sum(np.array(vec_a) < np.array(vec_b)))}/{N_FOLDS}")

with open("data/perfold_vectors.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote data/perfold_vectors.json")
