# Nadeau-Bengio corrected fold-level paired t-test for the GDA ranking results.
# Vectors are the 10 per-fold mean ranks (one number per disease-disjoint fold),
# extracted on ibex from the saved per-instance ranks (data/perfold_vectors.json).
#
# We report, per comparison:
#   (1) naive paired t over folds  (factor 1/k,        df k-1)  -- overpowered
#   (2) textbook Nadeau-Bengio     (factor 1/k+1/(k-1), df k-1) -- primary
#   (3) correctR::kfold_ttest      (the published library)      -- cross-check
#   (4) sign test over folds (binomial, direction-free evidence)
# (2) is the literature-standard corrected resampled t-test for k-fold CV
# (Nadeau & Bengio 2003; Bouckaert & Frank 2004). (3) should track (2) closely;
# any gap is the library's formula variant, reported honestly.

suppressMessages(library(correctR))
K <- 10

# --- helper: paired t over folds with an arbitrary variance-inflation factor ---
fold_t <- function(a, b, factor, df, alt) {
  d <- a - b                      # negative => A has lower (better) rank
  t <- mean(d) / sqrt(var(d) * factor)
  p <- switch(alt,
    less     = pt(t, df),          # H1: mean(d) < 0  (A better)
    greater  = 1 - pt(t, df),
    twosided = 2 * pt(-abs(t), df))
  list(t = t, p = p, dbar = mean(d), df = df)
}

run <- function(name, a, b, alt) {   # alt in {"less","twosided"}
  stopifnot(length(a) == K, length(b) == K)
  d <- a - b
  n_better <- sum(d < 0)             # folds where A ranks better
  naive <- fold_t(a, b, 1/K,             K - 1, alt)
  nb    <- fold_t(a, b, 1/K + 1/(K-1),   K - 1, alt)
  # correctR: n = number of paired CV results = K (self-consistent, df=K-1);
  # one-sided "A better" => test that B's value (y) is greater => greater = "y".
  if (alt == "less") {
    cr <- kfold_ttest(x = a, y = b, n = K, k = K, tailed = "one", greater = "y")
  } else {
    cr <- kfold_ttest(x = a, y = b, n = K, k = K, tailed = "two")
  }
  sign_alt <- if (alt == "less") "greater" else "two.sided"
  sg <- binom.test(n_better, K, 0.5, alternative = sign_alt)

  cat(sprintf("=== %s  [%s] ===\n", name, alt))
  cat(sprintf("  per-fold diff (A-B): mean %+8.2f  sd %6.2f   A better in %d/%d folds\n",
              mean(d), sd(d), n_better, K))
  cat(sprintf("  (1) naive paired-t   : t=%7.3f  p=%.3e   (df=%d)\n", naive$t, naive$p, naive$df))
  cat(sprintf("  (2) Nadeau-Bengio    : t=%7.3f  p=%.3e   (df=%d)   [PRIMARY]\n", nb$t, nb$p, nb$df))
  cat(sprintf("  (3) correctR kfold   :            p=%.3e            [library check]\n", cr$p.value))
  cat(sprintf("  (4) sign test        :            p=%.3e   (%d/%d)\n", sg$p.value, n_better, K))
  cat("\n")
}

# ---- RQ1: each LinkGDA phenotype variant vs INDIGENA (one-sided, A better) ----
indigena <- c(837.4650793650794,848.6912181303117,951.1733128834355,783.006309148265,
              854.160741885626,886.2919708029198,845.1775417298937,1013.7333333333333,
              832.306990881459,913.7786259541984)

p_var <- c(726.1476190476191,830.5198300283286,747.786809815951,792.3533123028391,
           764.5023183925812,757.1708029197081,735.3034901365705,826.4,
           704.273556231003,817.8106870229008)
pf_var <- c(555.2285714285714,603.5042492917847,573.5874233128834,578.9132492113565,
            607.6676970633694,598.3795620437957,607.2716236722307,618.1565891472868,
            581.9559270516718,629.2641221374046)
pfs_var <- c(579.063492063492,565.0368271954674,551.3588957055215,569.2066246056783,
             557.4482225656878,636.0905109489051,559.6418816388467,600.862015503876,
             607.7021276595744,608.0244274809161)

run("LinkGDA-p  vs INDIGENA",  p_var,   indigena, "less")
run("LinkGDA-pf vs INDIGENA",  pf_var,  indigena, "less")
run("LinkGDA-pfs vs INDIGENA", pfs_var, indigena, "less")

# ---- RQ3 projector ablation available here: -fs GDAProjector vs OWL2Vec* (2-sided) ----
fs_gda <- c(657.3873015873016,653.0382436260624,681.8128834355829,699.8927444794953,
            715.1715610510047,808.5138686131387,654.1805766312594,794.6651162790698,
            638.4893617021277,674.018320610687)
fs_owl <- c(641.6142857142858,687.570821529745,621.4493865030674,716.0615141955836,
            693.6476043276662,738.7065693430657,707.0,702.6821705426356,
            629.6428571428571,707.7603053435115)
run("LinkGDA-fs GDA vs OWL2Vec*", fs_gda, fs_owl, "twosided")
