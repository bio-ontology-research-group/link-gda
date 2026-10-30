# Matched KGE analysis, fixed before production test scoring

This specifies the graph-pair analysis required by the training protocol
`matched-kge-20261029.md`. It does not change that protocol, its training
settings, or the source snapshot used by the running campaign. The benchmark
has already informed development; this is a controlled sensitivity on reused
data, not independent external validation.

Require all 100 completed trajectories, their global freeze and all 400 score
files. Verify them with `build_matched_kge_manifest.py`, then use the shared
score evaluator to save its report and per-association ranks. Retain all four
validation choices for each model and graph, with raw and calibrated scores.

The primary choice and scoring mode are fixed:

| Family | Validation checkpoint | Test scoring |
|---|---|---|
| U1 | Calibrated mean rank | Per-gene calibrated |
| T1 | Raw head metrics | Raw |
| INDIGENA | Raw mean rank | Raw |
| ConvKB-D | Calibrated mean rank | Per-gene calibrated |
| P-pool | Raw head metrics | Raw |

For each family, compare no-bridge minus original on the same associations,
using that family's primary checkpoint and scoring mode. Pool associations
across all ten folds. The two primary endpoints are Hits@10, in percentage
points, and mean rank, in rank units. Positive Hits@10 differences and negative
mean-rank differences indicate improvement. These five families and two
endpoints constitute one family of ten comparisons.

Use 10,000 paired phenotype-profile-cluster bootstrap replicates with NumPy
generator seed 20261029. A cluster contains all evaluated associations having
the same sorted unique phenotype set, including repeated profiles across folds.
Sample clusters with replacement, retain the associations and their original
weights within each selected cluster, and divide by the number of resampled
associations. Apply identical resampling indices to all ten contrasts. Report
the empirical mean difference and the 0.25th/99.75th bootstrap percentiles:
99.5% marginal intervals, with nominal Bonferroni family coverage of 95%.
These intervals measure benchmark sampling variation conditional on the fitted
models; one training seed does not estimate training-seed uncertainty.

Check reconstructed primary metrics against the evaluator report and verify
rank-row identities against the frozen test pairs and phenotype profiles.
Record input and output checksums, the manifest, evaluation source, bootstrap
settings, candidate and association counts, and actual graph sizes and epoch
budgets. Graph removal changes embedding inventories and updates per epoch;
do not describe it as an equal-update or equal-FLOP comparison.

All other validation choices, ranking endpoints, calibration modes and the
42 scoring/cohort conditions are descriptive sensitivities. Do not change
primary choices based on their test results or combine these matched contrasts
with the earlier unmatched comparisons under a shared result label.
