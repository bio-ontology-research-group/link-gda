# Reproduce the PhenoQuery paper

The [manuscript snapshot](../manuscript/README.md) contains the complete
controlled-query, matched embedding-model and baseline results. This guide
connects those results to their inputs and commands. The
[status record](../reproducibility/STATUS.txt) supplies the verification evidence
and remaining requirements for the current release.

## Decide which stage to reproduce

| Stage | Inputs | Result |
|---|---|---|
| Table generation | Verified JSON reports | LaTeX tables and metric exports |
| Saved-score analysis | Predictions, candidate/query identities, benchmark inputs and frozen selections | Ranks, metrics, bootstrap contrasts and bias analyses |
| Checkpoint inference | Model weights, graph inputs, original scorer and its environment | Fresh predictions to compare with the verified reference |
| Training | Prepared inputs, pretrained initializations, fixed protocol and training environment | Validation trajectories, selected weights and predictions |

Each stage has its own checks. A successful table build establishes rendering
from the supplied report. Saved-score replay additionally checks the numerical
analysis. Checkpoint inference verifies model-to-prediction execution; retraining
also exercises optimization and validation selection.

## Inputs and benchmark identity

Start with the [frozen-input recipe](frozen-inputs.md), which downloads and
checks the prepared benchmark, ontology projections, exact ULTRA source,
pretrained checkpoints and environment capture. The matched KGE comparison
also requires the gene-to-MP annotation supplement linked there. Use a fresh
workspace and set `TMPDIR` to its scratch directory.

The main benchmark contains 6,571 associations from 6,059 diseases and ranks
13,625 phenotype-annotated candidate genes. Disease-disjoint splits retain
both previously associated and unassociated genes. The zero-same-gene-overlap
cohort contains 4,431 associations. The combined S4 protocol retains 4,265
associations from 4,017 diseases and weights 4,011 distinct phenotype profiles
equally. Each report specifies its scoring rule, cohort and weighting unit.

The separate excluded-gene experiment contains 425 associations from 417
diseases. Its primary candidate pool has 5,696 GO-annotated genes without MGI
phenotypes; its secondary pool has 18,904 GO-annotated genes. The
[embedding replay](embedding-archive-replay.md) supplies these exact inputs
and their selection funnel.

Some original annotation-provider snapshots are unavailable. Exact reproduction
therefore begins with the retained prepared tables and pinned ontologies.
The generic `prepare` dispatcher downloads current resources and creates new
association tables, folds and projections. Use its outputs for a new experiment;
the frozen manifests identify the paper's inputs.

## Regenerate the main controlled tables

Python's standard library is sufficient for these two commands. From the
repository root, choose fresh output directories:

```bash
python3 reproducibility/generate_controlled_query_tables.py \
  --report-json reproducibility/results/controlled-query-report.json \
  --out-dir generated-results/query
python3 reproducibility/generate_matched_kge_tables.py \
  --report-json reproducibility/results/matched-kge-summary.json \
  --out-dir generated-results/matched
```

Compare the three query and two matched `.tex` outputs with their namesakes in
`manuscript/tables/`. The query report is byte-identical to the report verified
by the [fresh public saved-score replay](../reproducibility/verification/controlled-query-public-score-replay.json).
The matched summary comes from the
[complete native analysis](../reproducibility/verification/matched-kge-complete-verification.json).

To regenerate the supporting bias and historical query tables, use the
[`generate-tables` workflow](../reproducibility/README.md#regenerate-the-saved-bias-tables)
with the pinned CPU analysis environment. Detailed controlled and matched bias
panels use the [production bias-table renderer](production-bias-tables.md),
which checks each report's metadata and corresponding score manifest.

## Recompute predictions-to-results analyses

Install the locked [CPU analysis environment](../reproducibility/environments/cpu-analysis/README.md)
and keep its package capture unchanged. Each recipe below downloads the
appropriate archives, verifies their member hashes and reconstructs the analysis
in a new workspace.

| Result family | Complete recipe | Verified scope |
|---|---|---|
| Controlled PhenoQuery | [Query saved-score replay](controlled-query-score-replay.md) | Fresh public replay of 480 bundles, eight primary contrasts, all 48-by-42 bias conditions and 6,720 rank arrays |
| Matched embedding models | [Matched saved-score replay](matched-kge-score-replay.md) | Full fresh public download and replay pass: 400 score files, 40-by-42 conditions, 5,600 exact rank arrays and ten contrasts |
| Native and adapted Phrank/PhenoRank | [Final baseline results](final-baseline-results.md) | Fresh public replay of the complete 26-variant collection |
| Supporting graph and query sensitivities | [Historical comparator replay](historical-comparator-replay.md) | Fresh public replay of the complete 25-variant collection |
| Embedding, annotation and excluded-cohort results | [Embedding archive replay](embedding-archive-replay.md) | Fresh public replay of 314 metric outputs, controls, statistical tests, eleven tables and rank-CDF artifacts |

The query score release alone contains approximately 73.1 GB of compressed
archives. Extraction and intermediate ranks need additional space. Follow each
recipe's storage instructions before downloading; the compact table-generation
commands above require only the repository reports.

Archive manifests identify the analysis source, benchmark, configurations and
reference outputs. Use the source revision and environment documented for that
collection. Preserve original checkpoints and verified outputs; write regenerated
outputs to separate directories. Rank-container compression can vary between
Python builds, so the matched replay compares exact array payloads and accounts
for container hashes explicitly.

## Generate fresh predictions and train models

For the query models, follow [checkpoint inference](controlled-query-checkpoints.md)
using the released initial and selected weights, archived ULTRA source and
[query environment](frozen-inputs.md#reconstruct-the-ultra-environment).
Then run the full saved-score and bias analysis on the regenerated predictions.
The [controlled-training instructions](controlled-query.md) specify planning,
technical pilots, resumable training, the global validation freeze and test
scoring for all 240 runs. The four arms share a 500-update budget, two
initializations, three seeds and ten folds. Both initial and validation-selected
weights are evaluated.

For the matched embedding comparison, use the
[matched KGE environment](../reproducibility/environments/matched-kge/README.md),
[training and analysis instructions](kge-execution.md), and
[checkpoint recipe](matched-kge-checkpoints.md). Each of five families is trained
on both graph constructions in ten folds with seed 0. Dimensions, objectives,
epoch budgets, warm starts and validation criteria are fixed within each family.
Four validation choices are retained per run. The primary analysis uses the
prespecified choice and scoring mode for each family.

The original embedding experiments and supporting single-seed query campaigns
retain their own configurations and scientific scope. Their saved-score recipes
above reproduce the supporting results. The dispatcher's `train-ultra` and
`train-phenoquery` commands invoke those historical query designs; the controlled
experiment uses `run_controlled_query.py` as documented in its dedicated guide.

Native HPO Phrank, historical-resource PhenoRank and the separate UPheno
adaptations have [baseline execution instructions](native-baselines.md).
Keep native rankings and each adaptation's resource choice identifiable through
the analysis. Post-hoc score transformations and changes to evaluation cohorts
are separate operations.

## Build and inspect the paper

The [manuscript instructions](../manuscript/README.md) specify the TeX toolchain,
four-command cross-document build, fixed build timestamp and figure generators.
The source manifest binds all 49 manuscript inputs. A fresh public source build
reproduces the 12-page paper and 46-page supplement byte for byte in the checked
TeX environment, and all pages have been visually inspected.

Run the [CPU regression suite](testing.md) after implementation changes. It
covers the training/resume machinery on small graphs, score handling, calibration,
archive integrity and report comparisons. Full-data reproduction additionally
requires the collection-specific numerical and inference checks above.
