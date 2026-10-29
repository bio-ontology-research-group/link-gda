# Reproduction workflows and verification scope

This branch repairs the experimental package for the PhenoQuery method and bias
analysis. The [status record](STATUS.txt) distinguishes completed verification
from pending full-data experiments and artifact publication. It is not yet a
verified end-to-end release.

## Regenerate the saved bias tables

```bash
./reproducibility/run_pipeline.sh generate-tables
```

This regenerates `revised-bias-stepwise.tex`, `revised-bias-baselines-full.tex`
and `revised-bias-factors.tex` from `results/stepwise-bias-report-v4.json`.
It also generates the three historical query tables from the verified
`results/query-table-report.json`; see the [table audit](../docs/query-table-audit.md).
The last table isolates reference deduplication, co-causal filtering, profile
weighting and training-profile exclusion relative to the same calibrated
reference condition. Table numbers depend on the manuscript version.

The v4 report evaluates Exomiser from real scores. Its other 65 variants retain
the historical score files: 11,700 corresponding aggregate checks agree with
the audited report. In particular, its UPheno simulated PhenoRank rows still use
the previous weighted/capped sampler. They must be replaced when all folds of
the published-tier rerun finish. Its UPheno Phrank rows also use the historical
zero-root adaptation; the native-root rerun supplies their corrected replacements.
Native HPO Phrank is a separate experiment;
the historical JSON key `Phrank-HPO` denotes human-only UPheno in these tables.
Do not use the superseded v3 Exomiser status-standardized or combined cells.

## Prepared inputs and paths

The baseline/evaluation workspace contains `data/foldN` and baseline resources.
The no-bridge KGE `--data-root` points directly to the directory containing
`foldN`. These are different path conventions. The individual instructions
below give the required layouts.

`run_pipeline.sh prepare` downloads current resources, builds association tables
and outer folds, and projects ontologies. It does not yet reconstruct all of the
specialized inner-training graph layouts or replace the pinned data archive
required for reproducing the paper. Current downloads may differ from the
study's source snapshots. The [frozen-input instructions](../docs/frozen-inputs.md) provide the versioned
prepared inputs, ontology snapshots, pretrained checkpoints and exact ULTRA
source/environment capture. Full upstream preparation remains a separate gate.

## Controlled query experiment

Use [the controlled query instructions](../docs/controlled-query.md) and
[the frozen protocol](protocols/controlled-query-20261028.md). The design has
four graph/readout arms, two initializations, three seeds and ten folds.
All 240 validation selections are frozen before any outer-test scoring. Both
selected and initial checkpoints are evaluated. The protocol specifies the
paired profile-cluster bootstrap and primary comparison family.

The older `train-ultra` and `train-phenoquery` dispatcher commands call
`run_query_campaign.py` and require explicit `--root`, `--base` and
`--ultra-root` arguments. They reproduce the historical campaign design, not
the new controlled initialization/seed experiment.

## KGE experiments

See [KGE execution, input layout and checkpoint provenance](../docs/kge-execution.md).
The original top-hit route is `kge_transd.py --head_experiment CONFIG.json`.
There is no `kge_head_experiment.py` executable.

The no-bridge driver now resumes complete epoch states, checks source/input
identities and completed artifact hashes, and uses the requested ConvKB budget.
The retained original/no-bridge comparisons still differ in selection, dimension,
negative filtering and budget. They are descriptive results, not a controlled
estimate of bridge removal. `train-kge` runs the generic TransD configuration;
it does not train every KGE family or profile objective.

## Historical comparator scores

The historical comparator export is separate from the matched rerun:
`build_historical_comparator_manifest.py` checks 24 retained score matrices per
fold and reconstructs the direct-overlap aggregation control from the verified
phenotype cache. It requires all ten folds, prepared input hashes and cache
provenance, and emits a 25-variant manifest for the shared evaluator. The control
omits cached score entries; it does not alter the graph or refit the model.
The [public archive replay recipe](../docs/historical-comparator-replay.md)
downloads both benchmark inputs and comparator outputs and checks every result
against the released report. Publication and fresh-download replay remain
pending until recorded in `STATUS.txt`.

```bash
python reproducibility/build_historical_comparator_manifest.py \
  --base-dir "$HISTORICAL_WORKSPACE" --comparator-dir "$HISTORICAL_COMPARATORS" \
  --output "$HISTORICAL_MANIFEST"
```

## Native and UPheno baseline runs

See [native baseline instructions](../docs/native-baselines.md) for native HPO
Phrank, historical-resource PhenoRank and the separate UPheno adaptations.
With the described resources already present:

```bash
python code/baselines/run_matched_phrank_phenorank.py \
  --base-dir "$WORKSPACE" --out-root "$NEW_BASELINE_OUTPUT" \
  --sampler published-tier --simulations 1000 \
  --folds 0 1 2 3 4 5 6 7 8 9 --workers 3
```

This uses the published tier sampler, including with-replacement draws within
partially used tiers, with uncapped simulated query lengths. The explicit
`legacy-weighted` option is only for the historical adapted null model.
Do not mix output files from the two procedures under one result label.

Assemble the five completed components into a checksum-bound manifest:

```bash
python reproducibility/build_baseline_manifest.py \
  --data-root "$WORKSPACE/data" \
  --native-phrank "$NATIVE_PHRANK_DIRECT" \
  --native-phrank-disease-max "$NATIVE_PHRANK_DISEASE_MAX" \
  --native-phenorank "$NATIVE_PHENORANK" \
  --upheno-phrank "$UPHENO_PHRANK" \
  --upheno-phenorank "$UPHENO_PHENORANK" \
  --output "$SCORE_MANIFEST"
```

Each component directory contains `fold0` through `fold9`. The full manifest
contains 26 variants: two native HPO Phrank rules, ten UPheno Phrank variants,
native PhenoRank simulated and continuous scores, and twelve UPheno PhenoRank
variants. Phrank adaptations require the author's parentless-term policy;
PhenoRank adaptations require 1,000 uncapped published-tier simulations. The
builder verifies completion records, scientific input identities and output
hashes before writing anything. It rejects partial folds and changed existing
manifests. Output paths are relative to the manifest for relocation with its
data directories.

Supplying fewer component options explicitly builds a partial collection and
records `all_baseline_components: false`; missing requested components always
fail. This supports verification of independently finished baselines while the
others run. Historical `Phrank-HPO` dictionary keys refer to human-only UPheno
adaptations; actual native HPO methods have distinct `Native HPO` labels. The
downstream evaluator checks matrix axes and finite scores and computes all
scoring/cohort sensitivities.

## Evaluate actual scores

Supply an explicit score manifest for new files and model labels, as described
by the evaluator's `--help` and `iter_models_for_fold` implementation. Each
entry identifies candidate/query axes and checksums. Write new reports separately
from the retained verified report:

```bash
python code/analysis/evaluate_stepwise_bias.py \
  --base-dir "$WORKSPACE" --score-manifest "$SCORE_MANIFEST" \
  --out-json "$NEW_REPORT" --ranks-dir "$NEW_RANK_DIRECTORY" --workers 4
```

The evaluator reports raw scores, per-gene calibration, status standardization,
within-status ranks and cohort/weighting sensitivities. These operations answer
different questions; their sequential differences are not independent causal
bias effects. Native baseline rankings remain necessary comparators.

Export the complete baseline report after all five components finish:

```bash
./reproducibility/run_pipeline.sh generate-baseline-tables \
  --report-json "$NEW_REPORT" --score-manifest "$SCORE_MANIFEST" \
  --out-dir "$NEW_BASELINE_TABLE_DIRECTORY"
```

This requires the evaluator's adjacent `.metadata.json` file, all 26 variants,
ten folds, 42 conditions, the matching manifest digest and the paper's cohort
counts. It writes separate Phrank and PhenoRank LaTeX tables, a CSV containing
all metrics in every condition, and an export record binding the input and
output hashes. Native HPO and native 2016 methods are labelled separately from
UPheno adaptations. Partial or historical baseline collections are rejected;
changed existing outputs require a new directory. The exporter checks report
consistency; the manifest builder and evaluator perform the checks of original
score files. It does not rerun those score checks or establish statistical
superiority from a table. The historical `generate-tables` command remains
separate and must not be presented as this final baseline export.

The complete benchmark has 6,571 pairs from 6,059 diseases. The zero-same-gene-
overlap cohort has 4,431 pairs. The combined sensitivity retains 4,265 pairs from
4,017 disease IDs grouped into 4,011 phenotype profiles. Of 210 exact training-
profile-matching pairs overall, 166 survive the preceding restriction and are
excluded at that final step.

Regression/integration tests are documented in [testing](../docs/testing.md).
Passing these tests and regenerating saved tables does not by itself reconstruct
all source data or reproduce full-data training.
