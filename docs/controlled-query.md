# Controlled PhenoQuery experiments

The [fixed protocol](../reproducibility/protocols/controlled-query-20261028.md)
specifies four arms, two initializations, three seeds and ten folds. Each run
uses 500 updates and validation-only checkpoint selection with the initial
checkpoint retained as a fallback. Both selected and initial checkpoints are
evaluated after every campaign's selection is frozen.

Inputs are a prepared EQ graph directory with `data/foldN` and `projection/`,
the benchmark workspace, and the pinned ULTRA source tree containing both
`ckpts/ultra_4g.pth` and `ckpts/ultraquery.pth`. The two checkpoint SHA256 values
used for the recorded GPU pilots are:

```
48a046e708adf5632d87c30eacae01f5f51466b2301effdc2cb42358d22854e0  ultra_4g.pth
9b6dc20801c35e7c9ebe65764acb4ed9cb5bffc6b0a0a727bd48935589e9388d  ultraquery.pth
```

Use the CUDA environment described in `environment-ultra.yml`; the executed
pilot environment records PyTorch 2.1.2+cu121, CUDA 12.1, NumPy 1.26.4 and
PyG 2.4.0. Full environment and input-archive publication is tracked separately
in [reproduction status](../reproducibility/STATUS.txt).

Run on an allocated GPU, with project-local scratch:

```bash
python code/baselines/run_controlled_query.py \
  --root RUNS --prepared-root PREPARED_EQ --base WORKSPACE --ultra-root ULTRA \
  --phase plan
python code/baselines/run_controlled_query.py \
  --root RUNS --prepared-root PREPARED_EQ --base WORKSPACE --ultra-root ULTRA \
  --phase pilot
python code/baselines/run_controlled_query.py \
  --root RUNS --prepared-root PREPARED_EQ --base WORKSPACE --ultra-root ULTRA \
  --phase all
```

`plan` writes the immutable protocol/source/configuration records and references
the prepared graph files. `pilot` verifies upstream projection parity, actual
updates and four-update uninterrupted/resumed equivalence under both
initializations. It uses four validation diseases and computes no test scores.
`all` runs the 240 training tasks serially, validates every completion, creates
the global selection record, and scores all initial and selected checkpoints.

For cluster dispatch, use the generated `RUNS/INITIALIZATION-seedN/configs/*.json`
with `run_phenoquery.py --root CHILD --base WORKSPACE --ultra-root ULTRA
--config CONFIG`. Production configurations require both technical gates.
Hold an exclusive lock per configuration, as the serial coordinator does.
After all training tasks finish, run the coordinator with `--phase freeze`,
then `--phase score`. The scoring driver checks the global selection before
loading test outputs. Initial checkpoints use the `_frozen` output suffix.

Source changes require a new campaign root and new pilot gates. Completed
trajectories are verified before reuse; incomplete trajectories restore model,
optimizer, selection and NumPy/PyTorch RNG states. GPU equivalence uses
`rtol=1e-5, atol=1e-6`; selected-step and validation-metric agreement are checked
as well. No test-informed configuration or failed run is silently substituted.

## Validation-boundary resumption

The first production attempt was invalidated before test scoring: interruption
between a saved update and its scheduled validation could skip that validation
on restart. The [invalidation record](../reproducibility/verification/controlled-v1-invalidation.json)
identifies the two affected completed runs. Its outputs must not be used as the
controlled comparison.

The repaired driver replays a pending scheduled validation before further updates
or completion. Selected weights and validation scores are saved in generations;
the resume checkpoint binds the committed generation by checksum. Restart restores
that selection even if an interrupted validation wrote newer public file aliases.
Completion and global freezing reject missing validation steps. Technical GPU gates
now also retain the exact state immediately before final validation, resume, and
require bitwise equality of weights, optimizer and RNG while the pending evaluation
is completed exactly once. A new campaign root and fresh gates are required.

## Verify and summarize the complete experiment

After all 240 training selections are frozen and all 480 initial/selected score
bundles are present:

```bash
python code/analysis/summarize_controlled_query.py \
  --root RUNS --output controlled-query-report.json
python reproducibility/generate_controlled_query_tables.py \
  --report-json controlled-query-report.json --out-dir controlled-query-tables
```

The report verifies the complete design, global freeze, benchmark identities,
validation trajectories and checkpoint choices. It recomputes ranks and metrics
from every score matrix, checks the retained ranks and validation references, and
requires step-zero selections to reproduce their initial ranks. Missing or
inconsistent runs fail the report instead of reducing its denominator.

For each prespecified contrast, it averages each association's Hits@10 indicator
across the three seeds before taking the paired difference. The bootstrap samples
whole identical-phenotype-profile clusters across disease IDs, retaining association
weights within clusters. It uses 10,000 replicates, seed 20261028, and 99.375%
intervals for the eight-contrast family. Each seed's result and the sample standard
deviation across seeds remain separate from the bootstrap intervals. Raw and
calibrated MR, MRR, Hits@1 and Hits@10 are retained for every arm and checkpoint,
along with training-status and overlap strata. The report's primary estimator is
association-pooled; it does not substitute the historical tables' equal-fold means.

The generated main comparison, contrast table and full supplementary tables contain
only values from the supplied verified report. No synthetic performance results are
bundled as manuscript results.

## Parallel inference after the complete training freeze

The operational launcher `scripts/run_controlled_query_scoring.py` assigns one
GPU worker to each initialization/seed campaign. Together, the six workers run
the same 60 fold commands and generate the same 480 bundles as the original
serial coordinator. It uses the frozen source's command builder and inference
driver; training, checkpoint choices and readouts remain those of the protocol.

Schedule all six workers only after all training jobs succeed. The first worker
obtains an exclusive freeze lock and invokes the original coordinator's `freeze`
phase. Other workers verify and reuse that complete freeze. Per-campaign and
per-fold locks prevent duplicate work, and children inherit the locks if their
supervisor exits. A shared lock excludes simultaneous use of the deployed serial
scoring wrapper. Changes to any campaign's frozen selection stop scoring.

On each allocated GPU, run one initialization/seed combination:

```bash
python scripts/run_controlled_query_scoring.py \
  --source-root "$TRAINING_SOURCE" --root "$RUNS" --prepared-root "$PREPARED_EQ" \
  --base "$BENCHMARK_WORKSPACE" --ultra-root "$ULTRA" \
  --initialization ultra_4g --seed 0
```

Use `ultra_4g` and `ultraquery`, each with seeds 0, 1 and 2. The deployed ORIX
array wrapper is `scripts/orix-controlled-query-score.sbatch`; its three seed
tasks each process both initializations in sequence to respect the account
submission limit. All tasks need successful-completion dependencies on all six
training jobs. The wrapper checks
the scientific source, upstream checkpoints and launcher hashes. A resumed worker
uses the original scorer's verification of existing output bundles. Full numerical
replay is still required after inference completes.

## Bias sensitivities for every controlled variant

The fixed protocol also requires separate reference deduplication, co-causal
filtering, equal-profile weighting and training-profile exclusion analyses. The
shared evaluator supplies these operations, with their separate populations,
alongside the cumulative S0--S4 conditions. Evaluate all 48 combinations of
initialization, seed, arm and checkpoint type; keep each seed visible.

After the complete controlled report exists, choose new output paths for
`CONTROLLED_BIAS_MANIFEST`, `CONTROLLED_BIAS_REPORT`, `CONTROLLED_BIAS_RANKS`
and `CONTROLLED_BIAS_VERIFICATION`. Store the manifest inside `RUNS` to preserve
relative paths when replaying the public saved-score collection.

```bash
python reproducibility/build_controlled_bias_manifest.py build \
  --root "$RUNS" --controlled-report "$CONTROLLED_REPORT" \
  --output "$CONTROLLED_BIAS_MANIFEST"
python code/analysis/evaluate_stepwise_bias.py \
  --base-dir "$RUNS/ultra_4g-seed0" --score-manifest "$CONTROLLED_BIAS_MANIFEST" \
  --out-json "$CONTROLLED_BIAS_REPORT" --ranks-dir "$CONTROLLED_BIAS_RANKS" --workers 4
python reproducibility/build_controlled_bias_manifest.py verify \
  --score-manifest "$CONTROLLED_BIAS_MANIFEST" --controlled-report "$CONTROLLED_REPORT" \
  --bias-report "$CONTROLLED_BIAS_REPORT" --ranks-dir "$CONTROLLED_BIAS_RANKS" \
  --output "$CONTROLLED_BIAS_VERIFICATION"
```

After verification, use the [production bias-table renderer](production-bias-tables.md)
to export all 48 variants and 42 conditions. Its tables preserve each seed and
both checkpoint types; the CSV retains exact values and population units.

Manifest construction replays the complete controlled report. Verification binds
the benchmark and every score to that report, checks phenotype/association row
identities, and requires all 960 raw/calibrated rank arrays and corresponding
aggregates to agree exactly across the two evaluation paths. This also checks
numerical ties before interpreting additional sensitivities. The 42 conditions
are descriptive analyses; they do not expand the family of eight primary contrasts
or choose a new checkpoint. Record every condition's denominator. The sensitivity
reports can be regenerated from the saved-score archive without model weights;
production results and their manuscript tables remain pending.

The [saved-score archive workflow](controlled-query-score-replay.md) packages all
480 score/rank bundles with benchmark data and selection evidence, and verifies
the complete report before export. A fresh extraction can regenerate every
controlled statistic and all three tables without a GPU.

The [checkpoint archive workflow](controlled-query-checkpoints.md) exports all
initial and selected model weights with their frozen selection evidence. It
uses the existing archive packer and verifies the packed inventory against the
files checked before export. The complete production checkpoint collection is
now public, with byte, validation-metric, packing and local extraction checks.
Fresh public archive download and independent inference remain separate gates.

An earlier boundary-gate implementation compared two independently executed GPU
training trajectories. Its [state audit](../reproducibility/verification/query-v2-boundary-comparison-audit.json)
found Adam-moment differences already present before interruption, while model
weights agreed within the declared tolerance and saved validation scores and choices
were identical. The direct same-checkpoint boundary gate isolates restoration and
requires exact state equality; it does not widen the numerical tolerance of the
separate two-update interruption/resumption comparison.
