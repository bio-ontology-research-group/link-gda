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

The [checkpoint archive workflow](controlled-query-checkpoints.md) exports all
initial and selected model weights with their frozen selection evidence. It
uses the existing archive packer and verifies the packed inventory against the
files checked before export. Production packaging and inference replay follow
completion of the campaign.

An earlier boundary-gate implementation compared two independently executed GPU
training trajectories. Its [state audit](../reproducibility/verification/query-v2-boundary-comparison-audit.json)
found Adam-moment differences already present before interruption, while model
weights agreed within the declared tolerance and saved validation scores and choices
were identical. The direct same-checkpoint boundary gate isolates restoration and
requires exact state equality; it does not widen the numerical tolerance of the
separate two-update interruption/resumption comparison.
