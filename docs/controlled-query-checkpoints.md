# Frozen controlled-query checkpoint archives

**Data access, 30 October 2026:** temporary bulk GitHub Release attachments have
been withdrawn. The archive commands below require retained server copies until
a replacement public archive is published. See [artifact storage and access](artifact-storage.md).

The checkpoint exporter preserves all 240 production selections, their 480
initial/selected model files, both validation-score matrices per run, complete
validation trajectories, 500-update training logs, configurations, source and
protocol manifests, the global selection, and the evidence files bound by the
technical gates. It requires the completed global freeze. Missing selections,
changed weights, incomplete validation schedules, missing training updates and
changed source or gate evidence stop export.

These archives support inference from the frozen models. Prepared graphs,
upstream ULTRA source and initial checkpoints, the execution environment, and
test-score bundles are separate dependencies. Production optimizer/resume
generations are excluded. The [training workflow](controlled-query.md) describes
fresh training and its resumption mechanism.

The [production checkpoint release](https://github.com/bio-ontology-research-group/link-gda/releases/tag/controlled-query-checkpoints-20261029-v1)
contains all 2,220 files in 17 archives (23,779,324,752 compressed bytes).
[Complete verification](../reproducibility/verification/controlled-query-checkpoint-release.json)
checks every file and recalculates every retained validation metric exactly
from all 480 matrices. Packing and extraction preserve the scientific inventory
after relocation.
The public tag, asset sizes/checksums and manifest also pass verification.
A [fresh unauthenticated public download](../reproducibility/verification/controlled-query-checkpoint-public-replay.json)
also verifies all 17 archives and 2,220 installed files. Rebuilding the checkpoint
inventory in the new workspace gives identical scientific provenance. This run
reuses the separately verified prepared inputs and checks the relocated inference
layout. The complete independent checkpoint-to-prediction and downstream replay
now passes, as described below. Comparative claims retain their specified
experimental design and evaluation population.

An [independent H200 validation replay](../reproducibility/verification/controlled-query-h200-validation-inference.json)
has passed for seed 0, fold 0, both initializations and all four query designs,
using the initial and validation-selected checkpoints. All 16 matrices
(118,810,000 score values), raw/calibrated ranks and saved prediction archives
are exactly identical to the original validation outputs. The replay used the
original GPU architecture and environment, with native checkpoint bytes bound
to the public release manifest. The public download was verified separately.
This earlier check covers validation inference in the specified fold. The full
public-checkpoint test replay has subsequently passed on H200 GPUs.

After the complete 240-run global freeze is validated, checkpoint packaging
can proceed while production test scoring runs: the checkpoint exporter does
not require test outputs. Use a separate extraction workspace for independent
inference and compare it with production scores when both are available.
Packaging alone does not establish numerical equivalence.

## Complete public-checkpoint inference replay

The [full verification receipt](../reproducibility/verification/controlled-query-full-public-inference.json)
and [retained operation](../reproducibility/verification/controlled-query-full-public-inference-operation/)
verify all 480 test bundles from freshly downloaded public checkpoints and
prepared inputs. Six H200 tasks cover both initializations, three seeds, ten
folds and every query design, preserving eight allocated CPUs and eight threads.
All 3,962,586,000 score cells match the reference exactly, with zero changed raw
or calibrated ranks. The independently regenerated primary report is byte-identical,
including all eight planned bootstrap contrasts.

The downstream check also reproduces all 48 variants across 42 bias conditions,
6,720 rank arrays and ten current manuscript tables. IBEX verification job
54815338 completed with exit zero. These checks establish frozen-model inference
and analysis agreement; the benchmark and clinical-validation limitations in the
paper still apply.

## Build and verify an archive collection

Use the CPU analysis environment from the [reproduction instructions](../reproducibility/README.md).
Set `RUNS` to the frozen campaign root, `CHECKPOINT_PLAN` to a new JSON path,
`PACKED_CHECKPOINTS` to an output directory, and `RELEASE_URL` to the intended
GitHub release download URL. The plan uses explicit file mappings and splits
them into parts of at most 1,400 MiB before compression.

Set `TRAINING_SOURCE` to the frozen training checkout, independently of the
analysis checkout running this exporter. The v3 production campaign uses commit
`7a3b0196d86dcf68e6dcd63f86cad1c4f07cf3b4`. The exporter compares that checkout's
actual source bytes and protocol with every campaign's retained source manifest;
later analysis or historical-driver fixes need not change the training source.

```bash
python reproducibility/build_controlled_query_artifacts.py plan \
  --root "$RUNS" --training-source "$TRAINING_SOURCE" --output "$CHECKPOINT_PLAN"
python code/reproduce/artifacts.py pack \
  --plan "$CHECKPOINT_PLAN" --output "$PACKED_CHECKPOINTS" \
  --url-base "$RELEASE_URL"
python reproducibility/build_controlled_query_artifacts.py verify-packed \
  --plan "$CHECKPOINT_PLAN" \
  --manifest "$PACKED_CHECKPOINTS/release-manifest.json"
```

The last command compares every packed member's size and checksum with the
inventory verified before packing. This detects an input change between
planning and packing. Before publication, also extract all parts with the
standard artifact fetcher and verify their archive and member checksums. The
local plan contains filesystem paths; publish the generated release manifest,
which contains relative archive member paths and their checksums.

## Recompute predictions after extraction

The archives install under `controlled-query/`. Set `WORKSPACE` and
`DOWNLOAD_CACHE` to fresh directories and fetch the
[versioned manifest](../reproducibility/artifacts/controlled-query-checkpoints-20261029-v1.json):

```bash
python code/reproduce/artifacts.py fetch \
  --manifest reproducibility/artifacts/controlled-query-checkpoints-20261029-v1.json \
  --root "$WORKSPACE" --cache "$DOWNLOAD_CACHE"
```

The manifest SHA256 is
`880e0dfd563f5f6f7a783d2ee669e568f56c969c15998db56ac4d502546a069c`.
Install the separately versioned prepared
EQ inputs and benchmark, and reconstruct the recorded ULTRA environment and
source as described in [frozen inputs](frozen-inputs.md).

On a GPU allocation, invoke the original training source's coordinator:

```bash
python "$TRAINING_SOURCE/code/baselines/run_controlled_query.py" \
  --root "$WORKSPACE/controlled-query" --prepared-root "$PREPARED_EQ" \
  --base "$BENCHMARK_WORKSPACE" --ultra-root "$ULTRA" --phase score
```

`TRAINING_SOURCE` must match each archived `source.sha256`. The coordinator
creates links to the prepared inputs, verifies the configurations and technical
gates, revalidates the frozen selections, and generates all 480 initial/selected
score bundles. No test outputs are included in the checkpoint archive, so a
fresh extraction exercises model inference. After inference, use the complete
controlled-query summary and compare its results and ranks with the separately
released score collection. Production numerical equivalence remains a required
verification step when the campaign finishes.

Compare the regenerated collection with the extracted production score collection:

```bash
python reproducibility/compare_controlled_query_inference.py \
  --reference-root "$PRODUCTION_SCORES/controlled-query" \
  --replay-root "$WORKSPACE/controlled-query" \
  --output "$WORKSPACE/inference-comparison.json"
```

The comparator requires the same complete global selection and verifies all 480
bundles, benchmark identities, validation evidence, saved metrics and rank arrays.
Prediction values must agree within `1e-6 + 1e-5 * abs(production_score)`; raw and
per-gene-calibrated association ranks must agree exactly. Numerical differences
produce a `requires_review` report and a nonzero exit code, even when score
differences fall inside tolerance but ranks change. Corrupt or incomplete input
collections stop verification. The report does not establish execution provenance
or agreement of the separate downstream bias-condition analyses; retain those
verification records as well.

Recompute the complete primary report from the independently generated scores:

```bash
python code/analysis/summarize_controlled_query.py \
  --root "$WORKSPACE/controlled-query" \
  --output "$WORKSPACE/replayed-controlled-report.json"
python reproducibility/compare_controlled_query_reports.py \
  --reference-report "$PRODUCTION_SCORES/controlled-query-reference/controlled-report.json" \
  --reference-sha256 51630e6e415b917d8a60301075081ceae83ad3a837e3513d1b2862d6fef90adf \
  --replay-report "$WORKSPACE/replayed-controlled-report.json" \
  --inference-comparison "$WORKSPACE/inference-comparison.json" \
  --output "$WORKSPACE/primary-report-comparison.json"
```

The reference digest above identifies the production report in the versioned
[score collection](controlled-query-score-replay.md). Its verified reference
digest must be retained independently of the report being compared. The report
comparator checks all primary statistics, per-seed and stratum results,
selection records, and all eight unrounded bootstrap contrasts and settings.
It binds each report's 480 input records to the successful score comparison.
Only the explicitly verified score, rank-container, and scoring-completion
digests may differ; training evidence and benchmark inputs must agree.
Different numerical results are retained in a `requires_review` result with a
nonzero exit code. This comparison does not rerun inference or replace the
[full bias-condition comparison](production-bias-tables.md).
