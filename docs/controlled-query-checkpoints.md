# Frozen controlled-query checkpoint archives

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

The exporter has passed synthetic complete-campaign packing, extraction and
relocation checks. Production export and checkpoint-to-prediction replay are
pending completion of the controlled campaign; no trained-model release is
claimed by this document.

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

The archives install under `controlled-query/`. Fetch them into a new workspace
with the standard artifact fetcher. Install the separately versioned prepared
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
