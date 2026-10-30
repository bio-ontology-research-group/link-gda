# Matched embedding checkpoint archives

**Data access, 30 October 2026:** temporary bulk GitHub Release attachments have
been withdrawn. Use the [curated archive restore instructions](frozen-archive.md#use-restored-views-with-the-existing-recipes)
to replace the initial archive download below with an exact restored view.
The institutional bundle has passed restoration and destination checksum checks;
public Zenodo access remains pending author-managed deposition. See [artifact storage and access](artifact-storage.md).

The exporter retains all 400 selected model files from the 100 matched training
trajectories, together with their training identities, completion records,
complete validation schedules and global freeze. It includes all four selection
rules for every model, graph and fold. The original U1 checkpoints needed by
ConvKB and P-pool remain in the collection.

Export requires the complete verified score manifest from
[KGE execution](kge-execution.md). The builder rechecks that manifest against
the actual scores, checkpoints, prepared inputs, annotation resource, fixed
training budgets, validation selections and prescribed warm starts. It rejects
partial campaigns and source/environment inconsistencies before producing a
plan. Test scores, score-completion markers and optimizer generations are
excluded from the checkpoint archives, so an extracted collection can run
inference into new score files.

Synthetic full-campaign packing and extraction pass, and the extracted files
pass the original trainer's freeze and selection validators after relocation.
The [complete production audit](../reproducibility/verification/matched-kge-complete-verification.json)
verifies all 400 selected checkpoints from 100 training runs. Byte identity,
finite tensors, strict model loads, graph mappings, warm starts and fixed
selection rules pass. The complete saved-score analysis also passes for all
400 score files. A separate [H200 validation replay](../reproducibility/verification/matched-kge-h200-validation-inference.json)
reproduces all 144 raw/calibrated metrics for twelve fixed checkpoints: fold-zero
U1 on both graphs and T1 on the original graph, each with all four selection
choices. It uses audited native weights and a fresh public source checkout,
with eight allocated CPUs and eight software threads. The corresponding
[A100 validation replay](../reproducibility/verification/matched-kge-a100-validation-inference.json)
also reproduces all 144 metrics. Its twelve retained validation score files were
rehashed and match the score hashes recorded by the H200 replay. Original
training-time validation matrices were not retained, so agreement to training
is checked through the recorded aggregate metrics. The checkpoint archive
passed its public asset, tag and manifest checks
before bulk-asset withdrawal; see the [publication proof](../reproducibility/verification/matched-kge-checkpoint-release.json).
Fresh inference from retained, checksum-bound checkpoints now reproduces all
400 score files byte for byte, covering 30,051,300,000 test and auxiliary
training score cells, with zero changed raw or calibrated ranks. See the
[full score comparison](../reproducibility/verification/matched-kge-retained-inference-score-comparison.json).
The [complete downstream verification](../reproducibility/verification/matched-kge-full-retained-inference.json)
also passes all ten contrasts, 40-by-42 bias conditions, 5,600 rank arrays,
all seven current manuscript tables and the metrics export.

## Historical release layout

The [checkpoint release](https://github.com/bio-ontology-research-group/link-gda/releases/tag/matched-kge-checkpoints-20261029-v1)
contained 161 archives plus the [release manifest](https://github.com/bio-ontology-research-group/link-gda/releases/download/matched-kge-checkpoints-20261029-v1/release-manifest.json),
about 155 GB in total. The manifest SHA-256 is
`52d102a97ece1b8de141d9c5fa4e2c557ebab1e405f45eec5aabed8d462a31f8`.
It installs 701 files under `matched-kge/`: 400 selected weights and 301
metadata records. Benchmark graphs and the INDIGENA annotation supplement are
separate [frozen inputs](frozen-inputs.md). The score and checkpoint releases
share identical hashes for all 301 common metadata files and the same global
training freeze. The first fresh download rejected part002 at its checksum
check. The partial public download recovery was stopped when the authors withdrew
the bulk assets; the [recovery record](../reproducibility/verification/matched-kge-public-checkpoint-recovery.json)
preserves that historical state. Complete public checkpoint access now depends
on the later Zenodo deposition. Independent inference used the retained
checkpoint collection with all input identities verified; it does not establish
a complete public checkpoint download.

## Earlier partial audits

An [interim artifact audit](../reproducibility/verification/matched-kge-interim-checkpoint-audit.json)
also verifies actual checkpoint bytes from 20 completed runs (19 U1, one T1).
All 80 selected weights pass hash, finite-tensor and strict model-loading checks;
graph mappings, inputs, training identities and validation choices agree.
This partial audit does not establish weight-to-prediction equivalence or
completion of the full campaign.

A [second, disjoint audit](../reproducibility/verification/matched-kge-interim-checkpoint-audit-delta.json)
extends this coverage to 25 runs and 100 selected checkpoints: all 20 U1 runs
and five T1 runs. The same checks pass; numerical inference remains separate.

A [third disjoint audit](../reproducibility/verification/matched-kge-interim-checkpoint-audit-delta-v3.json)
adds two T1 runs, bringing coverage to 27 runs and 108 selected checkpoints.
The same checks pass. The proof preserves an initial verifier count error and
its correction; scientific inputs and model files were unchanged.

A separate [metadata audit](../reproducibility/verification/matched-kge-training-metadata-interim.json)
verifies 40 completed runs and all 160 fixed validation selections, with the
previous 32 metadata records unchanged. This verifies metadata only;
checkpoint-byte verification and numerical inference require their own evidence.

A [fourth disjoint checkpoint audit](../reproducibility/verification/matched-kge-interim-checkpoint-audit-delta-v4.json)
verifies the five additional T1 runs. Coverage is now 32 runs, 128 checkpoints
and 224 files. Their byte hashes, finite tensors, strict model loads, graph
mappings, input identities and fixed selections pass. Numerical inference and
the complete 100-run campaign remain separate requirements.

A [fifth disjoint checkpoint audit](../reproducibility/verification/matched-kge-interim-checkpoint-audit-delta-v5.json)
adds the remaining eight T1 runs. All 20 U1 and 20 T1 runs now pass the same
checks: 160 selected checkpoints and 280 files. Their metadata agrees with
the 40-run metadata audit above. These checks do not establish inference
equivalence or completion of the remaining three model families.

## Package the verified campaign

Use the CPU analysis environment. Set `MATCHED_RUN` to the completed campaign
root containing `FROZEN.json`, `PREPARED_DATA` to the directory containing its
ten prepared folds, and `GENE_PHENOTYPES` to the frozen INDIGENA annotation
table. `MATCHED_SCORE_MANIFEST` is the output of
`reproducibility/build_matched_kge_manifest.py`. Choose new paths for
`CHECKPOINT_PLAN` and `PACKED_CHECKPOINTS`, and set `RELEASE_URL` to the intended
release download URL.

```bash
python reproducibility/build_matched_kge_artifacts.py plan \
  --root "$MATCHED_RUN" --data-root "$PREPARED_DATA" \
  --score-manifest "$MATCHED_SCORE_MANIFEST" \
  --gene-phenotypes "$GENE_PHENOTYPES" --output "$CHECKPOINT_PLAN"
python code/reproduce/artifacts.py pack \
  --plan "$CHECKPOINT_PLAN" --output "$PACKED_CHECKPOINTS" \
  --url-base "$RELEASE_URL"
python reproducibility/build_matched_kge_artifacts.py verify-packed \
  --plan "$CHECKPOINT_PLAN" \
  --manifest "$PACKED_CHECKPOINTS/release-manifest.json"
```

The archive parts contain at most 1,400 MiB before compression. The final
command checks every packed member against the inventory verified before
packing. Before publication, extract the entire collection with the standard
artifact fetcher and check all archive/member hashes. Publish the generated
release manifest; the local plan contains machine-specific source paths.

## Replay the frozen models

The archives install under `matched-kge/`. Fetch them into a new workspace and
install the separately released benchmark and INDIGENA annotation inputs.
Use the [recorded training environment](../reproducibility/environments/matched-kge/README.md)
and the original source checkout matching
the `source` hashes in every archived `identity.json`.

Use the frozen training/scoring checkout
`063efc263446b4d119d3dbf4b97edb8709859f1e`. The release provenance stores its
source-file map alongside the Torch/PyKEEN versions and protocol hash; each
run's `identity.json` records these fields separately.

On one allocated GPU with eight allocated CPUs, preserve the measured thread
setting and repeat inference for every model, fold and graph:

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for model in U1 T1 INDIGENA CONVKB PPOOL; do
  for fold in {0..9}; do
    for graph in original nobridge; do
      python "$TRAINING_SOURCE/code/training/run_matched_kge.py" --phase score \
        --root "$WORKSPACE/matched-kge" --data-root "$PREPARED_DATA" \
        --fold "$fold" --graph "$graph" --model "$model" \
        --gene-phenotypes "$GENE_PHENOTYPES"
    done
  done
done
```

The trainer verifies the archived identities, all selected weights and the
global freeze before inference. It produces all 400 score files. Rebuild their
manifest and run the shared evaluator and matched summary as described in
[KGE execution](kge-execution.md), then compare with the separately released
score results. Production numerical equivalence remains a required check.

Compare all regenerated predictions with the complete production collection
using the CPU analysis environment:

```bash
python reproducibility/compare_matched_kge_inference.py \
  --reference-root "$PRODUCTION_MATCHED_RUN" \
  --replay-root "$WORKSPACE/matched-kge" \
  --reference-data "$PRODUCTION_PREPARED_DATA" --replay-data "$PREPARED_DATA" \
  --reference-annotations "$PRODUCTION_GENE_PHENOTYPES" \
  --replay-annotations "$GENE_PHENOTYPES" \
  --output "$WORKSPACE/inference-comparison.json"
```

Both collections must include all 100 frozen training records, 400 checkpoints
and 400 score files. The comparator revalidates the fixed selections, training
identities, benchmark inputs and warm starts. It compares every test and auxiliary
training prediction using `atol=1e-6, rtol=1e-5`, relative to production, and
requires identical prediction dtypes and exact raw and per-gene-calibrated
test-association ranks. Numerical differences requiring review produce a report
and a nonzero exit status; incomplete or inconsistent inputs stop verification.
This comparison does not establish that inference was independently executed:
retain the source, environment, hardware and execution records separately.
Replay all 42 bias conditions as well, following the saved-score instructions.

The comparator's [16 focused tests](../reproducibility/verification/matched-kge-inference-comparison.json)
cover the complete synthetic collection, acceptable score differences, raw and
calibrated rank drift, auxiliary training prediction drift, and invalid or shared
inputs. The production score comparison now passes for all 400 fresh files;
the complete downstream analysis also passes, as documented above.

After regenerating the complete bias evaluation and matched primary summary,
run the [all-bias comparison](production-bias-tables.md). Then compare the two
primary summaries, binding each to its own manifest, report and verified ranks:

```bash
python reproducibility/compare_matched_kge_summaries.py \
  --reference-summary "$REFERENCE_SUMMARY" --reference-sha256 "$REFERENCE_SUMMARY_SHA256" \
  --replay-summary "$REPLAY_SUMMARY" \
  --reference-manifest "$REFERENCE_SCORE_MANIFEST" --replay-manifest "$REPLAY_SCORE_MANIFEST" \
  --reference-report "$REFERENCE_BIAS_REPORT" --replay-report "$REPLAY_BIAS_REPORT" \
  --bias-comparison "$BIAS_COMPARISON" \
  --output "$WORKSPACE/primary-summary-comparison.json"
```

Take `REFERENCE_SUMMARY_SHA256` from the retained verified production analysis,
independently of the file under comparison. The comparator requires a successful
full matched bias comparison, rechecks its artifact digests, and binds each
summary's 100 primary score digests and ten rank/row digests to that evidence.
Only these verified provenance differences may vary. All other fields, including
the ten unrounded bootstrap contrasts, primary metrics, selection epochs, graph
sizes and training settings, must agree exactly. Scientific differences produce
a retained `requires_review` result and nonzero exit status. Finally regenerate
and compare both primary tables, allowing only their verified summary-digest
comments to differ. Keep the independent inference execution and score-comparison
records as separate evidence.
