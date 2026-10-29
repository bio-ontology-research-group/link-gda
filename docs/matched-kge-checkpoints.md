# Matched embedding checkpoint archives

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
Production checkpoint publication and numerical inference replay remain
pending completion of training and scoring.

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
verifies 32 completed runs and all 128 fixed validation selections, with the
previous 28 metadata records unchanged. This extends metadata coverage only;
checkpoint-byte verification and numerical inference require their own evidence.

A [fourth disjoint checkpoint audit](../reproducibility/verification/matched-kge-interim-checkpoint-audit-delta-v4.json)
verifies the five additional T1 runs. Coverage is now 32 runs, 128 checkpoints
and 224 files. Their byte hashes, finite tensors, strict model loads, graph
mappings, input identities and fixed selections pass. Numerical inference and
the complete 100-run campaign remain separate requirements.

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

On an allocated GPU, repeat inference for every model, fold and graph:

```bash
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
