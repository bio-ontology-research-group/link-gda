# Matched KGE saved-score archives

This collection contains all 400 selected-checkpoint score files, the 100 training
and scoring records, validation histories, global freeze, portable score manifest,
benchmark data, full evaluation report, saved ranks and primary comparison summary.
It supports a complete CPU replay of all 40 model variants across 42 conditions,
the ten prespecified contrasts and both manuscript tables.

Export first checks the full training campaign, including the selected checkpoint
hashes, validation choices and prescribed warm starts. It then reruns the shared
evaluator and requires byte-identical report, metadata, row identities and NumPy
array payloads. ZIP headers and compression can differ between Python builds.
The primary summary must agree exactly after checking and accounting for the ten
rank-container hashes; its regenerated hashes always identify the actual files.
Download and archive-member integrity still require exact file hashes.
The archive contains saved scores and selection evidence; the
[checkpoint collection](matched-kge-checkpoints.md) supplies model weights for
fresh inference. Unrelated run files and optimizer states are excluded.

Synthetic complete-campaign packing and extraction pass. After removal of every
original input, the extracted collection reproduces the complete evaluation,
all saved ranks, primary summary and both tables exactly. The
[complete native analysis](../reproducibility/verification/matched-kge-complete-verification.json)
passes for all 100 runs and 400 score files: all transferred bytes, the full
40-by-42 evaluation, saved ranks, ten primary contrasts and exports agree.
The [primary summary](../reproducibility/results/matched-kge-summary.json) and
[table exports](../reproducibility/results/matched-kge-tables/) retain the actual
production results. Production archive publication and fresh public replay
remain pending.

The [production bias-table renderer](production-bias-tables.md) also exports all
40 variants across the 42 conditions, with separate rows for every checkpoint
choice and both graphs. This complements the primary contrast and design tables.

## Prepare the portable manifest

Before evaluating the production campaign, create its score manifest directly
inside `MATCHED_RUN`:

```bash
export MATCHED_MANIFEST="$MATCHED_RUN/score-manifest.json"
```

Run the manifest builder, evaluator and summary commands in
[KGE execution](kge-execution.md) using this manifest path. Its score paths then
have the form `fold0/original/U1/scores-raw_head.npz`, so archive relocation preserves
the manifest bytes and all downstream provenance hashes. Export rejects a
manifest stored elsewhere. If an earlier evaluation used a different location,
rebuild the manifest and evaluation at the portable location before export.

`PREPARED_DATA` must be a directory named `data` containing the ten folds,
as required by the shared evaluator. Use the same pinned
[CPU analysis environment](../reproducibility/README.md) and analysis source
for production evaluation, export verification and subsequent replay.

## Package the verified scores

Set `GENE_PHENOTYPES` to the frozen INDIGENA annotation table. Use the report,
summary and rank paths from the preceding evaluation. Choose fresh paths for
`SCORE_PLAN`, `EXPORT_VERIFICATION` and `PACKED_SCORES`; set `RELEASE_URL` to the
intended GitHub release download URL. Export verification reruns all 400 score
files and retains its evaluator log under `EXPORT_VERIFICATION`.

```bash
python reproducibility/build_matched_kge_scores.py plan \
  --root "$MATCHED_RUN" --data-root "$PREPARED_DATA" \
  --score-manifest "$MATCHED_MANIFEST" --gene-phenotypes "$GENE_PHENOTYPES" \
  --report-json "$MATCHED_BIAS_REPORT" --summary-json "$MATCHED_SUMMARY" \
  --ranks-dir "$MATCHED_RANKS" --verification-dir "$EXPORT_VERIFICATION" \
  --output "$SCORE_PLAN" --workers 4
python code/reproduce/artifacts.py pack \
  --plan "$SCORE_PLAN" --output "$PACKED_SCORES" --url-base "$RELEASE_URL"
python reproducibility/build_matched_kge_scores.py verify-packed \
  --plan "$SCORE_PLAN" --manifest "$PACKED_SCORES/release-manifest.json"
```

The packed inventory must match the files checked before packing. Before
publication, extract the entire collection with the standard fetcher and verify
all archive/member hashes. Publish the generated release manifest and archives;
the local plan contains machine-specific source paths. Parts contain at most
1,400 MiB before compression.

## Replay from a fresh extraction

Set `SCORE_RELEASE_MANIFEST` to the downloaded release manifest and choose fresh
`WORKSPACE`, `DOWNLOAD_CACHE` and `REPLAY_OUTPUT` paths. The manifest's provenance
records the installed `score_manifest`, `data_root`, `reference_report`,
`reference_summary` and `reference_ranks` paths. Set the corresponding variables
below to those paths under `WORKSPACE`.

```bash
python code/reproduce/artifacts.py fetch \
  --manifest "$SCORE_RELEASE_MANIFEST" --root "$WORKSPACE" --cache "$DOWNLOAD_CACHE"
python reproducibility/build_matched_kge_scores.py replay \
  --score-manifest "$REPLAY_MANIFEST" --data-root "$REPLAY_DATA" \
  --report-json "$REFERENCE_REPORT" --summary-json "$REFERENCE_SUMMARY" \
  --ranks-dir "$REFERENCE_RANKS" --output "$REPLAY_OUTPUT" --workers 4
python reproducibility/generate_matched_kge_tables.py \
  --report-json "$REPLAY_OUTPUT/summary.json" --out-dir "$WORKSPACE/tables"
```

Compare both generated `.tex` files with the final manuscript tables. The only
permitted difference is their first-line source-summary hash comment when verified
rank containers have different ZIP encodings. Successful
replay verifies every saved-score condition and the fixed paired analysis. Fresh
checkpoint inference and fresh training remain separate verification steps.
