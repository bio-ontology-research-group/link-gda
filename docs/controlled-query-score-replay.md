# Controlled-query saved-score archives

This collection supports a complete CPU replay of the controlled-query statistics
and tables. It contains all 480 initial/selected score and rank bundles, the 240
training-selection records and validation trajectories, both validation matrices
per run, the global freeze, source/protocol manifests, the benchmark files and
the verified reference report. Each campaign receives ordinary benchmark files
at its expected paths, so extraction requires no links to the training machine.

The exporter recomputes the full report before creating its allowlist. It rejects
missing bundles, changed scores or ranks, inconsistent frozen choices, changed
benchmark data and a reference report that differs from the replay. Model
weights, optimizer states and unrelated run-directory files are excluded. Use
the [checkpoint collection](controlled-query-checkpoints.md) to reproduce model
inference; this score collection verifies the subsequent analysis.

The synthetic complete-campaign test exercises packing, extraction into a new
directory, removal of all original inputs, exact report replay and generation of
all three tables. Production publication and numerical replay remain pending
completion of the controlled experiment.

## Create the release

Use the [CPU analysis environment](../reproducibility/README.md). Set `RUNS` to
the completed controlled campaign root and choose new output paths for
`CONTROLLED_REPORT`, `SCORE_PLAN` and `PACKED_SCORES`. Set `RELEASE_URL` to the
intended GitHub release's download URL. Use the same analysis source and pinned
environment for the original report and replay; the report records source hashes.

```bash
python code/analysis/summarize_controlled_query.py \
  --root "$RUNS" --output "$CONTROLLED_REPORT"
python reproducibility/build_controlled_query_scores.py plan \
  --root "$RUNS" --report-json "$CONTROLLED_REPORT" --output "$SCORE_PLAN"
python code/reproduce/artifacts.py pack \
  --plan "$SCORE_PLAN" --output "$PACKED_SCORES" --url-base "$RELEASE_URL"
python reproducibility/build_controlled_query_scores.py verify-packed \
  --plan "$SCORE_PLAN" --manifest "$PACKED_SCORES/release-manifest.json"
```

Archive parts contain at most 1,400 MiB before compression. The final command
checks packed member sizes and hashes against the verified pre-packing inventory.
Before publishing, extract every archive and check all archive/member hashes
with the standard artifact fetcher. Publish the release manifest and archives;
the local export plan contains machine-specific source paths.

## Replay after publication

Download the release manifest to `SCORE_RELEASE_MANIFEST`. Set `WORKSPACE` and
`DOWNLOAD_CACHE` to fresh directories, and choose a new path for `REPLAY_REPORT`.
The reference report's installed path is recorded in the release manifest's
`provenance.reference_report` field. Set `REFERENCE_REPORT` to that path under
`WORKSPACE`.

```bash
python code/reproduce/artifacts.py fetch \
  --manifest "$SCORE_RELEASE_MANIFEST" --root "$WORKSPACE" --cache "$DOWNLOAD_CACHE"
python code/analysis/summarize_controlled_query.py \
  --root "$WORKSPACE/controlled-query" --output "$REPLAY_REPORT"
cmp "$REFERENCE_REPORT" "$REPLAY_REPORT"
python reproducibility/generate_controlled_query_tables.py \
  --report-json "$REPLAY_REPORT" --out-dir "$WORKSPACE/tables"
```

Compare all three generated `.tex` files with the final manuscript's corresponding
tables. A successful report comparison establishes the saved-score calculations,
including all eight prespecified contrasts and bootstrap intervals. It does not
establish fresh training or checkpoint-to-score numerical equivalence.
