# Controlled-query saved-score archives

**Data access, 30 October 2026:** temporary bulk GitHub Release attachments have
been withdrawn. Use the [curated archive restore instructions](frozen-archive.md#use-restored-views-with-the-existing-recipes)
to replace the initial archive download below with an exact restored view.
Institutional bundle verification is in progress; public Zenodo access remains
pending author-managed deposition. See [artifact storage and access](artifact-storage.md).

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

The [production score release](https://github.com/bio-ontology-research-group/link-gda/releases/tag/controlled-query-scores-20261029-v1)
contains 59 archives (73,093,469,516 compressed bytes) and a
[checksum manifest](../reproducibility/artifacts/controlled-query-scores-20261029-v1.json)
covering 2,961 files. All 480 bundles passed the production analysis. Extraction
of every local archive reproduced the full report and all three tables exactly;
publication verified all 60 asset sizes and hashes, the tag commit, and public
manifest access. See the [release verification](../reproducibility/verification/controlled-query-score-release.json).
Fresh unauthenticated download of all 59 public archives and complete CPU replay
have also passed. The replay reproduced all eight primary contrasts, every
48-by-42 bias result and all 6,720 retained rank arrays. The three primary tables
match the manuscript exactly; six detailed bias tables match the native reference
apart from the verified relocated-manifest comment, and the complete metrics CSV
is identical. See the [public download/replay evidence](../reproducibility/verification/controlled-query-public-score-replay.json).
[Full independent checkpoint inference](../reproducibility/verification/controlled-query-full-public-inference.json)
also passes for all 480 bundles, every score cell and the complete downstream
primary and bias analysis.

The [committed primary report](../reproducibility/results/controlled-query-report.json)
is byte-identical to the reference verified in that public replay. For a small
report-to-table run, follow the [quick recipe](reproduction.md#regenerate-the-main-controlled-tables).
The full archive recipe below recomputes the report from predictions.

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

Use `reproducibility/artifacts/controlled-query-scores-20261029-v1.json` as
`SCORE_RELEASE_MANIFEST`, or download the same manifest from the release.
Its SHA-256 is `1d3388c9cdbb0c781125fea2b8470db357b34f1ae409861eb77b7c89caa19013`.
Set `WORKSPACE` and
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

The same extraction supplies every input for the
[48-variant bias sensitivity workflow](controlled-query.md#bias-sensitivities-for-every-controlled-variant).
Set `RUNS` to `$WORKSPACE/controlled-query` and `CONTROLLED_REPORT` to the verified
replayed report. Generate the bias manifest inside `RUNS`, then run the shared
42-condition evaluator and the controlled-rank parity check.
