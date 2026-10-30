# Bias tables for the production campaigns

The renderer accepts the complete controlled-query or matched-KGE bias report
and its score manifest. First run the corresponding score/rank verification in
[controlled queries](controlled-query.md#bias-sensitivities-for-every-controlled-variant)
or [matched KGE replay](matched-kge-score-replay.md). Table generation checks
report provenance and experimental coverage; it does not repeat model inference
or certify the score-to-metric calculations by itself.

```bash
python reproducibility/generate_production_bias_tables.py \
  --report-json "$CONTROLLED_BIAS_REPORT" \
  --score-manifest "$CONTROLLED_BIAS_MANIFEST" --out-dir "$QUERY_BIAS_TABLES"

python reproducibility/generate_production_bias_tables.py \
  --report-json "$MATCHED_BIAS_REPORT" \
  --score-manifest "$MATCHED_MANIFEST" --out-dir "$MATCHED_BIAS_TABLES"
```

The controlled collection supplies six LaTeX tables, one per initialization and
seed. Each retains all four arms and both initial/selected checkpoints. The
matched collection supplies five tables, one per family, retaining both graphs
and all four validation choices. An asterisk identifies the prespecified primary
choice; it does not identify the best test result. Every selected model receives
the same scoring/cohort operations, including scoring modes other than the one
used to select that checkpoint.

Each table separates cumulative scoring/cohort changes from individual changes
relative to S1. Population counts are checked against the fixed benchmark:
association pairs and equally weighted profiles are distinct units. No seed
averaging, best-seed selection, new significance tests or causal decomposition
is performed. The CSV retains every model, all 42 conditions, exact values,
experimental identities and population units. Its `med` field is the evaluator's
median rank; under equal-profile weighting it is the median of profile mean ranks.

`controlled-bias-export.json` or `matched-bias-export.json` records hashes of the
report, metadata, manifest, generator and generated files. Missing variants,
folds or conditions, incorrect cohort sizes, non-finite values, and changed
manifest/evaluator provenance stop rendering. Existing differing outputs are
not overwritten. Repeating the command on unchanged inputs is deterministic.

The renderer is tested with explicitly synthetic performance values using the
real benchmark schema and populations. These fixtures are not manuscript
results. A [fresh public-checkout replay](../reproducibility/verification/production-bias-tables-fresh-public-replay.json)
in the locked IBEX CPU analysis environment also reproduces all eleven tables,
both CSVs and both export records byte for byte, including a repeated export.
Production tables are included in the manuscript snapshot. The
[controlled-query public replay](../reproducibility/verification/controlled-query-public-score-replay.json)
verifies all six query panels and the metric CSV. The
[complete matched analysis](../reproducibility/verification/matched-kge-complete-verification.json)
verifies all five matched panels and their exports from the native saved scores.

## Compare analyses after independent inference

After the prediction comparison passes, run the shared evaluator on the
regenerated scores. `REPLAY_BENCHMARK` is the parent directory containing the
prepared `data/` directory. Use separate report and rank paths for the verified
reference and the new evaluation.

```bash
python code/analysis/evaluate_stepwise_bias.py \
  --base-dir "$REPLAY_BENCHMARK" --score-manifest "$REPLAY_SCORE_MANIFEST" \
  --out-json "$REPLAY_BIAS_REPORT" --ranks-dir "$REPLAY_BIAS_RANKS" --workers 4
python reproducibility/compare_production_bias_replays.py \
  --reference-manifest "$REFERENCE_SCORE_MANIFEST" \
  --replay-manifest "$REPLAY_SCORE_MANIFEST" \
  --reference-report "$REFERENCE_BIAS_REPORT" --replay-report "$REPLAY_BIAS_REPORT" \
  --reference-ranks "$REFERENCE_BIAS_RANKS" --replay-ranks "$REPLAY_BIAS_RANKS" \
  --output "$BIAS_COMPARISON"
```

The comparison requires the complete 48-variant controlled-query or 40-variant
matched-KGE design, all ten folds, all 42 conditions and the fixed benchmark
populations. Each report's metadata must bind its own score manifest and the
current shared evaluator. Distinct prediction hashes may change the manifest
hash; all other evaluation metadata must agree.

Every metric and all fourteen saved rank arrays per model and fold are compared.
Row identities and order must be byte-identical; rank dtypes, shapes and values
must agree exactly. ZIP compression and member order may differ. Missing arrays
in both collections are rejected, as are shared rank files. Metric or rank drift
produces a `requires_review` report and nonzero exit status. Invalid coverage or
provenance stops verification. Existing output reports are never overwritten.

The [32 focused tests](../reproducibility/verification/production-bias-comparison.json)
use explicitly synthetic values with the production schema. They cover both
complete designs, differing manifest hashes, rank recompression, non-primary
condition drift, incomplete coverage, invalid provenance and retained error
reports. This comparison establishes agreement with a previously verified
reference; it does not replace independent execution records, weight and score
verification, or rerunning the primary bootstrap analyses and table exports.
