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
results. Production rendering and manuscript integration require the completed,
verified campaign reports.
