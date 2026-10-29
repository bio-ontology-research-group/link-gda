# Original embedding comparison: metrics to tables

The portable table commands reproduce the ten original embedding-comparison
and excluded-gene tables, their paired statistical reports, within-status
summaries, and rank CDF. The calculations are preserved from the manuscript
analysis scripts; input and output locations are explicit arguments.

This stage starts from per-association metric files. The public raw-prediction
and annotation bundle for this analysis, and its complete prediction-to-metrics
driver, remain under preparation. These commands alone do not establish
reproduction from raw scores. Current verification is recorded in
[STATUS.txt](../reproducibility/STATUS.txt).

The input workspace contains:

- `revised-metrics/`: per-fold and per-seed metric files with candidate
  provenance, association identities, and raw/calibrated ranks. It also contains
  the exact wrong-query report used by the supporting table command.
- `revised-main-controls.json`, `revised-excluded-primary-controls.json`, and
  `revised-excluded-secondary-controls.json`: annotation-count control reports.

Use the same workspace for both commands. The first writes aggregate and paired
comparison reports into `revised-metrics/`; both write figures/tables only to
the specified output root. The commands require NumPy, SciPy, and Matplotlib.

```bash
python reproducibility/finalize_results.py \
  --workspace "$EMBEDDING_WORKSPACE" --out-dir "$GENERATED_ASSETS"
python reproducibility/supporting_tables.py \
  --workspace "$EMBEDDING_WORKSPACE" --out-dir "$GENERATED_ASSETS"
```

Pass `--replace` to deliberately regenerate existing tables and figures. The
figure command writes `fig/revised_rank_cdf.pdf` and `.png`; the ten tables are
under `tables/`. Setting `SOURCE_DATE_EPOCH` fixes PDF creation metadata for a
byte-level artifact comparison within the same plotting environment.

Verification against the complete retained metric collection reproduces all
ten manuscript tables exactly, both CDF artifacts exactly, both paired
statistical reports numerically, and the within-status summary exactly. The
[verification record](../reproducibility/verification/legacy-table-port.json)
lists the artifact hashes and the boundary of this check. It does not replace
the planned fresh public raw-score replay.
