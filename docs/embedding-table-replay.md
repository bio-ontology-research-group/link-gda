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

The raw-score and diagnostic commands are now available. Their artifact release
is still being prepared; these commands assume the checksum-bound input layout
already exists. `INPUT_ROOT` contains `legacy-embedding/`. The committed manifest
enumerates 314 metric outputs and all prediction, candidate, training,
reference-statistic and sidecar hashes. The runner rejects changed inputs and
writes a completion record only after every requested entry succeeds.

```bash
python reproducibility/replay_embedding_metrics.py \
  --manifest reproducibility/manifests/embedding-metrics-20261029.json \
  --input-root "$INPUT_ROOT" --workspace "$EMBEDDING_WORKSPACE"
LEGACY_INPUTS="$INPUT_ROOT/legacy-embedding"
python reproducibility/revised_controls.py \
  --data "$LEGACY_INPUTS/corrected-inputs" \
  --folds "$LEGACY_INPUTS/rebuilt-main/data/folds" \
  --pool "$LEGACY_INPUTS/expanded-phenotype-candidates.txt" \
  --out "$EMBEDDING_WORKSPACE/revised-main-controls.json"
python reproducibility/revised_controls.py \
  --data "$LEGACY_INPUTS/corrected-inputs" \
  --folds "$LEGACY_INPUTS/corrected-excluded/data/folds" \
  --pool "$LEGACY_INPUTS/corrected-inputs/function-no-mgi-candidates.txt" \
  --out "$EMBEDDING_WORKSPACE/revised-excluded-primary-controls.json"
python reproducibility/revised_controls.py \
  --data "$LEGACY_INPUTS/corrected-inputs" \
  --folds "$LEGACY_INPUTS/corrected-excluded/data/folds" \
  --pool "$LEGACY_INPUTS/corrected-inputs/function-candidates.txt" \
  --out "$EMBEDDING_WORKSPACE/revised-excluded-secondary-controls.json"
python reproducibility/query_permutation.py --workspace "$EMBEDDING_WORKSPACE" --workers 2
python reproducibility/annotation_audit.py \
  --input-root "$LEGACY_INPUTS" --workspace "$EMBEDDING_WORKSPACE"
python reproducibility/generate_prepared_input_table.py \
  --data-dir "$LEGACY_INPUTS/corrected-inputs" \
  --out "$GENERATED_ASSETS/tables/prepared-inputs.tex"
```

Then run both table commands above. The wrong-query calculation reads the new
metric provenance to locate its score matrices and checks that recalculated
actual ranks agree before computing wrong-query expectations. The annotation
controls reproduce all three retained reports exactly apart from the explicit
pool location. The excluded controls require `corrected-excluded` (425 pairs),
not the older `rebuilt-excluded` layout (409 pairs).

For parallel raw-metric replay, launch disjoint `--shard I --shards N` commands
and require every shard completion record before producing downstream results.
Existing metric outputs require explicit `--replace`; a failed replacement
removes its old completion marker. The full raw-score replay remains a separate
verification gate from the completed metrics-to-table replay.

The prepared-input count table is generated separately from the five retained
annotation-pair files. All five counts and the complete LaTeX table reproduce
exactly; its JSON sidecar records input hashes and distinguishes prepared pairs
from fresh provider downloads. See [the verification record](../reproducibility/verification/prepared-input-table.json).
