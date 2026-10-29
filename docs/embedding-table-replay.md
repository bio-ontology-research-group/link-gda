# Original embedding comparison: metrics to tables

The portable table commands reproduce the ten original embedding-comparison
and excluded-gene tables, their paired statistical reports, within-status
summaries, and rank CDF. The calculations are preserved from the manuscript
analysis scripts; input and output locations are explicit arguments.

The first commands below start from per-association metric files. The complete
raw-score replay is described later on this page and has passed for all 314
metric outputs from 1,722 checksum-verified retained inputs. The archives are
public and all uploaded asset digests verify; a fresh public download and full
replay are running. Current verification is recorded in
[STATUS.txt](../reproducibility/STATUS.txt).

The input workspace contains:

- `revised-metrics/`: per-fold and per-seed metric files with candidate
  provenance, association identities, and raw/calibrated ranks. It also contains
  the exact wrong-query report used by the supporting table command.
- `revised-main-controls.json`, `revised-excluded-primary-controls.json`, and
  `revised-excluded-secondary-controls.json`: annotation-count control reports.

Use the same workspace for both commands. The first writes aggregate and paired
comparison reports into `revised-metrics/`; both write figures/tables only to
the specified output root. Create a dedicated CPU analysis environment before running the commands:

```bash
python3 -m venv "$EMBEDDING_ENV"
"$EMBEDDING_ENV/bin/python" -m pip install -r reproducibility/requirements-embedding-replay.txt
"$EMBEDDING_ENV/bin/python" -m pip check
. "$EMBEDDING_ENV/bin/activate"
export SOURCE_DATE_EPOCH=1793232000
```

The version pins match the verified local NumPy, SciPy, pandas and Matplotlib
stack. Training environments remain separate. The workstation replay records
its Python and package versions alongside the results.

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
the fresh public raw-score replay currently running.

The [complete archive recipe](embedding-archive-replay.md) downloads the retained
scores and inputs, runs the following commands, and checks their outputs against
the archived references and manuscript. Its full public replay status is tracked separately.
The commands below assume that checksum-bound input layout already exists.
`INPUT_ROOT` contains `legacy-embedding/`. The committed manifest
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
validation_files=("$LEGACY_INPUTS"/completed-main-pfs/transd_fold_0_*_cal_disease.validation.tsv)
test "${#validation_files[@]}" -eq 1
test -f "${validation_files[0]}"
python reproducibility/paired_diagnostics.py \
  --workspace "$EMBEDDING_WORKSPACE" --validation-file "${validation_files[0]}"
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
removes its old completion marker. The full staged raw-score replay reproduces
all 314 metric files after excluding path-bearing provenance, all three controls
after excluding their pool locations, all ten tables, paired statistics and
rank diagnostics. Both CDF images match byte for byte. The
[complete replay record](../reproducibility/verification/legacy-embedding-full-replay.json)
identifies the source commit, environment and manifest. It establishes replay
from retained scores, not historical retraining or fresh public download.

The prepared-input count table is generated separately from the five retained
annotation-pair files. All five counts and the complete LaTeX table reproduce
exactly; its JSON sidecar records input hashes and distinguishes prepared pairs
from fresh provider downloads. See [the verification record](../reproducibility/verification/prepared-input-table.json).
