# Historical embedding scores: full archive replay

**Data access, 30 October 2026:** temporary bulk GitHub Release attachments have
been withdrawn. Use the [curated archive restore instructions](frozen-archive.md#use-restored-views-with-the-existing-recipes)
to replace the initial archive download below with an exact restored view.
Institutional bundle verification is in progress; public Zenodo access remains
pending author-managed deposition. See [artifact storage and access](artifact-storage.md).

The `embedding-outputs-20261029-v1` collection supplies the exact retained
predictions, candidate identities, prepared annotation files and reference
results for the revised historical embedding analysis. It contains 34 archive
parts and 1,723 files, including the manifest for all 314 metric outputs.
The [public release](https://github.com/bio-ontology-research-group/link-gda/releases/tag/embedding-outputs-20261029-v1)
is pinned to source `ccd8c0989cc1a6f5fe9c87a77de8de09c149bda6`.
All 35 asset sizes/digests and an unauthenticated manifest download verify;
see the [publication record](../reproducibility/verification/legacy-embedding-publication.json).
The full fresh-download replay passes from a clean public tagged checkout:
all 34 archives and 1,723 members verify, and all 314 metric outputs, three
controls, paired statistics, diagnostics, eleven manuscript tables (including
input counts) and both CDF artifacts agree. See the
[fresh public replay record](../reproducibility/verification/legacy-embedding-fresh-public-replay.json).

The main benchmark uses 13,625 candidate genes. The excluded analysis uses the
corrected 425-pair cohort; the older 409-pair reconstruction is not substituted.
Historical graph/model comparisons retain their declared differences in
training and selection. This replay does not make those comparisons matched or
retrain the original models. The new controlled campaigns remain separate.

Create and activate the pinned [CPU analysis environment](../reproducibility/README.md#cpu-score-analysis-environment).
Allow at least 80 GB free for downloads, extracted scores and temporary staging.
`DOWNLOAD_CACHE` can point to another filesystem if needed. Use a new workspace:
existing metric/table outputs require an explicit replacement decision.

## Full score-to-table recipe

Run this Bash block from the tagged repository checkout. The downloader checks
each archive and every extracted member. The metric driver then verifies its
own checksum-bound prediction/candidate/training inputs before calculating ranks.

```bash
set -euo pipefail
WORKSPACE="$PWD/replay-embedding"
INPUT_ROOT="$WORKSPACE/inputs"
EMBEDDING_WORKSPACE="$WORKSPACE/analysis"
GENERATED_ASSETS="$WORKSPACE/generated"
DOWNLOAD_CACHE="${DOWNLOAD_CACHE:-$WORKSPACE/downloads}"
mkdir -p "$WORKSPACE/scratch" "$WORKSPACE/plot-cache"
export TMPDIR="$WORKSPACE/scratch" MPLCONFIGDIR="$WORKSPACE/plot-cache"
export SOURCE_DATE_EPOCH=1793232000
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python code/reproduce/artifacts.py fetch \
  --manifest reproducibility/artifacts/embedding-outputs-20261029-v1.json \
  --root "$INPUT_ROOT" --cache "$DOWNLOAD_CACHE"
cmp reproducibility/manifests/embedding-metrics-20261029.json \
  "$INPUT_ROOT/legacy-embedding/metric-manifest.json"
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
python reproducibility/finalize_results.py \
  --workspace "$EMBEDDING_WORKSPACE" --out-dir "$GENERATED_ASSETS"
python reproducibility/supporting_tables.py \
  --workspace "$EMBEDDING_WORKSPACE" --out-dir "$GENERATED_ASSETS"

python - "$WORKSPACE" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
reference = root / 'inputs/legacy-embedding-reference'
work = root / 'analysis'
generated = root / 'generated'
manifest = Path('reproducibility/manifests/embedding-metrics-20261029.json')
entries = json.loads(manifest.read_text())['entries']
assert len(entries) == 314
completion = json.loads((work / 'embedding-replay-shard-0-of-1.json').read_text())
assert completion['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
assert len(completion['outputs']) == 314
for entry in entries:
    path = work / 'revised-metrics' / entry['output']
    assert completion['outputs'][str(path.relative_to(work))] == hashlib.sha256(path.read_bytes()).hexdigest()
    actual = json.loads(path.read_text())
    actual.pop('provenance')  # The absolute input paths change after relocation.
    assert actual == json.loads((reference / 'metrics' / entry['output']).read_text()), entry['output']
for name in ['main', 'excluded-primary', 'excluded-secondary']:
    filename = f'revised-{name}-controls.json'
    actual = json.loads((work / filename).read_text())
    actual.pop('pool')  # The pool file location changes; its contents are verified.
    assert actual == json.loads((reference / filename).read_text()), filename
for name in ['main-comparisons.json', 'excluded-comparisons.json',
             'within-status-summary.json', 'paired-rank-diagnostics.json',
             'tuning-reuse-sensitivity.json']:
    actual = json.loads((work / 'revised-metrics' / name).read_text())
    specification = actual.pop('specification', None)
    if specification is not None:
        assert set(specification) == {'mode', 'setting', 'comparisons'}
        assert specification['mode'] == actual['mode']
        assert specification['setting'] == actual['setting'] == 'disease_loo'
        groups = {
            'LinkGDA vs main-indigena-pfs-calsel': ('main-linkgda-pfs-calsel', 'main-indigena-pfs-calsel'),
            'LinkGDA vs main-convkb-pfs-calsel': ('main-linkgda-pfs-calsel', 'main-convkb-pfs-calsel'),
            'LinkGDA vs main-ultra-pfs': ('main-linkgda-pfs-calsel', 'main-ultra-pfs'),
            'LinkGDA-f vs ULTRA-f': ('primary-linkgda-f-calsel', 'primary-ultra-f'),
            'LinkGDA-fs vs ULTRA-fs': ('primary-linkgda-fs-calsel', 'primary-ultra-fs'),
        }
        assert len(specification['comparisons']) == len(actual['comparisons'])
        for declared, result in zip(specification['comparisons'], actual['comparisons']):
            assert set(declared) == {'label', 'paired_files'}
            assert declared['label'] == result['comparison']
            pair = groups[declared['label']]
            indices = [(i, 0 if actual['mode'] == 'excluded' else i) for i in range(10)]
            expected = [[str(work / 'revised-metrics' / group / f'fold_{index}.json')
                         for group, index in zip(pair, pair_indices)] for pair_indices in indices]
            assert declared['paired_files'] == expected
    assert actual == json.loads((reference / name).read_text()), name
tables = sorted((reference / 'tables').glob('*.tex'))
assert len(tables) == 10
for path in tables:
    assert (generated / 'tables' / path.name).read_bytes() == path.read_bytes(), path.name
    assert (generated / 'tables' / path.name).read_bytes() == (Path('manuscript/tables') / path.name).read_bytes(), path.name
assert (generated / 'tables/prepared-inputs.tex').read_bytes() == Path('manuscript/tables/prepared-inputs.tex').read_bytes()
for suffix in ['pdf', 'png']:
    name = 'revised_rank_cdf.' + suffix
    assert (generated / 'fig' / name).read_bytes() == (reference / 'fig' / name).read_bytes(), name
assert (generated / 'fig/revised_rank_cdf.pdf').read_bytes() == Path('manuscript/fig/revised_rank_cdf.pdf').read_bytes()
print('All 314 metric outputs, three controls, paired statistics, diagnostics, ten result tables, input-count table and CDF artifacts agree.')
PY
```

The comparison removes recorded input-path provenance. Before dropping the
duplicated comparison specification, it checks its mode, score setting, labels
and all paired fold/seed file locations. Ranks, identities, metrics and all
statistical results must agree exactly. Table and CDF checks
are byte comparisons in the pinned plotting environment. These checks establish
reproduction from the archived scores and prepared inputs, not from present-day
provider downloads or from fresh historical training.

The main comparisons pair matching outer folds. In the excluded-cohort analysis,
each of the ten LinkGDA seeds is compared with the same fixed ULTRA prediction
file; the baseline is not retrained for each seed. The comparison check enforces
this distinction explicitly.

The lower-level [metric and table commands](embedding-table-replay.md) can also
be run separately. Parallel metric replay uses disjoint `--shard I --shards N`
commands, with all completion records required before downstream analysis; the
recipe above deliberately uses the single complete-run marker.
