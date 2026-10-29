# Final native and UPheno baseline results

The final collection contains 26 variants across ten disease folds and all 42
scoring/cohort conditions. It combines native HPO Phrank (direct-gene and
disease-max), native PhenoRank with the retained 2016 resources (simulation and
continuous scores), ten UPheno Phrank variants, and twelve UPheno PhenoRank
variants. The UPheno PhenoRank simulations use the published tier sampler,
uncapped query sizes and 1,000 simulations. UPheno Phrank uses the
author-compatible information content for parentless terms.

The complete evaluator checked scientific input identities, completion metadata,
score/candidate/pair hashes and cohort counts before producing the report.
Every result for the twelve Phrank variants agrees with the separately verified
Phrank collection. [Verification](../reproducibility/verification/final-baseline-evaluation.json)
binds the complete report and score manifest.

Regenerate the two complete baseline tables and the 26-by-42 metric CSV:

```bash
python reproducibility/generate_baseline_tables.py \
  --report-json reproducibility/results/final-baseline-report.json \
  --score-manifest reproducibility/results/final-baseline-score-manifest.json \
  --out-dir "$GENERATED_TABLES"
```

The exporter rejects partial model/fold collections, changed manifest bindings,
non-finite metrics and unexpected cohorts. The retained score manifest records
the verified production locations; it is evidence for these report-based table
commands. The portable archive recipe below rebuilds that manifest with paths
to the extracted scores; do not use the original production paths for score replay.

`reproducibility/run_pipeline.sh generate-tables` includes these tables and
replaces the old Phrank/PhenoRank rows in the main and subgroup bias tables.
The historical graph/query rows remain under their existing protocols pending
the controlled production experiments. To regenerate just the updated bias
tables, use:

```bash
python reproducibility/generate_bias_tables.py \
  --report-json reproducibility/results/stepwise-bias-report-v4.json \
  --baseline-report reproducibility/results/final-baseline-report.json \
  --baseline-score-manifest reproducibility/results/final-baseline-score-manifest.json \
  --out-dir "$GENERATED_TABLES"
```

Omitting both baseline arguments retains the historical table path. Supplying
only one is an error. The updated path validates the complete baseline report
before replacing every historical Phrank/PhenoRank entry; it never combines old
and new baseline estimates under one label.

The native PhenoRank simulation score yields 5.87% Hits@10, compared with 0.65%
for its continuous propagation score on the full benchmark. Further per-gene
calibration of the simulation score yields 1.13%. Native HPO direct Phrank yields
11.00% on the full benchmark and 0.09% on the zero-overlap cohort under the same
native scoring rule; MGI-only UPheno Phrank yields 3.30% in that restricted cohort.
These are descriptive benchmark estimates. Native scores, additional score
transformations and changes in the evaluated population remain explicit.

## Replay the score archives

The `baseline-outputs-20261029-v1` collection contains 11 archives with all 588
score and provenance files, the reference report and complete table exports.
Its manifest is committed under `reproducibility/artifacts/`. The
[public artifact release](https://github.com/bio-ontology-research-group/link-gda/releases/tag/baseline-outputs-20261029-v1)
is bound to source `bdbf2b5`; its tag, all asset digests and unauthenticated
manifest download have been verified. Local extracted-score replay passes for
every aggregate. A fresh public tagged-checkout download and complete score
replay also passes: all 26 variants and 42 conditions, the CSV and both table
bodies agree exactly. The [fresh replay record](../reproducibility/verification/final-baseline-fresh-public-replay.json)
binds the source, environment, archive manifest, report and documented recipe.
The ten benchmark components come from `inputs-20261028-v1`. The native resources
are required for rerunning the scorers, but not for replaying these saved scores.

Use the analysis environment described in the [reproduction instructions](../reproducibility/README.md).
From the tagged checkout, with a new empty workspace:

```bash
WORKSPACE="$PWD/replay-final-baselines"
mkdir -p "$WORKSPACE/scratch"
export TMPDIR="$WORKSPACE/scratch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
components=()
for fold in {0..9}; do components+=("benchmark-fold$fold"); done
python code/reproduce/artifacts.py fetch \
  --manifest reproducibility/artifacts/inputs-20261028-v1.json \
  --root "$WORKSPACE/inputs" --cache "$WORKSPACE/downloads" \
  --components "${components[@]}"
python code/reproduce/artifacts.py fetch \
  --manifest reproducibility/artifacts/baseline-outputs-20261029-v1.json \
  --root "$WORKSPACE/inputs" --cache "$WORKSPACE/downloads"
BASELINES="$WORKSPACE/inputs/baseline-outputs"
python reproducibility/build_baseline_manifest.py \
  --data-root "$WORKSPACE/inputs/benchmark/data" \
  --native-phrank "$BASELINES/native-phrank" \
  --native-phrank-disease-max "$BASELINES/native-phrank-disease-max" \
  --native-phenorank "$BASELINES/native-phenorank" \
  --upheno-phrank "$BASELINES/upheno-phrank" \
  --upheno-phenorank "$BASELINES/upheno-phenorank" \
  --output "$WORKSPACE/verified-score-manifest.json"
python code/analysis/evaluate_stepwise_bias.py \
  --base-dir "$WORKSPACE/inputs/benchmark" \
  --score-manifest "$WORKSPACE/verified-score-manifest.json" \
  --out-json "$WORKSPACE/verified-report.json" \
  --ranks-dir "$WORKSPACE/ranks" --workers 2
python reproducibility/generate_baseline_tables.py \
  --report-json "$WORKSPACE/verified-report.json" \
  --score-manifest "$WORKSPACE/verified-score-manifest.json" \
  --out-dir "$WORKSPACE/tables"
python - "$WORKSPACE" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
reference = root / 'inputs/baseline-outputs'
actual = json.loads((root / 'verified-report.json').read_text())
assert len(actual) == 26 and {len(v) for v in actual.values()} == {42}
assert actual == json.loads((reference / 'verified-report.json').read_text())
assert actual == json.loads(Path('reproducibility/results/final-baseline-report.json').read_text())
csv = 'final-baseline-metrics.csv'
assert (root / 'tables' / csv).read_bytes() == (reference / 'tables' / csv).read_bytes()
def body(path):
    return '\n'.join(line for line in path.read_text().splitlines()
                     if not line.startswith('% Score manifest SHA256:'))
for name in ['final-baseline-phrank.tex', 'final-baseline-phenorank.tex']:
    assert body(root / 'tables' / name) == body(reference / 'tables' / name)
print('All 26 variants, 42 conditions, CSV and table bodies agree exactly.')
PY
```

Archive and member hashes are checked before installation. The manifest builder
then checks all five components, ten folds, prepared input identities, scorer
settings and completion evidence. Relocating the scores changes the manifest
hash, so table comparison excludes only that provenance-comment line; all
reported values and other table content must agree exactly. These artifacts
establish saved-score reproducibility. Scorer regeneration has its own
[native-baseline workflow](native-baselines.md).
