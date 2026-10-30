# Historical comparator score replay

**Data access, 30 October 2026:** temporary bulk GitHub Release attachments have
been withdrawn. Use the [curated archive restore instructions](frozen-archive.md#use-restored-views-with-the-existing-recipes)
to replace the initial archive download below with an exact restored view.
Institutional bundle verification is in progress; public Zenodo access remains
pending author-managed deposition. See [artifact storage and access](artifact-storage.md).

The `comparator-outputs-20261029-v1` artifact release supplies the retained
candidate scores for 24 historical comparator variants and the phenotype caches
needed to reconstruct the direct-overlap aggregation control. The shared
evaluator reports all 25 variants across ten folds and 42 scoring/cohort
conditions. The prepared benchmark inputs come from `inputs-20261028-v1`.

These outputs include the original embedding and semantic comparators,
Exomiser-Phive, two TransD diagnostics, three frozen-ULTRA graph controls, and
nine historical no-bridge embedding variants. The latter differ from the
original models in training settings and remain descriptive comparisons.
The new matched-KGE experiment has a separate protocol and output collection.

Use the environment documented in [reproduction instructions](../reproducibility/README.md).
From the tagged checkout, the following Bash commands fetch the ten benchmark
components and all comparator components. Each archive and extracted member
is checked against its published checksum. Use an empty workspace for a fresh
download; rerunning against an existing workspace verifies retained bytes and
rejects conflicting files.

```bash
WORKSPACE="$PWD/replay-historical-comparators"
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
  --manifest reproducibility/artifacts/comparator-outputs-20261029-v1.json \
  --root "$WORKSPACE/inputs" --cache "$WORKSPACE/downloads"
python reproducibility/build_historical_comparator_manifest.py \
  --base-dir "$WORKSPACE/inputs/benchmark" \
  --output "$WORKSPACE/verified-score-manifest.json"
python code/analysis/evaluate_stepwise_bias.py \
  --base-dir "$WORKSPACE/inputs/benchmark" \
  --score-manifest "$WORKSPACE/verified-score-manifest.json" \
  --out-json "$WORKSPACE/verified-report.json" \
  --ranks-dir "$WORKSPACE/ranks" --workers 2
python - "$WORKSPACE" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
actual = json.loads((root / 'verified-report.json').read_text())
expected = json.loads((root / 'inputs/historical-comparators/verified-report.json').read_text())
historical = json.loads(Path('reproducibility/results/stepwise-bias-report-v4.json').read_text())
assert len(actual) == 25 and {len(v) for v in actual.values()} == {42}
assert actual == expected
assert all(actual[name] == historical[name] for name in actual)
print('All 25 variants and 42 conditions agree exactly.')
PY
```

The manifest builder checks the candidate and disease axes, prepared-input
identities, phenotype-cache completion records and file checksums. It recreates
the overlap control by omitting cached phenotype scores associated with each
gene's training phenotypes, averaging the remaining query entries, and using
the cached background mean when no query entry remains. The cached scores
retain message passing through the original graph.

The release verifies score arithmetic and the specified aggregation operation.
Original training trajectories and complete trained-checkpoint archives are
separate requirements. Publication and fresh-download verification status are
recorded in [STATUS.txt](../reproducibility/STATUS.txt).

A fresh unauthenticated replay from the public tagged checkout verifies every
archive/member checksum and reproduces all 25 variants and 42 conditions exactly,
including agreement with the historical report. The [verification record](../reproducibility/verification/historical-comparator-fresh-public-replay.json) identifies the source commit and report hashes.
