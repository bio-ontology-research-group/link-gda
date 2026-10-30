# Frozen experimental inputs

**Data access, 30 October 2026:** temporary bulk GitHub Release attachments have
been withdrawn. Use the [curated archive restore instructions](frozen-archive.md#use-restored-views-with-the-existing-recipes)
to replace the initial archive download below with an exact restored view.
Institutional bundle verification is in progress; public Zenodo access remains
pending author-managed deposition. See [artifact storage and access](artifact-storage.md).

The `inputs-20261028-v1` GitHub release archives the prepared experimental
inputs used by the historical query campaigns and the controlled query rerun.
The repository manifest
[`inputs-20261028-v1.json`](../reproducibility/artifacts/inputs-20261028-v1.json)
pins every archive and every extracted file by SHA-256. This is an input
snapshot, not a completed paper-results release. Predictions, trained model
selections and the complete upstream annotation-preparation provenance are
separate requirements.

## Download and verify

Run from this repository checkout. The downloader uses Python's standard
library; it needs no ML environment or GitHub credentials. Allocate at least
20 GB for extraction and temporary staging, plus space for any rebuilt graphs.

```bash
export WORKSPACE="$PWD/workspace"
export TMPDIR="$WORKSPACE/scratch"
mkdir -p "$TMPDIR"
python3 code/reproduce/artifacts.py fetch \
  --manifest reproducibility/artifacts/inputs-20261028-v1.json \
  --root "$WORKSPACE/inputs" --cache "$WORKSPACE/downloads"
```

There are 35 archives (about 1.15 GB compressed and 12.8 GB extracted).
`--components benchmark-fold0 eq-fold0 ultra-source-checkpoints` downloads
only the named components. An interrupted download can be rerun: complete
cached archives are rechecked; incomplete archives are downloaded again.
The installer rejects traversal paths, links, missing members, incorrect
hashes and conflicting existing files. Embedded per-component manifests are
installed under `inputs/MANIFESTS`.

| Extracted path | Contents |
|---|---|
| `benchmark/data/foldN` | Raw graph triples, training/validation/test associations, phenotype profiles, 13,625 ordered candidates, vocabulary and retained ULTRA cache |
| `eq/ontologies` | Frozen UPheno, GO and UBERON OWL snapshots |
| `eq/projection` | Ontology-expression projection, axiom provenance and manifest |
| `eq/data/foldN` | Five PhenoQuery graph variants and their identity metadata |
| `matched/graphs/foldN` | Historical matched EQ-augmented ULTRA/UltraQuery graphs |
| `benchmark/baselines/phenorank/resources` | Historical PhenoRank resources and HGNC mapping |
| `benchmark/native-hpo` | HPO DAG extracted for the native Phrank run |
| `sources/phrank`, `sources/phenorank` | Phrank 1.6 author source and PhenoRank license |
| `ultra` | Exact ULTRA source and pretrained checkpoint bytes used for the controlled campaign |
| `environment/ultra` | Captured Linux-64 Conda package URLs/hashes and pinned pip versions |

The exact ULTRA source is identified by its per-file hashes, not an inferred
upstream Git commit. The general `ultra_4g.pth` checkpoint hash begins
`48a046e7`; `ultraquery.pth` begins `9b6dc208`. Complete digests are in the
manifest. Upstream licenses accompany the source components.

## Matched KGE annotation supplement

The separate `matched-inputs-20261029-v1` prerelease supplies the frozen
gene-to-MP table used by INDIGENA in the matched KGE experiment. This file was
missing from the earlier input release; the earlier archives remain unchanged.

```bash
python3 code/reproduce/artifacts.py fetch \
  --manifest reproducibility/artifacts/matched-inputs-20261029-v1.json \
  --root "$WORKSPACE/inputs" --cache "$WORKSPACE/downloads"
```

The 1.44 MB archive installs `matched-kge/inputs/gene_phenotypes.csv` and its
provenance record. The table has 213,988 distinct gene--MP pairs, 13,626 genes
and 10,685 mouse phenotype terms; the scorer restricts these annotations to the
13,625 benchmark candidates and represented phenotype nodes. Its SHA-256 is
`94829ab611a3f02c4c33d3686028d8eded187c256f852d4e2cecb2f548b4d0b6`.
This preserves the exact processed input; reconstructing it from current MGI
downloads is outside this archive's verification scope.

## Reconstruct the ULTRA environment

For the separate matched embedding comparison, use the
[matched-KGE Pixi capture](../reproducibility/environments/matched-kge/README.md).
Its package identities and runtime differ from the ULTRA environment below.

The capture records Python 3.10.22, PyTorch 2.1.2+cu121 and PyG 2.4.0 from
the actual controlled-training environment. With Conda already installed:

```bash
conda create --prefix "$WORKSPACE/ultra-env" \
  --file "$WORKSPACE/inputs/environment/ultra/conda-explicit-linux-64.txt"
"$WORKSPACE/ultra-env/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html \
  -r "$WORKSPACE/inputs/environment/ultra/pip-requirements.txt"
```

Fresh reconstruction with Micromamba 2.10.0 and pip reproduced all 70 Conda
package versions/builds/MD5 identities and all 43 Python distributions.
`pip check`, imports, CPU scatter and NumPy/Torch interchange passed; see the
[environment verification](../reproducibility/verification/ultra-environment-rebuild.json).
GPU training additionally needs an appropriate CUDA driver and the compiler
configuration documented for the controlled campaign. Package reconstruction
does not establish GPU numerical parity.

A separate [GPU readiness check](gpu-readiness.md) now passes in both reconstructed
environments on IBEX A100 hardware. Its synthetic graphs verify runtime and
CPU/GPU compatibility; production checkpoint inference remains separate.

## Rebuild prepared graphs

Use the archived ULTRA environment. Each preparation command requires a new
output fold directory. Rebuild the benchmark cache from raw triples, project
the frozen OWL files, then build both graph families:

```bash
export PYTHON_BIN="$WORKSPACE/ultra-env/bin/python"
export INPUTS="$WORKSPACE/inputs"
export REBUILT="$WORKSPACE/rebuilt"
mkdir -p "$REBUILT/eq"
ln -s "$INPUTS/eq/ontologies" "$REBUILT/eq/ontologies"
"$PYTHON_BIN" code/baselines/prepare_benchmark_cache.py \
  --base "$INPUTS/benchmark" --root "$REBUILT/benchmark" \
  --ultra-root "$INPUTS/ultra"
"$PYTHON_BIN" code/phenoquery/projection.py --root "$REBUILT/eq"
"$PYTHON_BIN" code/baselines/prepare_phenoquery.py \
  --base "$REBUILT/benchmark" --root "$REBUILT/eq" \
  --ultra-root "$INPUTS/ultra"
"$PYTHON_BIN" code/baselines/prepare_ultra_matched.py \
  --base "$REBUILT/benchmark" --eq "$REBUILT/eq" \
  --root "$REBUILT/matched" --ultra-root "$INPUTS/ultra"
"$PYTHON_BIN" code/reproduce/verify_preparation.py \
  --reference "$INPUTS" --benchmark "$REBUILT/benchmark" \
  --projection "$REBUILT/eq" --eq "$REBUILT/eq" \
  --matched "$REBUILT/matched" --output "$REBUILT/verification.json"
```

The verifier checks vocabulary mappings, graph structure, every tensor's
shape/dtype/value and projection hashes. It accepts identical graph values
when serialization bytes differ, and records both hashes. Rebuilding these
graphs starts from frozen benchmark triples; it does not demonstrate
reconstruction of those triples from the original annotation providers.
The [projection/matched-graph verification](../reproducibility/verification/preparation-matched-projection.json)
reproduces all five projection outputs and all ten matched graphs byte-for-byte.
The [PhenoQuery graph verification](../reproducibility/verification/preparation-eq.json)
also reproduces all 50 graph variants byte-for-byte. All ten
[benchmark caches](../reproducibility/verification/preparation-benchmark.json)
rebuild byte-for-byte from their raw triples.
The [public-download verification](../reproducibility/verification/public-input-replay.json)
checks all 35 archives and 469 files from a fresh tagged checkout without
credentials. The [complete public preparation chain](../reproducibility/verification/public-preparation-chain.json)
also passes from that tagged checkout and reconstructed environment: all 70 graph
caches and all five projection outputs are byte-identical to the archived inputs.

To check a reconstructed environment with the current repository verifier:

```bash
"$WORKSPACE/ultra-env/bin/python" code/reproduce/verify_environment.py \
  --capture "$INPUTS/environment/ultra/environment.json" \
  --output "$WORKSPACE/environment-verification.json"
```
