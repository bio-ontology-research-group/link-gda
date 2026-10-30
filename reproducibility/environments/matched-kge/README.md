# Matched embedding execution environment

`pixi.toml` and `pixi.lock` are the unchanged files used by the matched KGE
production jobs. `environment.json` records the installed Python distribution
versions and all Conda package versions, builds, MD5 and SHA-256 identities.
The Linux x86-64 environment contains Python 3.11.15, PyTorch 2.10.0+cu128,
PyKEEN 1.11.1, NumPy 2.4.2 and OpenJDK 17. It is separate from the ULTRA/query
environment and the pinned saved-score analysis environment.

From the repository root, with Pixi 0.67.2 installed:

```bash
KGE_MANIFEST="$PWD/reproducibility/environments/matched-kge/pixi.toml"
mkdir -p "$PWD/kge-scratch"
export TMPDIR="$PWD/kge-scratch"
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=$TMPDIR"
pixi install --locked --manifest-path "$KGE_MANIFEST"
pixi run --locked --manifest-path "$KGE_MANIFEST" \
  python "$PWD/reproducibility/verify_matched_kge_environment.py" \
  --capture "$PWD/reproducibility/environments/matched-kge/environment.json" \
  --output "$PWD/kge-environment-verification.json"
```

The verifier compares every captured package identity, runs real CPU TransD
and ConvKB scoring/backpropagation, checks NumPy/Torch interchange, and creates
an OWL ontology through mOWL. It requires Java to come from the reconstructed
environment. Use `pixi run` so the environment's OpenJDK activation is applied.
The lockfile must remain unchanged during reconstruction.

A clean reconstruction with Pixi 0.67.2 reproduced all 67 Conda package
identities and all 93 Python distributions. The CPU model, array-interchange
and pinned-Java checks pass; see the
[verification record](../../verification/matched-kge-environment-rebuild.json).

Run the original scientific source through this environment, for example:

```bash
pixi run --locked --manifest-path "$KGE_MANIFEST" \
  python "$TRAINING_SOURCE/code/training/run_matched_kge.py" --help
```

Training and full inference require their documented prepared inputs and a
GPU allocation. Installed-package reconstruction and CPU smoke checks do not
establish GPU trajectory or full prediction equivalence.
