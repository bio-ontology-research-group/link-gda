# CPU score-analysis environment

This Linux x86-64 Pixi environment pins Python 3.13.5 and the saved-score analysis
stack, including NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3 and Matplotlib 3.10.7.
The lock fixes all transitive packages. `environment.json` captures the 22 Conda
package identities and 16 installed Python distributions from the tested IBEX
environment. Use Pixi 0.67.2.

From the repository root:

```bash
ANALYSIS_MANIFEST="$PWD/reproducibility/environments/cpu-analysis/pixi.toml"
mkdir -p "$PWD/analysis-scratch"
export TMPDIR="$PWD/analysis-scratch"
pixi install --locked --manifest-path "$ANALYSIS_MANIFEST"
pixi run --locked --manifest-path "$ANALYSIS_MANIFEST" \
  python "$PWD/reproducibility/verify_analysis_environment.py" \
  --capture "$PWD/reproducibility/environments/cpu-analysis/environment.json" \
  --output "$PWD/analysis-environment-verification.json"
export PYTHON_BIN="$PWD/reproducibility/environments/cpu-analysis/.pixi/envs/default/bin/python"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
```

The verifier checks every captured Python distribution and Conda package identity,
the 15 direct analysis pins, dependency consistency and binary-backed imports.
Use a new output path for each check. Keep the manifest and lock unchanged.

Real IBEX execution reproduces complete synthetic score analyses generated in a
different Python environment: 480 controlled query bundles, 48 variants across
42 bias conditions, and 40 matched variants across 42 conditions. All 5,600
matched rank-array payloads and primary statistics agree exactly. ZIP container
headers differ across Python versions; the replay verifies their array contents
and accounts only for the corresponding provenance hashes. See the
[execution evidence](../../verification/cpu-analysis-cross-environment-replay.json).

A fresh locked rebuild from the public checkout reproduces all 22 Conda package
identities and all 16 Python distributions, with unchanged manifest, lock and
capture. Dependency checks and binary imports pass; deliberately altered Python
versions, distribution versions and Conda digests are rejected. See the
[rebuild evidence](../../verification/cpu-analysis-environment-rebuild.json).
Production score replay, checkpoint inference and GPU training are separate checks. This environment
supports saved-score analysis and table generation; use the documented query
and matched-KGE environments to generate model predictions.
