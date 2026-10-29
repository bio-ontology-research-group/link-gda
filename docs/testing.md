# Regression checks before and after cleanup

These checks use small synthetic score files and a graph fixture to exercise the
current implementation. The resume and execution checks train small real
TransD/profile models and compare interrupted trajectories and exported scores.
They do not reproduce full-data GPU training or the paper's tables from primary
inputs. Score/report and saved-table checks have their own narrower scopes.

## Test environment

Use the project's pinned `requirements.txt` with Python 3.11.15. This includes
PyKEEN and Phrank, which are required during test collection. The smaller
[saved-score analysis environment](../reproducibility/README.md#cpu-score-analysis-environment)
does not provide the full test dependencies. CPU-only PyTorch is sufficient
for this suite.

```bash
uv venv --python 3.11.15 .venv
uv pip install --python .venv/bin/python torch==2.10.0 \
  --index https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
```

This installs all LinkGDA Python packages. Training also needs a JVM, and GPU
training needs a suitable PyTorch CUDA build and driver. The baseline recorder
captures the interpreter and package versions used.

## Record before changing the implementation

```bash
.venv/bin/python code/reproduce/record_baseline.py --label before-cleanup
```

Records live under `.reproducibility/`, which is gitignored. Existing labels must
not be overwritten: keep the first baseline and choose a new label for later
runs. Preserve this directory when switching branches or cleaning local files.

## Compare after a change

```bash
.venv/bin/python code/reproduce/record_baseline.py --label after-cleanup
.venv/bin/python code/reproduce/record_baseline.py \
  --compare .reproducibility/before-cleanup \
  --new .reproducibility/after-cleanup
```

Inspect the comparison report and exit status. Source hashes may change when
files move. Missing outputs, numerical changes beyond the recorded tolerance,
and lost or failed tests require investigation before continuing cleanup.

## Coverage and limits

The suite covers deterministic average ranks, ties, score-file parsing,
leave-one-out calibration, corrected fold-level statistics, query alignment,
and a small graph-construction/fallback fixture. It also tests the baseline
comparator with deliberately changed and missing outputs.

Both existing AUC definitions are characterized separately. Passing these checks
does not resolve their disagreement. The known missing-fold behavior in
`code/analysis/rq1_table.py` is documented by characterization tests; those tests do not endorse
it as correct statistical practice.

Full numerical reproduction additionally requires the original processed data,
folds, selected checkpoints, and per-instance predictions. A separate check must
recompute each paper table from those artifacts and match cases and candidates.
Checkpoint-to-score inference and full retraining are not covered by this suite.

The [latest complete development-environment run](../reproducibility/verification/release-tests.json)
records the tested commit, package versions and log checksum. It passes 408 tests
and nine subtests using Python 3.10.16/PyTorch 2.9.0; that compatibility check is
separate from the Python 3.11.15/PyTorch 2.10.0 setup documented above.

A [fresh public-checkout run](../reproducibility/verification/fresh-public-regression-suite.json)
passes all 409 tests at commit `afe67b6` using Python 3.11.15 and every one of the
76 pinned project requirements. It uses the independently reconstructed locked
KGE environment with an isolated Phrank 1.6 overlay, executes models on the CPU,
and passes `pip check`. The initial fresh run exposed an eager optional graph
import in preparation CLI help; that issue is fixed and covered by a test that
disables third-party packages. Later focused checks remain separately recorded
in the [status record](../reproducibility/STATUS.txt).

The [new complete public-checkout run](../reproducibility/verification/fresh-public-regression-suite-5301c3f.json)
passes all 442 tests at commit `5301c3f` in 440.98 seconds, with 19 warnings.
It verifies the same 76 package pins, passes `pip check`, and leaves the source
and environment locks unchanged. This run also covers the controlled-query bias
manifest, parallel scoring coordination, frozen-source checkpoint export,
cross-version rank-archive comparison and CPU environment verification added
since the earlier full run. It remains a CPU regression check; production GPU
inference and complete paper-result replay require their separate evidence.

## Checks against real paper results

The [paper-result checks](paper_result_checks.md) additionally recompute two
reported results from 30 real prediction files. They are separate from the fast
synthetic suite because the files are large and are not bundled in this checkout.
Both checks passed in the recorded pre-cleanup run.
