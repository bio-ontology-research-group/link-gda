# Code

Active tools live under `code/`. [Code layout](../docs/code_layout.md) maps the
existing packages and their former locations.

## Compositional gene–disease association (`compositional_gda/`)

Path query-embedding models (GQE/TransE path, Query2Box, BetaE, GeometrE)
trained on one-hop facts and validated on composed two-hop phenotype queries.
See [the workflow](../docs/compositional-gda.md) for the input format,
configuration keys, outputs and the pause/resume protocol.

| module | role |
|---|---|
| `prepare.py` | builds a prepared fold from the frozen benchmark inputs |
| `models.py` | the four method families and the shared loss |
| `data.py` | prepared-fold verification, one-hop and two-hop query indices, checkpointable single and mixed samplers |
| `evaluate.py` | forward and inverse two-hop scoring, aggregation, calibration, ranks, selection |
| `train.py` | training (one-hop, or the two-hop diagnostic and its one-hop control), validation, selection publication, pause and resume |
| `recompute.py` | recomputes metrics and selections from saved validation arrays |
| `fit_diagnostics.py` | read-only training-fit ranks of a completed run's checkpoint on sampled training-graph queries |

Set up an environment and run the tests:

```bash
python -m venv .venv-compositional
.venv-compositional/bin/pip install -r requirements-compositional.txt
.venv-compositional/bin/python -m pytest tests/test_compositional_models.py tests/test_compositional_resume.py
```

Train one run, then check its selection from the saved arrays:

```bash
python code/compositional_gda/train.py --config CONFIG.json --run-dir RUN_DIR --device cuda --deadline-seconds 11400
python code/compositional_gda/recompute.py --run-dir RUN_DIR --data-dir PREPARED_FOLD
```

Setting `auxiliary` to `two_hop` or `one_hop_readout` with `auxiliary_slots`
runs the two-hop training diagnostic or its matched one-hop control; see
[the workflow](../docs/compositional-gda.md#two-hop-training-diagnostic).

Score how well a completed run fits its training graph:

```bash
python code/compositional_gda/fit_diagnostics.py --run-dir RUN_DIR --data-dir PREPARED_FOLD --output NEW_DIR \
    --device cuda --queries-per-relation 256 --seed 0 --expect-sample-sha256 SAMPLE_SHA256
```

The output directory must not exist. `--sample-only` builds and saves only the
query sample, on CPU, which gives the value for `--expect-sample-sha256`.

Exit code 0 means complete, 75 means paused and resumable with the same
command, and 3 means another process holds the run directory.

The optional `scripts/compositional/launch_checked.py` launcher checks two GPU
resume-test reports and the prepared data hashes before starting a pilot campaign.
See the [compositional study guide](../docs/compositional-gda.md) for the GPU check.
