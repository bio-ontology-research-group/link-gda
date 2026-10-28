# Controlled PhenoQuery experiments

The [fixed protocol](../reproducibility/protocols/controlled-query-20261028.md)
specifies four arms, two initializations, three seeds and ten folds. Each run
uses 500 updates and validation-only checkpoint selection with the initial
checkpoint retained as a fallback. Both selected and initial checkpoints are
evaluated after every campaign's selection is frozen.

Inputs are a prepared EQ graph directory with `data/foldN` and `projection/`,
the benchmark workspace, and the pinned ULTRA source tree containing both
`ckpts/ultra_4g.pth` and `ckpts/ultraquery.pth`. The two checkpoint SHA256 values
used for the recorded GPU pilots are:

```
48a046e708adf5632d87c30eacae01f5f51466b2301effdc2cb42358d22854e0  ultra_4g.pth
9b6dc20801c35e7c9ebe65764acb4ed9cb5bffc6b0a0a727bd48935589e9388d  ultraquery.pth
```

Use the CUDA environment described in `environment-ultra.yml`; the executed
pilot environment records PyTorch 2.1.2+cu121, CUDA 12.1, NumPy 1.26.4 and
PyG 2.4.0. Full environment and input-archive publication is tracked separately
in [reproduction status](../reproducibility/STATUS.txt).

Run on an allocated GPU, with project-local scratch:

```bash
python code/baselines/run_controlled_query.py \
  --root RUNS --prepared-root PREPARED_EQ --base WORKSPACE --ultra-root ULTRA \
  --phase plan
python code/baselines/run_controlled_query.py \
  --root RUNS --prepared-root PREPARED_EQ --base WORKSPACE --ultra-root ULTRA \
  --phase pilot
python code/baselines/run_controlled_query.py \
  --root RUNS --prepared-root PREPARED_EQ --base WORKSPACE --ultra-root ULTRA \
  --phase all
```

`plan` writes the immutable protocol/source/configuration records and references
the prepared graph files. `pilot` verifies upstream projection parity, actual
updates and four-update uninterrupted/resumed equivalence under both
initializations. It uses four validation diseases and computes no test scores.
`all` runs the 240 training tasks serially, validates every completion, creates
the global selection record, and scores all initial and selected checkpoints.

For cluster dispatch, use the generated `RUNS/INITIALIZATION-seedN/configs/*.json`
with `run_phenoquery.py --root CHILD --base WORKSPACE --ultra-root ULTRA
--config CONFIG`. Production configurations require both technical gates.
Hold an exclusive lock per configuration, as the serial coordinator does.
After all training tasks finish, run the coordinator with `--phase freeze`,
then `--phase score`. The scoring driver checks the global selection before
loading test outputs. Initial checkpoints use the `_frozen` output suffix.

Source changes require a new campaign root and new pilot gates. Completed
trajectories are verified before reuse; incomplete trajectories restore model,
optimizer, selection and NumPy/PyTorch RNG states. GPU equivalence uses
`rtol=1e-5, atol=1e-6`; selected-step and validation-metric agreement are checked
as well. No test-informed configuration or failed run is silently substituted.
