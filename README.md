# LinkGDA

LinkGDA ranks candidate genes for a new disease from its phenotype profile. It
learns gene–phenotype links on a graph combining phenotype, function, expression,
and known gene–disease associations. The main experiments use TransD and OWL2Vec*.

## Setup

Use Python 3.11.15 and Java 17+ (for mOWL), plus `wget` for data downloads. `requirements.txt` is the single
pinned Python dependency file for training, analysis, figures, and tests:

```bash
uv venv --python 3.11.15 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

For CPU-only checks, install the PyTorch CPU wheel first using the
[testing instructions](docs/testing.md). The external ULTRA baseline retains
its separate environment because its PyTorch/CUDA stack is incompatible.

Run commands from the repository root, or from the prepared benchmark directory
when evaluating the separate excluded-gene benchmark. W&B is optional for
standalone training.

## Reproduce the experiments

1. Prepare the input data and disease-disjoint folds:

   ```bash
   python code/data/download_data.py
   python code/data/build_association_files.py
   python code/data/generate_folds.py
   python code/projector/prepare_go_edges.py --data-dir data
   ```

   Use the original data snapshots to reproduce the paper's results; the download
   script retrieves current provider files. The preparation command uses the
   paper's default GO projection settings.

2. Train the main calibrated LinkGDA configuration on the ten folds:

   ```bash
   for fold in $(seq 0 9); do
     python code/training/kge_transd.py --fold "$fold" \
       --use_phenotypes --use_functions --use_site --use_graph \
       --projector_name owl2vecstar \
       --embedding_dim 200 --batch_size 65536 --learning_rate 0.001 \
       --random_seed 0 --tolerance 15 \
       --calibrated_selection --write_baselines --no_sweep
   done
   ```

3. Recompute tables and tests from saved per-instance predictions. See the
   [reproduction guide](docs/reproduction.md) for baseline commands, raw and
   calibrated settings, ConvKB-D initialization, excluded-benchmark construction,
   result filenames, and statistical analysis. Each comparison must use its
   documented configuration; the command above covers only the main LinkGDA model.

Original prediction and checkpoint files are not bundled. See the
[reproduction guide](docs/reproduction.md) for experiment-specific settings and
current reproduction limits.

## Tests

After installing the dependencies, run:

```bash
python -m pytest -q
```

The [testing guide](docs/testing.md) describes CPU setup and test coverage.
The [paper-result checks](docs/paper_result_checks.md) recompute selected results
from the original prediction files, which must be supplied separately.

## Repository layout

- Root level: `requirements.txt`, `environment-ultra.yml`, `config.toml` (`.example`), and
  `sweeps/`.
- `code/training/`: TransD and ConvKB-D entry points.
- `code/link_gda/`: shared evaluation, data splitting, and training utilities.
- `code/data/`: downloads, annotation preparation, benchmark construction, and folds.
- `code/baselines/`: external baselines (ULTRA drivers and launchers, Exomiser
  evaluation, semantic-similarity Groovy scripts).
- `code/projector/`: GDAProjector Scala source, compiler, and UPheno projection.
- `code/analysis/`: results analysis and reporting tools (metric tables, seed
  aggregation, RQ1/RQ2 tests, leakage checks, overlap labels, rank-CDF values,
  W&B sweep extraction under `code/analysis/wandb/`).
- `code/figures/`: figure generators; see the layout map for historical-data limits.
- `tests/`: regression tests.
- `code/reproduce/`: baseline recording and comparison tools.
- `code/archive/`: [superseded workflows](code/archive/README.md).
- `data/`: inputs and saved results.
- `paper/`: separate manuscript repository, excluded from this repository's history.

Active tools are organized under `code/`; the
[code layout map](docs/code_layout.md) records which tools live where and where
each one moved from.
