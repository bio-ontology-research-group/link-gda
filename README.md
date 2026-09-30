# LinkGDA

LinkGDA ranks candidate genes for a new disease from its phenotype profile. It
learns gene–phenotype links on a graph combining phenotype, function, expression,
and known gene–disease associations. The main experiments use TransD and OWL2Vec*.

## Setup

Use Python 3.11.15 and Java 17+ (for mOWL). `requirements.txt` is the single
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
   ```

   Downloads use moving provider endpoints. Exact numerical reproduction requires
   the original source snapshots and processed inputs.

2. Train the main calibrated LinkGDA configuration on the ten folds:

   ```bash
   for fold in $(seq 0 9); do
     python kge_transd.py --fold "$fold" \
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

The complete prediction/checkpoint collection is not bundled in this checkout.
Exact OWL2Vec* excluded-run commands still need verification against archived run
metadata. Two metric paths currently use different AUC definitions; this remains
an explicit issue in the reproduction guide. The repository therefore does not
yet provide a verified end-to-end reproduction of every paper table.

Two [saved-prediction checks](docs/paper_result_checks.md) have verified the RQ1
mean ranks and adjusted p-value, and excluded-set LinkGDA-fs mean rank and Hits@10,
against 30 prediction files. The guide gives the commands and coverage limits.

## Regression checks

Use the [testing guide](docs/testing.md) to install the lightweight CPU test
environment, record a baseline, and compare later cleanup runs. These checks use
fixed synthetic examples and a small graph fixture. They test implementation
behavior; they do not replace verification against the paper's prediction files.
Local records are saved under gitignored `.reproducibility/`.

## Repository layout

- Root Python and shell files: current training, data, baseline, and evaluation tools.
- `analysis/`: fold-level statistical analysis (RQ1/RQ2 tests).
- `code/data/`: downloads, annotation preparation, benchmark construction, and folds.
- `code/analysis/`: results analysis and reporting tools (metric tables, seed
  aggregation, leakage checks, overlap labels, rank-CDF values).
- `code/figures/`: figure generators; see the layout map for historical-data limits.
- `tests/`: regression tests.
- `code/reproduce/`: baseline recording and comparison tools.
- `code/archive/`: [superseded workflows](code/archive/README.md).
- `data/`: inputs and saved results.
- `paper/`: separate manuscript repository, excluded from this repository's history.

Active tools are moving into the agreed `code/` layout in batches; the
[code layout map](docs/code_layout.md) records which tools live where and where
each one moved from.
