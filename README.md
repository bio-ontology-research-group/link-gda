# Disentangling Structural and Training-Status Biases in Inductive Gene–Disease Prioritization (`LinkGDA` & `PhenoQuery`)

Code, command-line workflows, baseline refitting tools, and reproducibility suite for **Disentangling structural and training-status biases in inductive gene–disease prioritization across knowledge graphs and graph foundation models**.

This repository provides:
1. **`LinkGDA` (`code/training/kge_transd.py`, `code/training/kge_head_experiment.py`, `code/training/retrain_nobridge_kge.py`)**: Multi-hop knowledge-graph embedding rankers (`TransD` and `ConvKB-D` with `OWL2Vec*` and `GDAProjector` projections) combining cross-species phenotype (`UPheno v2`, `HPO`, `MP`), function (`GO`), and anatomical expression (`UBERON`, `GTEx`) annotations under untyped (`U1`), candidate-pool typed (`T1`), and profile hard-negative (`P-pool`) sampling, on both original graphs and disease-bridge-free graphs.
2. **Relational Graph Foundation Models (`code/baselines/run_ultra_matched.py`, `code/baselines/run_phenoquery.py`)**: Zero-shot and adapted `ULTRA` (`ultra_4g`), `UltraQuery` (`ultraquery.pth`), and our disease-bridge-free, logical Entity–Quality (`EQ`) ontology-augmented conjunctive adaptation (`PhenoQuery`: `bridge_pool`, `plain_pool`, `plain_and`, `entity_pool`, `entity_and`, `typed_pool`, `typed_and`, `structure_pool`, `structure_and`).
3. **Matched 10-Fold `Phrank` and `PhenoRank` Refitting (`code/baselines/run_matched_phrank_phenorank.py`)**: Fold-specific refitting of `Phrank` (`HPO` and cross-species `UPheno v2 + MGI`, `disease_max`, `direct`, and `MGI-Only`) and `PhenoRank` (`HM-Add+PPI`, `HM-Bal+PPI`, `Mouse+PPI`, and `Mouse-NoPPI`, with 1,000-simulation empirical $p$-value calibration per query length $|P_d|$, RWR tie-breaking, and continuous pre-simulation scoring) across all 13,625 candidates.
4. **Five-Stage Stepwise Bias Decomposition and Correction (`code/analysis/evaluate_stepwise_bias.py`, `reproducibility/`)**: Sequential evaluation from raw scoring (`S0`) through leave-one-disease-out per-gene reference calibration (`S1`), within-query status-stratum standardization (`S2 Status-Z`), zero-direct-overlap cohort evaluation (`S3`, $J_{\mathrm{same}} = 0$), and duplicate-profile collapse with multi-gene co-causal filtering (`S4`).

---

## Quick Start & Reproducibility Suite (`reproducibility/`)

See **[`reproducibility/README.md`](reproducibility/README.md)** for complete step-by-step instructions and the master CLI dispatcher [`reproducibility/run_pipeline.sh`](reproducibility/run_pipeline.sh):

```bash
./reproducibility/run_pipeline.sh --help
```

To immediately regenerate the main and supplementary bias-correction LaTeX tables (`revised-bias-stepwise.tex` and `revised-bias-baselines-full.tex`) from the precomputed 10-fold evaluation summary (`reproducibility/results/stepwise-bias-report-v4.json`):

```bash
./reproducibility/run_pipeline.sh generate-tables
```

---

## Environment Setup

Use Python 3.10+ and Java 17+ (for `mOWL`), plus `wget` for data downloads:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

For CPU-only checks, see the [testing instructions](docs/testing.md). The external `ULTRA` / `UltraQuery` environment specification is in `environment-ultra.yml`.

---

## Command-Line Entry Points

### 1. Data Preparation and Ontology Projection
```bash
python code/data/download_data.py
python code/data/build_association_files.py
python code/data/generate_folds.py
python code/projector/project_independent.py --data-dir data
```

### 2. Knowledge-Graph Embedding Training (`LinkGDA`, `ConvKB-D`, `INDIGENA`)
- **Standard graph (`LinkGDA-pfs`, dual raw/calibrated selection):**
  ```bash
  for fold in $(seq 0 9); do
    python code/training/kge_transd.py --fold "$fold" \
      --use_phenotypes --use_functions --use_site --use_graph \
      --projector_name owl2vecstar --graph_policy independent-unique \
      --candidate_pool phenotype --calibration_unit disease \
      --random_seed 0 --val_seed 0 --embedding_dim 200 \
      --batch_size 65536 --learning_rate 0.001 --tolerance 15 \
      --dual_arms --write_baselines --no_sweep
  done
  ```
- **Disease-bridge-free retraining (`LinkGDA-U1`, `LinkGDA-T1`, `LinkGDA-Ppool`, `ConvKB-D`, `INDIGENA`):**
  ```bash
  for fold in $(seq 0 9); do
    for cond in U1_INDIGENA_PPOOL T1 CONVKB; do
      python code/training/retrain_nobridge_kge.py \
        --fold "$fold" --condition "$cond" \
        --data-root data --out-root data/results/nobridge_runs
    done
  done
  ```

### 3. `ULTRA`, `UltraQuery`, and `PhenoQuery` Training & Evaluation
```bash
./reproducibility/run_pipeline.sh train-ultra
./reproducibility/run_pipeline.sh train-phenoquery
```

### 4. Matched 10-Fold `Phrank` and `PhenoRank` Refitting
```bash
python code/baselines/run_matched_phrank_phenorank.py \
  --base-dir data --out-root data/results/corrected_v3
```

### 5. Stepwise Bias Evaluation (`S0`–`S4`) and Table Generation
```bash
python code/analysis/evaluate_stepwise_bias.py \
  --base-dir data \
  --eq-dir data/results/phenoquery \
  --out-json reproducibility/results/stepwise-bias-report-v4.json

python reproducibility/generate_bias_tables.py \
  --report-json reproducibility/results/stepwise-bias-report-v4.json \
  --out-dir reproducibility/tables
```

---

## Running the Test Suite

```bash
uv run pytest -q
```

---

## Repository Layout

- `reproducibility/`: End-to-end CLI pipeline (`run_pipeline.sh`), table and metric generators (`generate_bias_tables.py`, `revision_metrics.py`, `supporting_tables.py`, `structural_controls.py`), and precomputed 10-fold JSON evaluation results (`results/`).
- `code/training/`: `TransD` (`kge_transd.py`), `ConvKB-D` (`kge_convkb_d.py`), candidate-pool / profile hard-negative training (`kge_head_experiment.py`), and disease-bridge-free KGE retraining (`retrain_nobridge_kge.py`).
- `code/link_gda/`: Shared evaluation, calibration (`calibration.py`), candidate-pool selection (`candidates.py`), negative sampling (`negative_sampling.py`), profile ranking (`profile_ranking.py`), and data utilities.
- `code/baselines/`: `ULTRA` and `UltraQuery` drivers (`prepare_ultra_data.py`, `score_ultra.py`, `run_ultra_matched.py`), `PhenoQuery` driver (`run_phenoquery.py`, `eq_model_reference.py`), matched `Phrank` and `PhenoRank` refitting (`run_matched_phrank_phenorank.py`), `Exomiser` evaluation (`exomiser_eval.py`), and semantic similarity scripts.
- `code/projector/`: `GDAProjector` Scala source, compiler, and independent ontology projection (`project_independent.py`).
- `code/analysis/`: Stepwise bias decomposition (`evaluate_stepwise_bias.py`), statistical tests (`rq1_stats.py`, `rq2_stats.py`), overlap stratification, and popularity controls.
- `code/figures/`: Calibration schematic and rank-CDF figure generators.
- `tests/`: Automated unit, layout, and reproducibility regression tests.
- `docs/`: [Revision protocol](docs/revision-20261013.md), [reproduction guide](docs/reproduction.md), [code layout map](docs/code_layout.md), and [testing guide](docs/testing.md).
