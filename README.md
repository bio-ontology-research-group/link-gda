# PhenoQuery: phenotype-query learning and evaluation of gene–disease prioritization

Code, baseline adapters and evaluation tools for PhenoQuery and the analysis of candidate-status, phenotype-overlap and graph-structure effects in gene–disease prioritization.

The [reproduction status](reproducibility/STATUS.txt) distinguishes verified score replay from ongoing controlled retraining and input-archive publication. This branch is an active repair; a complete clean-download reproduction has not yet been certified.

This repository provides:
1. **`LinkGDA` (`code/training/kge_transd.py`, `code/link_gda/head_experiment.py`, `code/training/retrain_nobridge_kge.py`)**: Multi-hop knowledge-graph embedding rankers (`TransD` and `ConvKB-D` with `OWL2Vec*` and `GDAProjector` projections) combining cross-species phenotype (`UPheno v2`, `HPO`, `MP`), function (`GO`), and anatomical expression (`UBERON`, `GTEx`) annotations under untyped (`U1`), candidate-pool typed (`T1`), and profile hard-negative (`P-pool`) sampling, on both original graphs and disease-bridge-free graphs.
2. **Relational Graph Foundation Models (`code/baselines/run_ultra_matched.py`, `code/baselines/run_phenoquery.py`)**: Zero-shot and adapted `ULTRA` (`ultra_4g`), `UltraQuery` (`ultraquery.pth`), and our disease-bridge-free, logical Entity–Quality (`EQ`) ontology-augmented conjunctive adaptation (`PhenoQuery`: `bridge_pool`, `plain_pool`, `plain_and`, `entity_pool`, `entity_and`, `typed_pool`, `typed_and`, `structure_pool`, `structure_and`).
3. **Phrank and PhenoRank baselines**: [Native HPO Phrank, the historical-resource PhenoRank adapter, and separate UPheno adaptations](docs/native-baselines.md). Human annotations and information content are fitted within each training fold. Native scores remain primary; extra post-hoc calibration is reported separately.
4. **Evaluation sensitivities** (`code/analysis/evaluate_stepwise_bias.py`): per-gene calibration, candidate-status normalization, training-phenotype overlap, reference deduplication, co-causal filtering and profile weighting. The sequential S0–S4 panels combine score and population changes; individual-factor panels separate their operations.

---

## Quick Start & Reproducibility Suite (`reproducibility/`)

See **[`reproducibility/README.md`](reproducibility/README.md)** for the available workflow commands and the master CLI dispatcher [`reproducibility/run_pipeline.sh`](reproducibility/run_pipeline.sh):

```bash
./reproducibility/run_pipeline.sh --help
```

To immediately regenerate the main and supplementary bias-correction LaTeX tables (`revised-bias-stepwise.tex` and `revised-bias-baselines-full.tex`) from the precomputed 10-fold evaluation summary (`reproducibility/results/stepwise-bias-report-v4.json`):

```bash
./reproducibility/run_pipeline.sh generate-tables
```

---

## Environment Setup

The [working manuscript source](manuscript/README.md) includes the main paper,
supplement, bibliography and required figures/tables, with clean-build
instructions. It is an explicitly dated draft; the final controlled-query and
matched-KGE result integrations await an updated public manuscript snapshot.
The complete matched saved-score analysis and checkpoint audit have passed;
see the [verification record](reproducibility/verification/matched-kge-complete-verification.json)
and [primary results](reproducibility/results/matched-kge-summary.json).
The source snapshot is separate from the
original manuscript repository.

Use Python 3.11.15 for the pinned project dependencies and Java 17+ (for
`mOWL`), plus `wget` for data downloads:

```bash
uv venv --python 3.11.15 .venv
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

The [embedding archive replay](docs/embedding-archive-replay.md) provides the
complete retained-score-to-table recipe for the original embedding and
excluded-cohort analyses. The [lower-level commands](docs/embedding-table-replay.md)
can also run individual stages. The archives are public, and a fresh tagged
checkout reproduces all 314 metric outputs, controls, diagnostics, tables and
CDF artifacts. See the [verification record](reproducibility/verification/legacy-embedding-fresh-public-replay.json).

### 2. Historical knowledge-graph embedding experiments

The retained results differ in budget, checkpoint selection, dimension and negative filtering, and must not be interpreted as matched graph-only contrasts. The repaired no-bridge driver now restores complete optimizer/RNG state and respects the requested ConvKB budget; run it into a fresh output root. See [KGE execution and verification](docs/kge-execution.md) and the [repair status](reproducibility/STATUS.txt).
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

### 3. Controlled PhenoQuery initialization and seed comparison

Follow the [controlled experiment instructions](docs/controlled-query.md). The
portable driver prepares immutable configurations, checks GPU pilots, freezes
all 240 validation selections, and then scores selected and initial checkpoints.

### 4. Native and adapted baselines

See [native baseline commands and input layouts](docs/native-baselines.md).
For the UPheno adaptations, `WORKSPACE` contains `data/foldN` and the preserved
baseline resources:

```bash
python code/baselines/run_matched_phrank_phenorank.py \
  --base-dir WORKSPACE --out-root WORKSPACE/results/upheno-baselines --workers 2
```

### 5. Stepwise Bias Evaluation (`S0`–`S4`) and Table Generation
```bash
python code/analysis/evaluate_stepwise_bias.py \
  --base-dir WORKSPACE \
  --eq-dir WORKSPACE/results/phenoquery \
  --out-json reproducibility/results/stepwise-bias-report-v4.json

python reproducibility/generate_bias_tables.py \
  --report-json reproducibility/results/stepwise-bias-report-v4.json \
  --out-dir reproducibility/tables
```

---

## Running the Test Suite

```bash
.venv/bin/python -m pytest -q tests
```

---

## Repository Layout

- `reproducibility/`: CLI dispatcher (`run_pipeline.sh`), table and metric generators (`generate_bias_tables.py`, `revision_metrics.py`, `supporting_tables.py`, `structural_controls.py`), and precomputed 10-fold JSON evaluation results (`results/`).
- `code/training/`: `TransD` (`kge_transd.py`), `ConvKB-D` (`kge_convkb_d.py`), candidate-pool / profile hard-negative support in `code/link_gda/head_experiment.py`, and disease-bridge-free KGE retraining (`retrain_nobridge_kge.py`).
- `code/link_gda/`: Shared evaluation, calibration (`calibration.py`), candidate-pool selection (`candidates.py`), negative sampling (`negative_sampling.py`), profile ranking (`profile_ranking.py`), and data utilities.
- `code/baselines/`: `ULTRA` and `UltraQuery` drivers (`prepare_ultra_data.py`, `score_ultra.py`, `run_ultra_matched.py`), `PhenoQuery` driver (`run_phenoquery.py`, `eq_model_reference.py`), matched `Phrank` and `PhenoRank` refitting (`run_matched_phrank_phenorank.py`), `Exomiser` evaluation (`exomiser_eval.py`), and semantic similarity scripts.
- `code/projector/`: `GDAProjector` Scala source, compiler, and independent ontology projection (`project_independent.py`).
- `code/analysis/`: Sequential and individual-factor sensitivities (`evaluate_stepwise_bias.py`), statistical tests (`rq1_stats.py`, `rq2_stats.py`), overlap stratification, and popularity controls.
- `code/figures/`: Calibration schematic and rank-CDF figure generators.
- `tests/`: Automated unit, layout, and reproducibility regression tests.
- `docs/`: [Revision protocol](docs/revision-20261013.md), [reproduction guide](docs/reproduction.md), [code layout map](docs/code_layout.md), and [testing guide](docs/testing.md).
