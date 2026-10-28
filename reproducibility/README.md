# Reproducibility Suite (`S0`–`S4` Debiasing, Retraining, and Baseline Refitting)

This directory contains the command-line workflows, evaluation drivers, table generators, and precomputed 10-fold JSON evaluation reports for:

> **Disentangling structural and training-status biases in inductive gene–disease prioritization across knowledge graphs and graph foundation models**

## Quick Verification (Regenerate All Bias-Correction Tables Immediately)

Precomputed 10-fold evaluation summaries across all 66 model and baseline variants are provided in `reproducibility/results/stepwise-bias-report-v4.json`. To regenerate main-text Table 9 (`revised-bias-stepwise.tex`) and Supplementary Table S13 (`revised-bias-baselines-full.tex`):

```bash
uv run python3 reproducibility/generate_bias_tables.py \
  --report-json reproducibility/results/stepwise-bias-report-v4.json \
  --out-dir reproducibility/tables
```

Or via the master CLI pipeline script:

```bash
./reproducibility/run_pipeline.sh generate-tables
```

---

## Step-by-Step Command-Line Reproduction

### 1. Prepare Input Tables, 10 Disease-Disjoint Folds, and Independent Ontology Projections

```bash
./reproducibility/run_pipeline.sh prepare
```

This runs:
- `code/data/download_data.py`
- `code/data/build_association_files.py` (splitting semicolon-separated Entrez Gene IDs into 18,904 functional genes and 13,625 phenotype-annotated candidate genes)
- `code/data/generate_folds.py` (10 disease-disjoint folds, `6,571` test associations across `6,059` diseases)
- `code/projector/project_independent.py --data-dir data` (independent JVM OWL2Vec* projections of UPheno v2, GO, and UBERON)

### 2. Train Knowledge-Graph Embedding Models (`LinkGDA`, `ConvKB-D`, `INDIGENA`)

#### A. Original Graph (with Training Disease Nodes)
```bash
./reproducibility/run_pipeline.sh train-kge
```
To train candidate-pool negative sampling (`T1`) and profile hard-negative fine-tuning (`P-pool`) on the original graph:
```bash
for fold in $(seq 0 9); do
  uv run python3 code/training/kge_head_experiment.py --fold "$fold" --arm T1
  uv run python3 code/training/kge_head_experiment.py --fold "$fold" --arm P_pool
done
```

#### B. Matched Disease-Bridge-Free Graph (Removing Training Disease Nodes, Bias 1)
Retrains `LinkGDA-U1`, `LinkGDA-T1`, `LinkGDA-Ppool`, `ConvKB-D`, and `INDIGENA` across all 10 folds after removing all training `associated_with` and `has_symptom` edges while preserving materialized `(g, causes_phenotype, p)` edges, gene annotations, and ontology projections:

```bash
for fold in $(seq 0 9); do
  for cond in U1_INDIGENA_PPOOL T1 CONVKB; do
    uv run python3 code/training/retrain_nobridge_kge.py \
      --fold "$fold" \
      --condition "$cond" \
      --data-root data \
      --out-root data/results/nobridge_runs
  done
done
```

### 3. Train and Evaluate Relational Graph Foundation Models (`ULTRA`, `UltraQuery`, `PhenoQuery`)

#### A. Matched `ULTRA` (`ultra_4g`) and `UltraQuery` (`ultraquery.pth`) Arms
```bash
./reproducibility/run_pipeline.sh train-ultra
```

#### B. `PhenoQuery` (Disease-Bridge-Free Graph + Logical EQ Ontology Axioms + Conjunctive Intersection)
Evaluates all 9 `PhenoQuery` arms (`bridge_pool`, `plain_pool`, `plain_and`, `entity_pool`, `entity_and`, `typed_pool`, `typed_and`, `structure_pool`, `structure_and`) across folds `0..9`:
```bash
./reproducibility/run_pipeline.sh train-phenoquery
```

### 4. Matched 10-Fold Refitting of `Phrank` and `PhenoRank` on UPheno v2 + MGI

Refits `Phrank` (`HPO` and `UPheno v2 + MGI`, `disease_max`, `direct`, and `MGI-Only`, both raw and normalized) and `PhenoRank` (`HM-Add+PPI`, `HM-Bal+PPI`, `Mouse+PPI`, and `Mouse-NoPPI`, including 1,000-simulation empirical $p$-values per query length $|P_d|$ with and without RWR tie-breaking, alongside continuous pre-simulation RWR scores) per fold across all 13,625 candidates:

```bash
uv run python3 code/baselines/run_matched_phrank_phenorank.py \
  --base-dir data \
  --out-root data/results/corrected_v3 \
  --folds 0 1 2 3 4 5 6 7 8 9 \
  --workers 10
```

### 5. Run the Five-Stage Stepwise Bias Evaluation (`S0 -> S1 -> S2 -> S3 -> S4`)

Computes:
- **Step 0 (`S0`)**: Raw uncalibrated scores on the full benchmark (`n = 6,571` pairs across `13,625` candidates).
- **Step 1 (`S1`)**: Leave-one-disease-out per-gene reference calibration ($z(g,d) = (\mathrm{score}(g,d)-\mu_g)/\sigma_g$).
- **Step 2 (`S2`)**: Within-query status-stratum standardization (`Status-Z`) equalizing the mean and variance of seen (`3,779.2`) and unseen (`9,845.8`) candidate pools per query, alongside exact within-status mean ranks.
- **Step 3 (`S3`)**: Restriction to the zero-direct-overlap cohort ($J_{\mathrm{same}} = 0$, `n = 4,431` pairs: `945` zero-overlap seen pairs + `3,486` unseen pairs).
- **Step 4 (`S4`)**: Collapse of identical phenotype sets ($P_d = P_{d'}$, `201` diseases in `70` duplicate sets) in the reference panel, removal of `210` exact training-profile duplicates ($J_{\mathrm{any}} = 1$), co-causal gene filtering for `200` multi-gene diseases, and equal weighting across `4,011` unique disease profiles.

```bash
uv run python3 code/analysis/evaluate_stepwise_bias.py \
  --base-dir data \
  --eq-dir data/results/phenoquery \
  --out-json reproducibility/results/stepwise-bias-report-v4.json \
  --workers 4
```

---

## Included Artifacts in `reproducibility/`

- `run_pipeline.sh`: Master CLI entrypoint for all stages.
- `generate_bias_tables.py`: Generates `revised-bias-stepwise.tex` (Table 9) and `revised-bias-baselines-full.tex` (Suppl. Table S13).
- `revision_metrics.py`: Computes per-association and per-disease main benchmark and excluded-cohort metrics.
- `supporting_tables.py`: Generates supplementary calibration, modality, and status tables.
- `structural_controls.py`: Computes query-independent degree, training-frequency, and annotation-depth control rankings.
- `results/stepwise-bias-report-v4.json`: Complete 10-fold stepwise bias evaluation report across all 66 model and baseline variants.
- `results/ultra-matched-report.json`: Matched 10-fold evaluation summary for `ULTRA` and `UltraQuery`.
- `results/ultra-eq-query-report.json`: 10-fold evaluation summary for all 9 `PhenoQuery` arms.
- `results/revised-main-controls.json` & `results/structural-controls.json`: Control and diagnostic summaries.
