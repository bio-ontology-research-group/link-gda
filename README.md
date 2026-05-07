# multihop-gda

> *Overview / abstract: TODO — to be written once the rest of the document is settled.*

## Background

This repository extends [INDIGENA](indigena.pdf) (Zhapa-Camacho & Hoehndorf,
*INDIGENA: inductive prediction of disease–gene associations using phenotype
ontologies*).

INDIGENA lifted the classical Resnik / Lin / SimGIC semantic-similarity
comparison of two phenotype sets to a latent-space comparison: pairwise
phenotype similarities `sim^e(p_g_i, p_d_j) = σ(<emb(p_g_i), emb(p_d_j)>)`
were aggregated with BMA into a single GDA score. The setting is *inductive*:
the test disease's phenotype set may be unseen at training time, but the
phenotype terms themselves come from the (fixed) UPheno ontology.

This repo goes one step further:

1. **Scoring is link-prediction-based, not similarity-based.** The pairwise
   score between a gene-side feature and a disease phenotype is computed via
   the KGE scoring function `f(h, r, t)` of the trained model (TransE-style
   dot product with an inverse-relation offset, or TransD-style negative
   squared distance) — not a generic embedding similarity.
2. **The gene side is multi-modal.** It does not need gene phenotypes; any
   subset of MP phenotypes, GO functions, and UBERON expression sites works,
   reconstructed via the corresponding inverse relations
   (`has_phenotype`, `has_function`, `expressed_in`). The disease side stays
   anchored on HPO phenotypes — that is the inductive bottleneck.
3. **The KG is built with a GDA-aware OWL2Vec* projection** of the UPheno
   + annotation axioms — standard SubClassOf/equivalence triples plus
   phenotype → GO/UBERON edges extracted from nested ObjectSomeValuesFrom
   axioms, for both HP and MP phenotypes (see
   `projector/.../OWL2VecStarGDAProjector.scala`).

## Scoring rule

For a candidate gene `g` annotated with feature set `F_g ⊆ {phenotypes,
functions, expression_sites}` and a query disease `d` with HPO phenotype set
`P_d`, the GDA score is a Best-Match-Average over the pairwise KGE scores:

```
score(g, d) = BMA_{(t, p) ∈ F_g × P_d}  s_KGE(t, r_t, p)
```

where `r_t` is the relation linking `g` to `t` in the KG (`has_phenotype`,
`has_function`, or `expressed_in`), and `s_KGE(t, r, p)` is computed in the
trained model's geometry — concretely:

- **TransE-like:** `σ( <emb(t) + emb(r⁻¹), emb(p)> )`
- **TransD-like:** `σ( -‖ emb(t) + emb(r⁻¹) - emb(p) ‖² )` (matches the
  L2 training objective)

`emb(r⁻¹)` is the inverse-relation embedding, so `emb(t) + emb(r⁻¹)`
is the gene-side reconstruction of feature `t` (for TransE: from `h + r ≈ t`
we have `t - r ≈ h`).

BMM (Best-Match-Maximum) is also computed for comparison with the classical
semantic-similarity literature. See `compare_vectorized` in `evaluation.py`.

## Inductive split

The 10 cross-validation folds are split over **diseases**, not over individual
gene–disease pairs:

- 90% of OMIM diseases are kept in the training graph, together with their
  HPO phenotype links and their known `(disease) -- associated_with --> (gene)`
  triples (the supervised signal, *Graph 4* in the INDIGENA paper).
- 10% are held out. At test time the model sees only the held-out disease's
  HPO phenotypes; it has never seen any `associated_with` triple involving
  that disease.

Concretely each fold lives in `data/folds/fold_{0..9}/`:

| File                | Format         | Columns       |
|---------------------|----------------|---------------|
| `train.csv`         | TSV (header)   | `Gene`, `Disease` |
| `test.csv`          | TSV (header)   | `Gene`, `Disease` |
| `test_no_leakage.csv` | CSV (header) | post-leakage-check subset (see `check_data_leakage.py`) |

Train and test disease sets are disjoint by construction
(`kge_transd.py` re-asserts this on every run).

## Pipeline (reproducible from scratch)

The full pipeline, from raw downloads to per-fold metrics:

```bash
# 1. Download raw association sources (MGI, HPO, GO, UPheno, GTEx)
python download_data.py

# 2. Build the per-task association CSVs
#    (gene_phenotypes.csv, disease_phenotypes.csv, gene_functions.csv,
#     gene_diseases.csv, gene_site.csv, etc.)
python build_association_files.py

# 3. Generate the 10 disease-disjoint folds under data/folds/fold_{0..9}/
python generate_folds.py

# 4. Compile the OWL2Vec*-GDA Scala projector into build/OWL2VecStarGDAProjector.jar
./compile_projector.sh

# 5. Project UPheno into the phenotype edge list (data/upheno_edges_gda.tsv)
#    GO and UBERON edge lists are written on first kge_transd.py invocation
#    via the standard OWL2VecStarProjector.
python project_ontologies.py

# 6a. KGE training + evaluation (TransD, all modalities, all 10 folds)
for fold in $(seq 0 9); do
  python kge_transd.py --fold $fold \
      --use_phenotypes --use_functions --use_site --no_sweep
done

# 6b. Semantic-similarity baselines (5 measures × 10 folds, in parallel)
./run_all_sem_sim.sh

# 6c. Exomiser phenotype-only baselines
for fold in $(seq 0 9); do
  python exomiser_eval.py --fold $fold
done

# 7. Aggregate per-fold results into mean ± std
python aggregated_sem_sim_metrics.py -pw resnik -gw bma
python aggregated_sem_sim_metrics.py -pw resnik -gw bmm
python aggregated_sem_sim_metrics.py -pw lin    -gw bma
python aggregated_sem_sim_metrics.py -pw lin    -gw bmm
python aggregated_sem_sim_metrics.py            -gw simgic
```

## Dependencies

- Python (see `environment.yml` / `requirements.txt`); recommended invocation:
  `conda run -n multihopgda --no-capture-output python ...`
- Scala 2.11.12 (to align with mOWL) for the projector
- Groovy + slib-sml 0.9.1 (auto-resolved via `@Grab`) for the
  semantic-similarity baselines
- Java 17+ for Exomiser (tested with OpenJDK 21)

## Semantic-similarity baselines

Five hand-crafted phenotype-similarity baselines run via slib-sml on the
UPheno ontology (information content always corpus-Resnik over the
gene-/disease-phenotype annotations):

| Pairwise   | Groupwise | Driver script                       | Output filename suffix                   |
|------------|-----------|-------------------------------------|------------------------------------------|
| Resnik     | BMA       | `semantic_similarity.groovy`        | `resnik_resnik_bma_fold{N}_results.txt`  |
| Resnik     | BMM       | `semantic_similarity.groovy`        | `resnik_resnik_bmm_fold{N}_results.txt`  |
| Lin        | BMA       | `semantic_similarity.groovy`        | `resnik_lin_bma_fold{N}_results.txt`     |
| Lin        | BMM       | `semantic_similarity.groovy`        | `resnik_lin_bmm_fold{N}_results.txt`     |
| —          | SimGIC    | `semantic_similarity_simgic.groovy` | `resnik_simgic_fold{N}_results.txt`      |

Filenames follow the pattern `<IC>_<pairwise>_<groupwise>_fold<N>_results.txt`
(SimGIC has no pairwise component).

Run a single configuration manually:

```bash
# Resnik-BMA, fold 0
groovy semantic_similarity.groovy -r data -ic resnik -pw resnik -gw bma -fold 0

# SimGIC, fold 0
groovy semantic_similarity_simgic.groovy -r data -ic resnik -fold 0
```

Run all 5 measures × 10 folds in parallel (5 concurrent groovy processes,
each iterating folds 0..9 sequentially):

```bash
./run_all_sem_sim.sh
```

Per-run logs are written to `logs/sem_sim/<measure>_fold<N>.log`; per-measure
master logs (start/end timestamps for each fold) are at
`logs/sem_sim/<measure>.master.log`. Per-fold raw scores go to
`data/baseline_results/`.

Aggregate to mean ± std MR / MRR / Hits@{1,3,10,100} / AUC across the 10 folds:

```bash
python aggregated_sem_sim_metrics.py -pw resnik -gw bma
python aggregated_sem_sim_metrics.py -pw resnik -gw bmm
python aggregated_sem_sim_metrics.py -pw lin    -gw bma
python aggregated_sem_sim_metrics.py -pw lin    -gw bmm
python aggregated_sem_sim_metrics.py            -gw simgic
```

## Knowledge graph embedding (TransD)

The main embedding model is **TransD** trained on the supervised graph
*Graph 4* from the INDIGENA paper (UPheno + gene–phenotype + disease–phenotype
+ known `associated_with` GDAs for the training-fold diseases). The other
graph variants (G1–G3, and the transductive G3T/G4T) are kept in the codebase
for reproducing INDIGENA results but are *not* used for the headline numbers
here; cleanup is tracked as a follow-up.

Modality flags (additive — pick any combination):

| Flag                | Adds to the gene side                                |
|---------------------|------------------------------------------------------|
| `--use_phenotypes`  | gene → MP phenotype links (`has_phenotype`)          |
| `--use_functions`   | gene → GO term links (`has_function`)                |
| `--use_site`        | gene → UBERON expression-site links (`expressed_in`) |

Other relevant flags (see `python kge_transd.py --help`):
`--fold`, `--embedding_dim`, `--batch_size`, `--learning_rate`,
`--random_seed`, `--only_test`, `--no_sweep`, `--projector_name`.

Headline configuration (phenotypes + functions + expression, fold 0,
no W&B sweep):

```bash
python kge_transd.py --fold 0 \
    --use_phenotypes --use_functions --use_site --no_sweep
```

Per-fold raw scores are written to `data/results/`.

### Inductive TransD on Graph 4

10-fold cross-validation, BMA-over-`f(h,r,t)` scoring, increasing modalities.

#### Semantic-similarity baselines

> *Table to be re-filled with mean ± std across the full 10-fold sweep
> launched via `run_all_sem_sim.sh`. The two rows below come from the
> earlier single-fold pilot.*

| Method     | MR      | MRR    | Hits@1 | Hits@3 | Hits@10 | Hits@100 | AUC    |
|------------|---------|--------|--------|--------|---------|----------|--------|
| Resnik-BMA | 1234.83 | 0.0492 | 0.0183 | 0.0496 | 0.1070  | 0.3169   | 0.7201 |
| Resnik-BMM | 1273.48 | 0.0359 | 0.0123 | 0.0334 | 0.0773  | 0.2549   | 0.7113 |

#### TransD results

**Phenotypes only:**

| Method     | MR      | MRR    | Hits@1 | Hits@3 | Hits@10 | Hits@100 | AUC    |
|------------|---------|--------|--------|--------|---------|----------|--------|
| TransD BMA | 1049.24 | 0.0454 | 0.0143 | 0.0426 | 0.1020  | 0.3580   | 0.7623 |
| TransD BMM | 1038.53 | 0.0297 | 0.0094 | 0.0228 | 0.0603  | 0.2870   | 0.7647 |

**Phenotypes + functions:**

| Method     | MR     | MRR    | Hits@1 | Hits@3 | Hits@10 | Hits@100 | AUC    |
|------------|--------|--------|--------|--------|---------|----------|--------|
| TransD BMA | 907.98 | 0.0446 | 0.0166 | 0.0395 | 0.0930  | 0.3355   | 0.7944 |
| TransD BMM | 946.27 | 0.0383 | 0.0129 | 0.0351 | 0.0812  | 0.3067   | 0.7857 |

**Phenotypes + functions + expression:**

| Method     | MR     | MRR    | Hits@1 | Hits@3 | Hits@10 | Hits@100 | AUC    |
|------------|--------|--------|--------|--------|---------|----------|--------|
| TransD BMA | 685.21 | 0.0547 | 0.0211 | 0.0514 | 0.1143  | 0.3913   | 0.8451 |
| TransD BMM | 763.65 | 0.0467 | 0.0181 | 0.0398 | 0.0973  | 0.3502   | 0.8272 |

## Data sources

| Source file              | Provider | Used for                                 |
|--------------------------|----------|------------------------------------------|
| `MGI_GenePheno.rpt`      | MGI      | gene → MP phenotype                      |
| `MGI_Geno_DiseaseDO.rpt` | MGI      | gene → OMIM disease (supervised signal)  |
| `phenotype.hpoa`         | HPO      | disease → HP phenotype                   |
| `goa_human.gaf.gz`       | GO       | gene → GO function (with MGI orthology mapping) |
| `tpmss.tsv`              | GTEx     | gene → UBERON expression site            |
| `upheno.owl`             | UPheno   | cross-species MP↔HP alignment + ontology |

Approximate counts after filtering for terms present in UPheno (from a recent
run of `build_association_files.py`):

| Relation             | Entities            | Pairs   |
|----------------------|---------------------|---------|
| Gene–Phenotype       | 13 626 genes        | 213 988 |
| Disease–Phenotype    | 8 573 diseases      | 164 006 |
| Gene–Disease         | (kept GDA pairs)    | 3 363   |
| Gene–Function        | (genes × GO terms)  | 321 532 |
| Gene–Expression site | (genes × UBERON)    | 576 060 |
| Phenotypes in UPheno | 14 387 MP + 18 546 HP |       |

Final filtered set of `(gene, disease)` pairs with non-empty phenotype
profiles on both sides: **2 476**.

## Compiling the Scala projector

```bash
./compile_projector.sh
```

Outputs `build/OWL2VecStarGDAProjector.jar`. See
`projector/src/main/scala/org/mowl/Projectors/OWL2VecStarGDAProjector.scala`
for the source.

## Exomiser baseline

We compare against [Exomiser](https://github.com/exomiser/Exomiser) using
phenotype-only gene prioritisation (no VCF / variant analysis).

### Setup

- **Version:** Exomiser CLI 14.0.0
- **Phenotype data:** 2406_phenotype
- **Downloaded:** April 12th, 2026 from <https://data.monarchinitiative.org/exomiser/latest>
- **Java requirement:** Java 17+ (tested with OpenJDK 21)

### Installation

```bash
mkdir -p exomiser && cd exomiser

wget https://data.monarchinitiative.org/exomiser/latest/exomiser-cli-14.0.0-distribution.zip
wget https://data.monarchinitiative.org/exomiser/latest/2406_phenotype.zip

unzip exomiser-cli-14.0.0-distribution.zip
unzip 2406_phenotype.zip -d exomiser-cli-14.0.0/data/
```

### Prioritisers used (phenotype-only, no VCF required)

| Prioritiser  | Description                                                      |
|--------------|------------------------------------------------------------------|
| **hiPhive**  | Cross-species phenotypes (human, mouse, fish) + PPI network      |
| **Phive**    | Cross-species phenotypes only (no PPI)                           |
| **PhenIX**   | Human HPO phenotypes only (IC-based semantic similarity)         |

### Running the evaluation

```bash
python exomiser_eval.py --fold 0
```

Runs all three prioritisers on the specified fold and writes
`data/results/exomiser_{prioritiser}_fold_{fold}.tsv`, using the same
metric definitions (MR, MRR, AUC, Hits@K) as the KGE pipeline.
