# link-gda

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
   the TransD scoring function `f(h, r, t) = -‖ h + r - t ‖²` of the trained
   model — not a generic embedding similarity.
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
`has_function`, or `expressed_in`), and `s_KGE(t, r, p)` is the TransD
negative squared distance in the trained model's geometry:

```
s_KGE(t, r, p) = σ( -‖ emb(t) + emb(r⁻¹) - emb(p) ‖² )
```

`emb(r⁻¹)` is the inverse-relation embedding, so `emb(t) + emb(r⁻¹)`
is the gene-side reconstruction of feature `t`, and the score matches
the L2 training objective.

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
#    The standard OWL2Vec* edge lists (upheno_edges.tsv, go_edges.tsv,
#    uberon_edges.tsv) are written on first kge_transd.py invocation.
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

# 6d. ConvKB-D (warm-starts from the TransD checkpoints written in 6a)
for fold in $(seq 0 9); do
  python kge_convkb_d.py --fold $fold \
      --use_phenotypes --use_functions --use_site --no_sweep
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
  `conda run -n link-gda --no-capture-output python ...`
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

## Knowledge graph embedding

Two embedding architectures are trained on the supervised graph *Graph 4*
from the INDIGENA paper (UPheno + gene–phenotype + disease–phenotype +
known `associated_with` GDAs for the training-fold diseases): **TransD**
(`kge_transd.py`), the main model, and **ConvKB-D** (`kge_convkb_d.py`), a
secondary architecture evaluated alongside it. The other graph variants
(G1–G3, and the transductive G3T/G4T) are kept in the codebase for
reproducing INDIGENA results but are *not* used for the headline numbers
here; cleanup is tracked as a follow-up.

Both scripts take **`--use_graph`**, which selects the evaluation:

| Flag           | Evaluation                                                             |
|----------------|------------------------------------------------------------------------|
| `--use_graph`  | `evaluate_by_graph` — the link-prediction scoring rule described above. |
| *(omitted)*    | `evaluate_by_similarity` — the INDIGENA-style similarity evaluation.    |

The link-prediction evaluation is used for every method variant reported in
this work. The choice is recorded in both the checkpoint filename
(`use_graph_{True,False}`) and the result filename (`by_graph_*` vs
`inductive_*`).

### TransD

Modality flags (additive — pick any combination):

| Flag                | Adds to the gene side                                |
|---------------------|------------------------------------------------------|
| `--use_phenotypes`  | gene → MP phenotype links (`has_phenotype`)          |
| `--use_functions`   | gene → GO term links (`has_function`)                |
| `--use_site`        | gene → UBERON expression-site links (`expressed_in`) |

The flags are also what define the *phenotype-free* settings: **omitting
`--use_phenotypes` withholds the gene's mouse-ortholog (MP) phenotype
annotations**, so the gene side is reconstructed from function and/or
expression alone. Those are the RQ2 variants (`-f`, `-s`, `-fs` in the
paper); the disease side stays anchored on HPO phenotypes either way, and
the evaluation gene/disease set is unchanged. No extra flag is needed.

Other relevant flags (see `python kge_transd.py --help`):
`--fold`, `--embedding_dim`, `--batch_size`, `--learning_rate`,
`--random_seed`, `--only_test`, `--no_sweep`, `--projector_name`.

Headline configuration (phenotypes + functions + expression, fold 0,
no W&B sweep):

```bash
python kge_transd.py --fold 0 \
    --use_phenotypes --use_functions --use_site --no_sweep
```

Per-fold raw scores are written to `data/results/`. Aggregated metrics
across the 10 folds — including the headline numbers cited in the
paper — are produced by the `wandb_scripts/extract_metrics_*` helpers
once a W&B sweep has finished (see below).

### ConvKB-D

ConvKB-D replaces TransD's translational scoring function with a
convolutional network over the stacked head/relation/tail embeddings. It is
**warm-started from a pretrained TransD checkpoint**: the entity and
relation embeddings are copied in and then fine-tuned jointly with the
convolutional filters.

Two consequences follow, and both are easy to trip over:

- **`--embedding_dim` is not a flag.** The dimension is inherited from the
  TransD checkpoint, because the pretrained embeddings are copied straight
  in and the dimensions must match. `model_resolver` in `kge_convkb_d.py`
  hardcodes the checkpoint coordinates `(dim, batch_size, learning_rate)`
  per modality in order to locate
  `data/models/transd_fold_{fold}_..._{source}_proj_{projector}...pt`.
- **The matching TransD run must exist first.** ConvKB-D for a given
  (modality, projector) requires the TransD checkpoint for that same
  (modality, projector); the script raises `FileNotFoundError` if it is
  absent. If you retune TransD, update the coordinates in `model_resolver`
  to match the filenames `kge_transd.py` now writes.

The modality and projector flags are identical to TransD's. ConvKB-D's own
hyperparameters stay on the CLI: `--num_filters`, `--hidden_dropout_rate`,
`--batch_size`, `--learning_rate`, `--tolerance`.

```bash
# ConvKB-D, phenotypes + functions + expression, fold 0
python kge_convkb_d.py --fold 0 \
    --use_phenotypes --use_functions --use_site \
    --num_filters 200 --no_sweep
```

### Hyperparameter sweeps and 10-fold runs (W&B)

All HPO and 10-fold runs are driven by W&B sweeps. The sweep definitions
live under `sweeps/`, are registered in batch by
`wandb_scripts/create_sweeps.py`, and their assigned IDs are persisted
in `wandb_scripts/sweep_ids.yaml` (a registry, not a generated artefact).

The active sweep groups in `create_sweeps.py` are:

| Group key              | Model    | Stage          | What it does                                                                                     |
|------------------------|----------|----------------|--------------------------------------------------------------------------------------------------|
| `hpo_rq1`              | TransD   | HPO (legacy)   | Initial single-split HPO grid; superseded.                                                       |
| `hpo_rq1_cv3`          | TransD   | HPO (legacy)   | First 3-fold-CV HPO grid; superseded.                                                            |
| `hpo_rq1_cv3_v2`       | TransD   | HPO (current)  | 3-fold-CV HPO over {dim, lr, projector}; **winners feed the paper**.                             |
| `folds_rq1_cv3_bs32k`  | TransD   | 10-fold (final)| Paper-bound 10-fold evaluation at bs=32 768, lr=1e-3, dim = `hpo_rq1_cv3_v2` winner per setting. |
| `hpo_rq2_cv3`          | TransD   | HPO            | 3-fold-CV HPO for the phenotype-free variants.                                                   |
| `folds_rq2`            | TransD   | 10-fold (final)| Paper-bound 10-fold evaluation of the phenotype-free variants.                                   |
| `hpo_convkbd_rq1_cv3`  | ConvKB-D | HPO            | 3-fold-CV HPO over {filters, lr, batch size, projector} for the phenotype-bearing variants.      |
| `hpo_convkbd_rq2_cv3`  | ConvKB-D | HPO            | The same grid for the phenotype-free variants.                                                   |
| `folds_convkbd_rq1`    | ConvKB-D | 10-fold (final)| Paper-bound 10-fold evaluation, phenotype-bearing variants.                                      |
| `folds_convkbd_rq2`    | ConvKB-D | 10-fold (final)| Paper-bound 10-fold evaluation, phenotype-free variants.                                         |

In the RQ2 and ConvKB-D groups the 3-fold HPO sweeps are projector-specific:
each (variant, projector) was registered as its own sweep, so their keys carry
a projector suffix (`no_pheno_owl2vecstar`, `no_pheno_gda`) just as the 10-fold
keys do. The RQ1 3-fold groups instead use one sweep per variant, spanning both
projectors.

Each group is a dict mapping a friendly setting name (`all_gda`,
`only_pheno_owl2vecstar`, ...) to a sweep YAML under `sweeps/`. The
naming convention is

```
sweeps/hpo_kge_transd[_<variant>]_folds_<projector>.yml      # 10-fold runs
sweeps/hpo_kge_transd[_<variant>]_cv3_v2.yml                 # HPO grids
```

with `<variant>` ∈ {ø (all features), `no_func`, `no_site`,
`no_func_no_site`, `indigena`} and `<projector>` ∈ {`owl2vecstar`,
`owl2vecstar_gda`}.

#### Register sweeps

```bash
python wandb_scripts/create_sweeps.py
```

Re-running is safe: entries already in `sweep_ids.yaml` are skipped, so
the script only creates sweeps that are new.

#### Launch agents on IBEX

```bash
# Submit one sbatch array per sweep ID in a group
./submit_sweeps.sh hpo_rq1_cv3_v2  0-17       # 3 folds × 3 dims × 2 lrs = 18 cells
./submit_sweeps.sh folds_rq1_cv3_bs32k 0-9    # 10 folds per (setting, projector)
```

Each array task runs `run_sweep.sh <sweep_id>`, which loads the
`link-gda` conda env and calls `wandb agent --count 1` on that ID.

Right-size the array range to the grid (surplus agents print a
harmless "Sweep is not running" warning but cost nothing).

#### Aggregate

```bash
python wandb_scripts/extract_metrics_from_folds.py
```

reports mean ± std across folds per (setting, projector).

#### Archived sweep YAMLs

Older fold-sweep variants (v1, retry, v2 at bs=16k, no_site diagnostic
sweeps) have been moved into `sweeps/archive/`, which is gitignored.
They stay preserved locally for forensic reference but are not part of
the active pipeline.

## Significance testing

`p_value.py` runs the paired significance tests reported in the paper. Test
instances `(fold, disease, gene)` are paired across two methods and pooled
over the 10 folds; for each comparison it reports mean/median rank, the
per-instance win rate, a paired *t*-test and a Wilcoxon signed-rank test.

```bash
python p_value.py
```

`METHODS` maps a label to `(architecture, config)`, where `config` is the part
of the result filename between `seed_0_` and `.tsv`, so TransD and ConvKB-D
methods can both be declared. `COMPARISONS` lists the pairs to test: each
phenotype-bearing variant against the INDIGENA baseline, and `-f` against `-p`
under both architectures. A comparison whose result files have not been
generated is skipped with a message rather than raising.

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
