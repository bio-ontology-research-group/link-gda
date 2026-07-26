# link-gda

> *Overview / abstract: TODO, to be written once the rest of the document is settled.*

## Background

This repository extends INDIGENA (Zhapa-Camacho & Hoehndorf,
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
   model, not a generic embedding similarity.
2. **The gene side is multi-modal.** It does not need gene phenotypes; any
   subset of MP phenotypes, GO functions, and UBERON expression sites works,
   reconstructed via the corresponding inverse relations
   (`has_phenotype`, `has_function`, `expressed_in`). The disease side stays
   anchored on HPO phenotypes; that is the inductive bottleneck.
3. **The KG is built with a GDA-aware OWL2Vec* projection** of the UPheno
   + annotation axioms: standard SubClassOf/equivalence triples plus
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

## Excluded-gene benchmark

The main benchmark keeps only pairs whose gene carries at least one MGI-propagated
phenotype annotation (`build_association_files.py`, the `genes_with_phenotypes`
check). That filter exists so every method can score every candidate: the
semantic-similarity measures, Exomiser-Phive and INDIGENA all match phenotype sets,
and a gene with no phenotype profile receives no score from any of them.

The filter has a side effect. Genes whose mouse orthologs lack phenotype annotations
are the population that motivates the phenotype-free variants, and the filter removes
them, so `LinkGDA-f` measures a simulation of that setting rather than the setting
itself: annotations are withheld from genes that have them. The two populations may
differ in a direction that flatters the method, because a gene with MGI knockout
phenotypes is a well-studied gene and can carry richer, more specific GO annotations
than a gene nobody has characterized.

`build_excluded_benchmark.py` builds the complementary benchmark from the pairs the
filter discards, so the claim can be measured rather than extrapolated:

```bash
python build_excluded_benchmark.py --data-dir data --out-dir ../link-gda-excluded
```

A pair enters when its gene has no MGI phenotype but does carry GO function
annotations, and its disease has HPO phenotypes. Two leakage filters then apply,
because the model represents a query disease only by its HPO phenotype set:

1. **Identifier overlap.** The script drops diseases that also appear in the training
   pairs.
2. **Profile overlap.** A disease whose phenotype set closely matches a training
   disease's is the same query under a different identifier, so the script also drops
   diseases whose maximum Jaccard similarity to any training disease reaches
   `--jaccard-threshold`. The default of 0.5 is a choice rather than a standard, and
   the script reports the count at several thresholds so the sensitivity stays visible.

Construction, on the data release described above:

| step | pairs |
|------|-------|
| all pairs in `genes_to_disease.txt` | 15,782 |
| main benchmark, after both filters | 6,571 |
| gene has no MGI phenotype | 1,366 |
| and gene has GO functions, disease has HPO | 532 |
| and disease identifier not in training | 477 |
| and phenotype profile not a near-duplicate | **409** |

The test set holds 409 pairs over 350 genes and 402 diseases, and the candidate pool
grows from 4,399 to 4,749 so every true gene stays rankable. Mean ranks are therefore
computed over a slightly larger pool than the main benchmark and are not directly
comparable to it. Of the identifier-disjoint candidates, 1.9% have an exact duplicate
profile in training and the median maximum Jaccard is 0.267, so the set is largely
novel to the model. `disease_leakage.csv` records the nearest training disease for
every test disease, and `funnel.txt` records the counts above.

The script writes `train.csv` (the 6,571 main-benchmark pairs) and `test.csv` (the 409
excluded pairs) into `data/folds/fold_0/`, and symlinks the annotation CSVs and
projected edge lists rather than copying them. Every path `kge_transd.py` reads is
relative to the working directory, so the new benchmark needs no change to the
training code: run the trainer from the output directory and it picks up the new data.

### Ten-seed campaign

Reviewers reasonably ask whether an effect exceeds initialization variance, and the
main benchmark reports across-fold standard deviations from a single seed. The
excluded benchmark uses one held-out test set rather than cross-validation, because 409
pairs would split into folds too small to inform anything. It spends that compute on
seeds instead:

```bash
cd ../link-gda-excluded
bash ../link-gda/run_excluded_seeds.sh
```

The runner trains ten seeds each of `LinkGDA-f` and `LinkGDA-fs` (on separate GPUs if
available) and reports mean and standard deviation across seeds. It inherits the
hyperparameters from the main benchmark's grid search (dimension 100, batch size
16,384, learning rate 0.001; Supplementary Table 3) and does **not** retune them. If we tuned on a
held-out set we would spend the property that makes it worth running, and a result
obtained at hyperparameters selected elsewhere is the stronger claim.

Each run writes its own log to `logs/train_{f,fs}_seed{N}.log` and its per-instance
ranks to `data/results/`, so we recompute metrics from disk rather than read them from
stdout. `analyze_excluded_seeds.py` produces the aggregate, and
`diagnose_early_stopping.py` reports validation noise, the correlation between
validation and test mean rank, and the effect of stopping time.

### What a seed varies, and two flags that control it

`--random_seed` drives three things: weight initialization, negative sampling, and,
through `create_train_val_split`, which diseases are held out for validation. That
third one matters, because the split partitions by disease, so a multi-seed run
resamples the training data as well as the initialization and the resulting spread
mixes the two.

`--val_seed` fixes the split independently of `--random_seed`. Set it to hold the
validation diseases constant so that seeds vary only initialization and negative
sampling. It defaults to `--random_seed`, which reproduces the original behaviour, so
existing invocations are unaffected. Note that a fixed-split spread and a varying-split
spread answer different questions and should not be quoted side by side without saying
which is which.

`--tolerance` sets the early-stopping patience, in validation evaluations of 20 epochs
each, and defaults to 5, the value previously hard-coded. Validation mean rank is noisy
on a split this size, so too little patience can end a run that is still improving.

Early stopping keeps the checkpoint with the best validation mean rank and
`kge_transd.py` reloads it before testing, so a short run is not the same as an
undertrained model: every run is evaluated at its own validation optimum.

### Running the seed campaign on a cluster

`run_excluded_seeds.sh` is the portable entry point; wrap it in whatever submission
script your scheduler uses. Two things to get right on any cluster:

- **Build the dataset once and point every task at that same copy**, so all seeds (and,
  for the tolerance sensitivity check below, both arms) evaluate an identical test set.
- **Give each `--tolerance` value its own working directory.** `file_identifier` in
  `kge_transd.py` encodes fold, seed, dimensions, batch size, learning rate, modality,
  projector and `use_graph`, but **not** tolerance, so two tolerance settings sharing a
  directory overwrite each other's checkpoints and result files without warning. Run
  each in its own `tol<N>/` directory (symlinking the shared inputs in) and compare them
  with `compare_tolerance_arms.py`.

## Environment

The reported numbers were produced with **Python 3.11.14** and the exact package
versions pinned in `requirements.txt` (`mowl-borg` 1.0.3, `torch` 2.10.0, `pykeen`
1.11.1, `wandb` 0.24.2, `numpy` 2.4.2, `scipy` 1.15.3, `pandas` 3.0.0). Recreate the
environment from the committed files:

```bash
conda env create -f environment.yml   # creates env `link-gda`, Python 3.11 + requirements.txt
conda activate link-gda
# then run any script below as `python <script>.py ...`
```

The `environment.yml` env is named `link-gda`. Recommended non-interactive invocation:
`conda run -n link-gda --no-capture-output python ...`.

**Weights & Biases is optional.** The training scripts run with W&B disabled when no
`config.toml` is present, which is all that is needed to reproduce the reported numbers
(hyperparameters are passed on the command line). W&B is required only to run the
hyperparameter *sweeps*, or if you want standalone runs to log to your account: copy
`config.toml.example` to `config.toml` and set your own `entity`/`project`.

Non-Python toolchains (only needed for the projector and the baselines, not for the
KGE pipeline itself):

- Scala 2.11.12 (to align with mOWL) for the OWL2Vec*-GDA projector.
- Groovy + slib-sml 0.9.1 (auto-resolved via `@Grab`) for the semantic-similarity
  baselines.
- Java 17+ for Exomiser (tested with OpenJDK 21).

## Reproducing the paper's tables and figures

Every reported number is recomputed from saved per-instance rank files on disk, not
from a training run's stdout (see *Experiment hygiene* in `data/results/`). Once the
pipeline below has produced those files, each table/figure maps to one script:

| Paper artifact                                   | Script                                                        |
|--------------------------------------------------|---------------------------------------------------------------|
| RQ1/RQ2/RQ3 main metric tables (mean ± std)      | `aggregated_sem_sim_metrics.py`, `wandb_scripts/extract_metrics_from_folds.py` |
| Fold-level significance tests (headline p-values)| `p_value_per_fold.py`                                         |
| Nadeau–Bengio corrected p-values (RQ1)           | `dump_perfold_vectors.py` → `data/perfold_vectors.json` → `nb_corrected_ttest.R` |
| Overlap-stratified tables (memorization, Table 2)| `leakage_overlap_perfold.py` (KGE), `sem_sim_overlap.py` (baselines), rows via `gen_overlap_tables.py` |
| Rank-CDF figures                                 | `rank_cdf_median.py` → `make_rankcdf_fig.py` (writes `paper/fig/`) |
| Excluded-gene benchmark (Table 4, 10 seeds)      | `build_excluded_benchmark.py` → `run_excluded_seeds.sh` → `analyze_excluded_seeds.py` |
| Selection-inflation / clean-fold check (RQ1)     | `clean_folds_check.py`                                         |
| Seed-variance controls                           | `variance_decomposition.py`, `diagnose_early_stopping.py`, `compare_tolerance_arms.py` |
| Leakage / data-provenance controls               | `check_data_leakage.py`, `leakage_overlap.py`, `leakage_overlap_verify.py`, `popularity_controls.py` |

The pooled-vs-fold-level significance distinction and the corrected test are detailed
under *Significance testing* below; the analysis/control scripts are cataloged under
*Analysis and verification scripts*.

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
| `--use_graph`  | `evaluate_by_graph`: the link-prediction scoring rule described above. |
| *(omitted)*    | `evaluate_by_similarity`: the INDIGENA-style similarity evaluation.    |

The link-prediction evaluation is used for every method variant reported in
this work. The choice is recorded in both the checkpoint filename
(`use_graph_{True,False}`) and the result filename (`by_graph_*` vs
`inductive_*`).

### TransD

Modality flags (additive; pick any combination):

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
across the 10 folds, including the headline numbers cited in the
paper, are produced by the `wandb_scripts/extract_metrics_*` helpers
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
  per modality to locate
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

#### Run sweep agents

Run an agent for a registered sweep ID directly with W&B, with no scheduler needed. Each
agent pulls one configuration from the grid, runs `python kge_transd.py` (or
`kge_convkb_d.py`) with those hyperparameters, and logs the result:

```bash
wandb agent <entity>/<project>/<sweep_id>      # sweep IDs are in wandb_scripts/sweep_ids.yaml
```

Launch as many agents as you want parallelism, on any machine (or across a cluster with
your own submission script). To reproduce a single fold *without* sweeps, call the
trainer directly with the winning hyperparameters from Supplementary Table 3:

```bash
python kge_transd.py --fold 0 --use_phenotypes --use_functions --use_site \
    --projector_name owl2vecstar_gda --embedding_dim 100 --batch_size 32768 \
    --learning_rate 0.001 --use_graph --no_sweep
```

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

The paper reports **fold-level** paired tests, and there are two scripts with a
deliberate difference in the unit of analysis:

- **`p_value_per_fold.py`, the headline test used in the paper.** Each of the 10
  cross-validation folds is collapsed to one number (that method's mean rank in the
  fold), giving 10 paired observations, and a single paired *t*-test / Wilcoxon runs
  over those 10 points. The 10 folds are the unit of (approximate) independence. This
  is the conservative version: with only 10 points a genuinely tiny effect may
  not reach significance, which is the correct outcome. Supports `--two-sided` and the
  RQ3 projector comparisons.

  ```bash
  python p_value_per_fold.py
  ```

- **`p_value.py`, pooled, kept for contrast.** Pools all ~6,571 `(fold, disease,
  gene)` instances and treats them as independent paired samples. They are *not*
  independent (the same gene recurs across diseases, and every instance in a fold is
  scored by the same trained model), so this over-states significance (a 52.7% win
  rate reaching p<1e-14). Reported only to show why the fold-level test is the right
  one.

In both, `METHODS` maps a label to `(architecture, config)` where `config` is the part
of the result filename between `seed_0_` and `.tsv`, so TransD and ConvKB-D variants
can both be declared; `COMPARISONS` lists the pairs (each phenotype-bearing variant vs
the INDIGENA baseline, and `-f` vs `-p`). A comparison whose result files are missing is
skipped with a message rather than raising.

### Fold-correlation correction (Nadeau–Bengio)

The 10 folds share ~80% of their training data, so a standard paired *t*-test over
folds still under-estimates the variance. For the RQ1 headline comparisons the paper
applies the **Nadeau–Bengio corrected resampled *t*-test** via the R package
`correctR` (`kfold_ttest`), in two steps:

```bash
# 1. Dump the per-fold mean-rank vectors (10 floats per method) from the per-instance
#    rank TSVs, so the correction needs only these small aggregates, not the raw ranks.
python dump_perfold_vectors.py            # writes data/perfold_vectors.json

# 2. Run the tests in R: naive t-test, textbook NB formula, correctR::kfold_ttest,
#    and a sign test, all from the committed JSON. Requires R with the correctR package.
Rscript nb_corrected_ttest.R
```

`data/perfold_vectors.json` is committed, so step 2 reproduces the corrected p-values on
its own; step 1 only needs re-running if the per-instance ranks change. The correction
only inflates the variance (it can lower a comparison's significance, never raise it);
the R script cross-checks `correctR` against the textbook NB formula, and they agree.

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
