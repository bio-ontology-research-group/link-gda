# link-gda

**LinkGDA** reframes gene–disease association (GDA) prediction for rare diseases from
phenotypic similarity to inductive link prediction over a knowledge graph. We train a
TransD model over the UPheno ontology together with gene phenotype, function, and
expression annotations and known gene–disease associations, then score a candidate gene
for a query disease through the trained model's scoring function rather than by generic
embedding similarity. The setting is inductive over diseases: the query disease can be
unseen at training time, and the gene side needs no phenotype annotations, so the method
scores genes that similarity-based prioritizers cannot.

This repository holds the code and data-preparation workflow for the paper *Beyond
similarity: inductive gene–disease associations as link prediction*, and reproduces the
reported tables and figures (see [Reproducing the paper's tables and
figures](#reproducing-the-papers-tables-and-figures)).

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
   `projector/.../OWL2VecStarGDAProjector.scala`). This projector is now
   available upstream as `mowl.projection.GDAProjector` — see
   [Compiling the Scala projector](#compiling-the-scala-projector).

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

# 6a. KGE training + evaluation (TransD, all 10 folds, both settings)
#     --use_graph selects the link-prediction readout, which is LinkGDA;
#     omit it to score the same model by similarity, which is INDIGENA.
#     --dual_arms yields the uncalibrated and calibrated results from one run.
#     Hyperparameters below are the selected ones for -pfs; see the tuning
#     section for how they were chosen.
for fold in $(seq 0 9); do
  python kge_transd.py --fold $fold \
      --use_phenotypes --use_functions --use_site --use_graph \
      --projector_name owl2vecstar \
      --embedding_dim 200 --batch_size 65536 --learning_rate 0.001 \
      --random_seed 0 --tolerance 15 \
      --dual_arms --no_sweep
done

# 6b. Semantic-similarity baselines (5 measures × 10 folds, in parallel)
./run_all_sem_sim.sh

# 6c. Exomiser phenotype-only baselines
for fold in $(seq 0 9); do
  python exomiser_eval.py --fold $fold
done

# 6d. ConvKB-D (warm-starts from the TransD checkpoints written in 6a).
#     The --transd_* coordinates must match the checkpoint from 6a exactly,
#     including the fold: warm-starting one fold from another's weights would
#     leak that fold's training diseases into evaluation.
for fold in $(seq 0 9); do
  python kge_convkb_d.py --fold $fold \
      --use_phenotypes --use_functions --use_site --use_graph \
      --projector_name owl2vecstar \
      --transd_dim 200 --transd_batch 65536 --transd_lr 0.001 \
      --transd_tolerance 15 --transd_arm _calsel \
      --batch_size 32768 --learning_rate 0.0001 \
      --num_filters 100 --hidden_dropout_rate 0.0 \
      --random_seed 0 --tolerance 15 \
      --dual_arms --no_sweep
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
phenotype annotation (`build_association_files.py`), so every method can score every
candidate. `LinkGDA-f` therefore measures the phenotype-free setting by *withholding*
annotations from genes that have them, rather than on genes that genuinely lack them.
`build_excluded_benchmark.py` builds the complementary benchmark from the discarded
pairs, so the claim is measured directly:

```bash
python build_excluded_benchmark.py --data-dir data --out-dir ../link-gda-excluded
```

A pair enters when its gene has no MGI phenotype but carries GO functions and its
disease has HPO phenotypes. Because the model represents a disease only by its HPO
phenotype set, two leakage filters then drop diseases that appear in training
(identifier overlap) or whose phenotype profile is a near-duplicate of a training
disease (max Jaccard >= `--jaccard-threshold`, default 0.5; counts reported at several
thresholds). Construction funnel:

| step | pairs |
|------|-------|
| all pairs in `genes_to_disease.txt` | 15,782 |
| main benchmark, after both filters | 6,571 |
| gene has no MGI phenotype | 1,366 |
| and gene has GO functions, disease has HPO | 532 |
| and disease identifier not in training | 477 |
| and phenotype profile not a near-duplicate | **409** |

The test set holds 409 pairs over 350 genes and 402 diseases; the candidate pool grows
to 4,749, so mean ranks are not directly comparable to the main benchmark's 4,399-gene
pool. `disease_leakage.csv` and `funnel.txt` record the nearest training disease per
test disease and the counts above. The script writes `train.csv`/`test.csv` into
`data/folds/fold_0/` and symlinks the shared inputs, so the trainer runs unchanged from
the output directory.

### Ten-seed campaign

The excluded benchmark uses one held-out test set (409 pairs is too few to fold), so it
reports across-**seed** rather than across-fold variance:

```bash
cd ../link-gda-excluded
bash ../link-gda/run_excluded_seeds.sh
```

This trains ten seeds each of `LinkGDA-f` and `LinkGDA-fs` at the main benchmark's
hyperparameters (dim 100, batch 16,384, lr 0.001; Supplementary Table 3), without
retuning. Each run logs to `logs/` and writes per-instance ranks to `data/results/`;
`analyze_excluded_seeds.py` aggregates across seeds and `diagnose_early_stopping.py`
reports validation noise and selection skill.

Two flags control what a seed varies: `--val_seed` fixes the train/validation disease
split (defaults to `--random_seed`) so seeds vary only initialization and negative
sampling; `--tolerance` sets early-stopping patience in validation evaluations (default
5). Early stopping reloads the best-validation checkpoint before testing, so a short run
is still tested at its own optimum. On a cluster, give each `--tolerance` value its own
`tol<N>/` working directory (the checkpoint filename does not encode tolerance) and
compare with `compare_tolerance_arms.py`.

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

### Two settings: uncalibrated and calibrated

Every method is reported twice, and understanding this is necessary before
running anything.

The trained model carries a near-binary "does this gene appear in the supervised
association edges" signal. It shifts a gene's scores up or down regardless of
which disease is queried, so it inflates ranking without reflecting anything the
model knows about the query. Per-gene calibration removes it: for each candidate
gene we subtract that gene's own baseline, its mean score over the other queries,
and divide by the corresponding spread. What remains is how much a disease raises
a gene above its own norm.

Calibration is applied identically to every method, including the symbolic
baselines, and both settings appear in the results tables. The two settings are
kept internally consistent end to end:

| setting | checkpoint selected on | metrics computed |
|---|---|---|
| uncalibrated | raw validation mean rank | raw scores |
| calibrated | calibrated validation mean rank | calibrated scores |

`--dual_arms` produces both from a single training run: the stopper tracks the
two validation metrics in parallel and keeps a best checkpoint for each, written
as `<identifier>.pt` and `<identifier>_calsel.pt`. The arms therefore share an
initialisation and an optimisation path and differ only in the epoch each one
selected. Without `--dual_arms`, a run produces one arm — raw by default, or
calibrated with `--calibrated_selection`.

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

`--use_graph` selects the evaluation rule rather than the graph contents. With
it, the model is scored by link prediction over `causes_phenotype`, which is
LinkGDA. Without it, the same trained model is scored by embedding similarity,
which is the INDIGENA baseline. Both are trained on the identical graph.

Run control:

| Flag | Meaning |
|---|---|
| `--fold` | which of the ten disease-disjoint folds to evaluate |
| `--random_seed` | seed for training; also the default train/validation split seed |
| `--val_seed` | seed for the train/validation disease split, if it should differ |
| `--init_seed` | seed for the initial embeddings, so seeds can vary only sampling and batch order |
| `--tolerance` | early-stopping patience in validation evaluations, one per 20 epochs |
| `--dual_arms` | keep a best checkpoint for each of the two settings from one run |
| `--calibrated_selection` | single-arm runs: stop on the calibrated metric instead of the raw one |
| `--skip_test` | do not evaluate on test; use for hyperparameter search |
| `--write_baselines` | score the training diseases and write the per-gene calibration vectors |
| `--force_overwrite` | permit replacing an existing result file |
| `--typed_negatives` | corrupt the gene side of `causes_phenotype` from the evaluation candidate pool |
| `--num_negs_per_pos` | negatives per positive; pykeen's default of 1 is weak for a 4,399-way ranking |

Use `--skip_test` for every hyperparameter-search run. Selection is on validation
mean rank, and omitting test evaluation makes selecting on test structurally
impossible rather than merely discouraged.

Hyperparameter search, one configuration, both settings:

```bash
python kge_transd.py --fold 0 \
    --use_phenotypes --use_functions --use_site --use_graph \
    --projector_name owl2vecstar \
    --embedding_dim 200 --batch_size 65536 --learning_rate 0.001 \
    --random_seed 0 --tolerance 15 \
    --dual_arms --skip_test --no_sweep
```

The chosen configuration, evaluated on test:

```bash
python kge_transd.py --fold 0 \
    --use_phenotypes --use_functions --use_site --use_graph \
    --projector_name owl2vecstar \
    --embedding_dim 200 --batch_size 65536 --learning_rate 0.001 \
    --random_seed 0 --tolerance 15 \
    --dual_arms --no_sweep
```

Each such run writes four score files under `data/results/`, one per arm and
aggregation:

```
kge_results_<identifier>_by_graph_bma.tsv
kge_results_<identifier>_by_graph_bmm.tsv
kge_results_<identifier>_calsel_by_graph_bma.tsv
kge_results_<identifier>_calsel_by_graph_bmm.tsv
```

`_calsel` marks the calibrated arm. `by_graph` becomes `inductive` when
`--use_graph` is omitted, so INDIGENA's files carry the other suffix — worth
knowing before globbing for results. The identifier encodes every setting that
changes the model or the readout, so runs that differ in any of them cannot
overwrite one another.

Metrics are computed from these files rather than from anything the training
process reports, which keeps every reported number recomputable:

```bash
python evaluate_sem_sim.py data/results/kge_results_<identifier>_by_graph_bma.tsv
python rq1_table.py --spec <spec>.tsv --reference INDIGENA
```

### ConvKB-D

ConvKB-D replaces TransD's translational scoring function with a
convolutional network over the stacked head/relation/tail embeddings. It is
**warm-started from a pretrained TransD checkpoint**: the entity and
relation embeddings are copied in and then fine-tuned jointly with the
convolutional filters.

Two consequences follow, and both are easy to trip over:

- **`--embedding_dim` is not a flag.** The dimension is inherited from the
  TransD checkpoint, because the pretrained embeddings are copied straight in
  and the dimensions must match. Pass the checkpoint's dimension as
  `--transd_dim`.
- **The matching TransD run must exist first**, for the same fold, seed,
  modality and projector. The script raises `FileNotFoundError` if it is absent.
  Warm-starting a fold from another fold's checkpoint would leak that fold's
  training diseases into evaluation, so the coordinates must match the run being
  extended.

The checkpoint is located from explicit coordinates rather than a hardcoded
table:

| Flag | Meaning |
|---|---|
| `--transd_dim` | embedding dimension of the TransD checkpoint; ConvKB inherits it |
| `--transd_batch` | batch size of that checkpoint |
| `--transd_lr` | learning rate of that checkpoint, as it appears in the filename |
| `--transd_tolerance` | early-stopping tolerance of that checkpoint |
| `--transd_arm` | which arm to start from: empty for raw, `_calsel` for calibrated |

Learning rates enter filenames as Python renders them, so `1e-05` and not
`0.00001`. Passing the wrong spelling produces a path that does not exist.

ConvKB-D's own hyperparameters stay on the CLI: `--num_filters`,
`--hidden_dropout_rate`, `--batch_size`, `--learning_rate`, `--tolerance`. The
two-setting flags behave exactly as they do for TransD.

```bash
python kge_convkb_d.py --fold 0 \
    --use_phenotypes --use_functions --use_site --use_graph \
    --projector_name owl2vecstar \
    --transd_dim 200 --transd_batch 65536 --transd_lr 0.001 \
    --transd_tolerance 15 --transd_arm _calsel \
    --batch_size 32768 --learning_rate 0.0001 \
    --num_filters 100 --hidden_dropout_rate 0.0 \
    --random_seed 0 --tolerance 15 \
    --dual_arms --skip_test --no_sweep
```

### Hyperparameter sweeps and 10-fold runs (W&B)

All HPO and 10-fold runs are driven by W&B sweeps whose definitions live under
`sweeps/`. `wandb_scripts/create_sweeps.py` registers them in batch and records their
IDs in `wandb_scripts/sweep_ids.yaml` (a hand-maintained registry); see those two files
for the full list of groups. The paper-bound groups are the 3-fold-CV HPO grids
(`hpo_rq1_cv3_v2`, `hpo_rq2_cv3`, and the ConvKB-D equivalents) whose winners feed the
final 10-fold evaluations (`folds_rq1_cv3_bs32k`, `folds_rq2`, `folds_convkbd_*`).

```bash
python wandb_scripts/create_sweeps.py                 # register sweeps (skips existing IDs)
wandb agent <entity>/<project>/<sweep_id>             # run an agent; IDs in sweep_ids.yaml
python wandb_scripts/extract_metrics_from_folds.py    # mean ± std across folds per (setting, projector)
```

Each agent pulls one grid configuration, runs `python kge_transd.py` (or
`kge_convkb_d.py`) with those hyperparameters, and logs the result; launch as many as
you want parallelism. To reproduce a single fold *without* sweeps, call the trainer
directly with the winning hyperparameters from Supplementary Table 3:

```bash
python kge_transd.py --fold 0 --use_phenotypes --use_functions --use_site \
    --projector_name owl2vecstar_gda --embedding_dim 100 --batch_size 32768 \
    --learning_rate 0.001 --use_graph --no_sweep
```

Older sweep variants live in `sweeps/archive/` (gitignored), kept locally but not part
of the active workflow.

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

> **Also available in mOWL.** The implementation is kept in this repository for
> reproducibility of the paper. It has also been contributed upstream as
> `mowl.projection.GDAProjector`
> ([bio-ontology-research-group/mowl#142](https://github.com/bio-ontology-research-group/mowl/issues/142)),
> and will be available from the next mOWL release — for now it can only be
> obtained by installing mOWL from the GitHub source:
>
> ```bash
> pip install git+https://github.com/bio-ontology-research-group/mowl
> ```
>
> ```python
> from mowl.projection import GDAProjector
>
> projector = GDAProjector()      # HP/MP sources, GO/UBERON targets by default
> edges = projector.project(ontology)
> ```
>
> It produces the same edges as the local projector, including the composed
> relation names under `http://mowl.borg/`, and additionally exposes the
> source/target IRI prefixes as `source_prefixes` and `target_prefixes`
> parameters.

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
