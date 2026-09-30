# link-gda

**LinkGDA** reframes gene–disease association (GDA) prediction for rare diseases from
phenotypic similarity to inductive link prediction over a knowledge graph. We train a
TransD model over the UPheno ontology together with gene phenotype, function, and
expression annotations and known gene–disease associations, then score a candidate gene
for a query disease through the trained model's scoring function rather than by generic
embedding similarity. The setting is inductive over diseases: the query disease can be
unseen at training time. Phenotype-withheld variants can represent genes through function
or expression edges, with a documented fallback edge for candidates missing the selected
modalities; the separate excluded benchmark tests true genes without MGI phenotypes.

This repository holds the code and data-preparation workflow for the paper *Beyond
similarity: inductive gene–disease associations as link prediction*. The code can rebuild
the benchmark and rerun the methods, but this checkout does not contain the complete set
of processed inputs, checkpoints, and per-instance predictions needed to reproduce every
reported number exactly. See [Reproducing the paper's tables and
figures](#reproducing-the-papers-tables-and-figures) for the artifact requirements.

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

1. **Scoring is link-prediction-based, not similarity-based.** The evaluator
   scores forward `(gene, causes_phenotype, phenotype)` triples with the
   trained model. For TransD, the head and tail embeddings are first projected
   with relation- and entity-specific projection vectors before their
   translated distance is measured; `code/link_gda/evaluation.py` delegates this operation
   to PyKEEN's `predict_hrt`.
2. **The training graph is multi-modal.** MP phenotypes, GO functions, and
   UBERON expression sites provide gene-side edges. Known training
   gene–disease associations and disease HPO profiles are materialized as
   supervised gene-to-phenotype `causes_phenotype` triples. Evaluation then
   ranks genes directly; it does not reconstruct genes from inverse feature
   relations.
3. **The main KG uses the standard OWL2Vec* projection.** The repository also
   contains an exploratory GDAProjector that adds phenotype-to-GO/UBERON edges
   extracted from nested `ObjectSomeValuesFrom` axioms for HP and MP. The paper
   reports that extension separately; it is not the graph used for the headline
   OWL2Vec* results.

## Scoring rule

For each query disease `d` with HPO phenotype set `P_d`, the graph evaluator
scores every candidate gene `g` against each phenotype `p` in `P_d` as a
`(g, causes_phenotype, p)` link. It then aggregates the per-phenotype scores
for that gene:

```
gene_centric(g) = max_{p ∈ P_d} s_KGE(g, causes_phenotype, p)
disease_centric(g) = mean_{p ∈ P_d} s_KGE(g, causes_phenotype, p)
BMA(g, d) = (gene_centric(g) + disease_centric(g)) / 2
```

The default graph readout calls PyKEEN's `predict_hrt`; with TransD this is the
model's trained negative squared distance in the relation-specific geometry.
The query relation is the forward `causes_phenotype` relation; PyKEEN handles
the inverse-triple table internally. BMM uses the maximum of the two aggregate
scores and is retained for comparison with the semantic-similarity baselines.
The graph evaluator is implemented in `code/link_gda/evaluation.py:evaluate_by_graph`.

```
s_KGE(g, causes_phenotype, p) = model.predict_hrt(g, causes_phenotype, p)
```

## Inductive split

The 10 cross-validation folds are split over **diseases**, not over individual
gene–disease pairs:

- 90% of OMIM diseases enter the fold's training/validation pool. A further
  disease-level split reserves 10% of that pool for validation. Validation
  diseases retain their phenotype profiles, while their gene associations and
  derived `causes_phenotype` triples are withheld.
- 10% are held out for testing. Test diseases are absent from the assembled
  training graph; their HPO profiles are supplied only as evaluation queries.

Concretely each fold lives in `data/folds/fold_{0..9}/`:

| File                | Format         | Columns       |
|---------------------|----------------|---------------|
| `train.csv`         | TSV (header)   | `Gene`, `Disease` |
| `test.csv`          | TSV (header)   | `Gene`, `Disease` |
| `test_no_leakage.csv` | CSV (header) | post-leakage-check subset (see `check_data_leakage.py`) |

Train and test disease sets are disjoint by construction
(`code/training/kge_transd.py` re-asserts this on every run).

## GO projection

Prepare the GO edges before launching folds:

```bash
python code/projector/prepare_go_edges.py --data-dir data
```

The default projects `data/upheno.owl`, then `data/go.owl`, using the same
OWL2Vec* projector with bidirectional taxonomy. The projector retains UPheno's
relation mappings for the GO projection. This sequence reproduces the GO edge
file used in the paper, even when `upheno_edges.tsv` already exists. Both trainers
use this default when function annotations are enabled.

For UPheno 2025-10-12 and GO 2026-01-23 with mOWL 1.0.3, the default produces
219,802 rows (187,797 unique triples). An audit reproduced every triple and its
duplicate count in the original February 7 cache. Projecting GO independently
produces 177,413 rows (162,520 unique triples). These counts apply to those
snapshots; current downloads can produce different graphs.

The command writes `data/go_edges.tsv` and `data/go_edges.tsv.metadata.json`, recording the mode,
input hashes and output hash. It checks cached files before reuse. The recorded
mOWL version identifies the generator; reuse depends on matching input and output
hashes, rather than the version installed by a later consumer. The known
paper cache is recognized by its SHA-256; an unrecognized cache without metadata
requires explicit regeneration. Use `--rebuild` only when you intend to replace
the selected cache, and keep a copy of any original experimental inputs.

To generate independent GO edges as an alternative experiment:

```bash
python code/projector/prepare_go_edges.py --data-dir data --go-projection-mode independent
```

Train that alternative with `--go_projection_mode independent` and
`--use_functions`. It uses `data/go_edges_independent.tsv` and distinct result
and checkpoint names with a `_go_independent` suffix. ConvKB-D uses a TransD
checkpoint from the same GO mode.
The default keeps the existing paper-run names. GO-Plus is not an input to either
mode. The ULTRA export launcher retains the paper-default GO mode; the independent
mode is available through the trainers and their explicit graph-export command.
For other ontologies, keep the original projected edge files when reproducing the
paper. This change makes GO preparation explicit; UPheno and UBERON retain their
existing cache behavior.

## Pipeline reconstruction

The following commands rebuild the inputs and run representative evaluations.
They are a pipeline recipe, not an assertion that the exact paper artifacts are
publicly bundled in this checkout.

```bash
# 1. Download raw association sources (MGI, HPO, GO, UPheno, GTEx)
python code/data/download_data.py

# 2. Build the per-task association CSVs
#    (gene_phenotypes.csv, disease_phenotypes.csv, gene_functions.csv,
#     gene_diseases.csv, gene_site.csv, etc.)
python code/data/build_association_files.py

# 3. Generate the 10 disease-disjoint folds under data/folds/fold_{0..9}/
python code/data/generate_folds.py
python code/projector/prepare_go_edges.py --data-dir data

# 4. Optional: compile the exploratory GDAProjector. The main OWL2Vec* pipeline
#    does not require this jar.
./code/projector/compile_projector.sh

# 5. Optional: project UPheno with GDAProjector into data/upheno_edges_gda.tsv.
#    The standard UPheno and UBERON edge lists are generated by TransD
#    when absent; GO preparation is explicit above.
python code/projector/project_ontologies.py

# 6a. KGE training + evaluation (TransD-pfs, all 10 folds)
#     --use_graph selects the link-prediction readout, which is LinkGDA;
#     omit it to score the same model by similarity, which is INDIGENA.
#     The reported uncalibrated and calibrated procedures selected different
#     hyperparameters, so they are separate runs.
for fold in $(seq 0 9); do
  python code/training/kge_transd.py --fold $fold \
      --use_phenotypes --use_functions --use_site --use_graph \
      --projector_name owl2vecstar \
      --embedding_dim 100 --batch_size 16384 --learning_rate 0.001 \
      --random_seed 0 --tolerance 15 \
      --write_baselines --no_sweep

  python code/training/kge_transd.py --fold $fold \
      --use_phenotypes --use_functions --use_site --use_graph \
      --projector_name owl2vecstar \
      --embedding_dim 200 --batch_size 65536 --learning_rate 0.001 \
      --random_seed 0 --tolerance 15 \
      --calibrated_selection --write_baselines --no_sweep
done

# 6b. Semantic-similarity baselines (5 measures × 10 folds, in parallel)
./code/baselines/run_all_sem_sim.sh

# 6c. Exomiser phenotype-only baselines
for fold in $(seq 0 9); do
  python code/baselines/exomiser_eval.py --folds "$fold"
done

# 6d. ConvKB-D warm-starts from each fold's calibrated-selected TransD
#     checkpoint. See the ConvKB-D section for its separately selected raw and
#     calibrated configurations.

# 7. Aggregate per-fold results into mean ± std
python code/analysis/aggregated_sem_sim_metrics.py -pw resnik -gw bma
python code/analysis/aggregated_sem_sim_metrics.py -pw resnik -gw bmm
python code/analysis/aggregated_sem_sim_metrics.py -pw lin    -gw bma
python code/analysis/aggregated_sem_sim_metrics.py -pw lin    -gw bmm
python code/analysis/aggregated_sem_sim_metrics.py            -gw simgic
```

Several URLs in `code/data/download_data.py` are moving provider endpoints (`current_release`,
unversioned ontology PURLs, and current report paths). They reconstruct the workflow at
the provider's current state, not the exact 2026 input snapshot. Exact reproduction
requires archived source versions and checksums, which still need to be provided with
the release.

## Excluded-gene benchmark

The main benchmark keeps only pairs whose gene carries at least one MGI-propagated
phenotype annotation (`code/data/build_association_files.py`), so every method can score every
candidate. `LinkGDA-f` therefore measures the phenotype-free setting by *withholding*
annotations from genes that have them, rather than on genes that genuinely lack them.
`code/data/build_excluded_benchmark.py` builds the complementary benchmark from the discarded
pairs, so the claim is measured directly:

```bash
python code/data/build_excluded_benchmark.py --data-dir data --out-dir ../link-gda-excluded
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

### Ten-seed campaign artifacts

The excluded benchmark uses one held-out test set, so the paper reports variation
across seeds 0--9 rather than folds. Saved predictions from those runs are aggregated
by `code/analysis/excluded_table.py` and `code/analysis/analyze_excluded_seeds.py`.

The archived [excluded-benchmark launcher](../code/archive/excluded_gda/README.md)
uses GDAProjector with dimension 100. It does not
identify the configurations behind the reported OWL2Vec* excluded-benchmark numbers
and must not be used as their reproduction command. The exact selected commands still
need to be recovered from the archived run metadata before a release can claim
end-to-end numerical reproduction.

Two flags control what a new seed campaign varies: `--val_seed` fixes the train/validation disease
split (defaults to `--random_seed`) so seeds vary training randomness rather than the
held-out diseases; `--tolerance` sets early-stopping patience in validation evaluations (default
5). Early stopping reloads the best-validation checkpoint before testing. Non-default
tolerance values are encoded as `_tol_N` in current filenames; separate working
directories remain useful when comparing older artifacts whose names predate that suffix.

## Environment

Use Python 3.11.15 and the single pinned `requirements.txt` for training,
analysis, figures, and tests. The main training-library versions were checked
against the working environment; the remaining resolved Python dependencies are
also pinned. This installation was validated with CPU tests and training-library
imports, including mOWL's JVM integration. It is not a frozen copy of every
historical training environment or a new full training reproduction.

```bash
uv venv --python 3.11.15 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

The [testing guide](testing.md) gives the CPU-wheel installation command. The
`torch==2.10.0` pin accepts its CPU or CUDA build; choose the build for your
hardware. CUDA runtime dependencies are platform-specific. ULTRA remains in
`environment-ultra.yml` because it requires a different PyTorch/NumPy/CUDA stack.

**Weights & Biases is optional.** The training scripts run with W&B disabled when no
`config.toml` is present, which is sufficient for standalone training once the required
data and run coordinates are available. W&B is required only to run the
hyperparameter *sweeps*, or if you want standalone runs to log to your account: copy
`config.toml.example` to `config.toml` and set your own `entity`/`project`.

Non-Python toolchains:

- Java 17+ for the mOWL-based training and projection scripts.
- Scala 2.11.12 (to align with mOWL) for the OWL2Vec*-GDA projector.
- Groovy + slib-sml 0.9.1 (auto-resolved via `@Grab`) for the semantic-similarity
  baselines.
- Java 17+ for Exomiser (tested with OpenJDK 21).

## Reproducing the paper's tables and figures

Reported metrics are computed from saved per-instance score/rank files rather than
training stdout. Exact numerical reproduction therefore requires the processed inputs,
fold files, selected checkpoints, calibration baselines, and prediction files for every
reported run. Only a subset of result metadata and derived summaries is tracked here;
public availability of the complete artifact set has not been verified. Given the
required files, the main analysis entry points are:

| Paper artifact                                   | Script                                                        |
|--------------------------------------------------|---------------------------------------------------------------|
| Fold/seed metric summaries                        | `code/analysis/aggregated_sem_sim_metrics.py`, `code/analysis/wandb/extract_metrics_from_folds.py`, `code/analysis/excluded_table.py` |
| Nadeau–Bengio corrected RQ1 tests                | `code/analysis/rq1_stats.py` on saved result TSVs |
| Nadeau–Bengio corrected RQ2 tests                | `code/analysis/rq2_stats.py` on supplied matched fold mean ranks |
| Phenotype-overlap strata                          | `code/analysis/leakage_overlap_perfold.py` (KGE), `code/analysis/sem_sim_overlap.py` (baselines), rows via `code/analysis/gen_overlap_tables.py` |
| Overlap strata across hosts (one label set, all methods)| `code/analysis/make_overlap_labels.py` → `code/analysis/strata_from_labels.py` → `data/results/strata_all_methods_{graph_dump,train_csv}.tsv` |
| Historical rank-CDF figures (older configurations; verify before reuse) | `code/analysis/rank_cdf_median.py` → `code/figures/make_rankcdf_fig.py` (writes `paper/fig/`) |
| Calibration schematic (Figure 1b)                | `code/figures/make_calibration_fig.py` (illustrative values, writes `paper/fig/`) |
| Excluded-gene benchmark (10 seeds)                | `code/data/build_excluded_benchmark.py`, then archived predictions → `code/analysis/excluded_table.py` / `code/analysis/analyze_excluded_seeds.py` |
| Leakage / data-provenance controls               | `code/analysis/check_data_leakage.py`, `code/analysis/leakage_overlap.py`, `code/analysis/leakage_overlap_verify.py`, `code/analysis/popularity_controls.py` |

Before running figure generators, create their output directory with
`mkdir -p paper/fig` from the repository root. The manuscript repository
is separate and is not required merely to generate these files.

The pooled-vs-fold-level significance distinction and the corrected test are detailed
under *Significance testing* below.

Superseded selection checks, excluded-set diagnostics, and campaign launchers are
preserved in the [historical code archive](../code/archive/README.md).

Two metric implementations currently disagree on AUC: `code/link_gda/evaluate_sem_sim.py` integrates
the empirical cumulative-rank curve with the trapezoid rule, while `code/analysis/rq1_table.py` derives
AUC algebraically from mean rank. Their values can differ beyond rounding. Treat the AUC
definition as unresolved release work and use one implementation consistently when
comparing regenerated tables.

## Semantic-similarity baselines

Five hand-crafted phenotype-similarity baselines run via slib-sml on the
UPheno ontology (information content always corpus-Resnik over the
gene-/disease-phenotype annotations):

| Pairwise   | Groupwise | Driver script                       | Output filename suffix                   |
|------------|-----------|-------------------------------------|------------------------------------------|
| Resnik     | BMA       | `code/baselines/semantic_similarity.groovy`        | `resnik_resnik_bma_fold{N}_results.txt`  |
| Resnik     | BMM       | `code/baselines/semantic_similarity.groovy`        | `resnik_resnik_bmm_fold{N}_results.txt`  |
| Lin        | BMA       | `code/baselines/semantic_similarity.groovy`        | `resnik_lin_bma_fold{N}_results.txt`     |
| Lin        | BMM       | `code/baselines/semantic_similarity.groovy`        | `resnik_lin_bmm_fold{N}_results.txt`     |
| —          | SimGIC    | `code/baselines/semantic_similarity_simgic.groovy` | `resnik_simgic_fold{N}_results.txt`      |

Filenames follow the pattern `<IC>_<pairwise>_<groupwise>_fold<N>_results.txt`
(SimGIC has no pairwise component).

Run a single configuration manually:

```bash
# Resnik-BMA, fold 0
groovy code/baselines/semantic_similarity.groovy -r data -ic resnik -pw resnik -gw bma -fold 0

# SimGIC, fold 0
groovy code/baselines/semantic_similarity_simgic.groovy -r data -ic resnik -fold 0
```

Run all 5 measures × 10 folds in parallel (5 concurrent groovy processes,
each iterating folds 0..9 sequentially):

```bash
./code/baselines/run_all_sem_sim.sh
```

Per-run logs are written to `logs/sem_sim/<measure>_fold<N>.log`; per-measure
master logs (start/end timestamps for each fold) are at
`logs/sem_sim/<measure>.master.log`. Per-fold raw scores go to
`data/baseline_results/`.

Aggregate to mean ± std MR / MRR / Hits@{1,3,10,100} / AUC across the 10 folds:

```bash
python code/analysis/aggregated_sem_sim_metrics.py -pw resnik -gw bma
python code/analysis/aggregated_sem_sim_metrics.py -pw resnik -gw bmm
python code/analysis/aggregated_sem_sim_metrics.py -pw lin    -gw bma
python code/analysis/aggregated_sem_sim_metrics.py -pw lin    -gw bmm
python code/analysis/aggregated_sem_sim_metrics.py            -gw simgic
```

## Knowledge graph embedding

Two embedding architectures are trained on the supervised graph *Graph 4*
from the INDIGENA paper (UPheno + gene–phenotype + disease–phenotype +
known `associated_with` GDAs for the training-fold diseases): **TransD**
(`code/training/kge_transd.py`), the main model, and **ConvKB-D** (`code/training/kge_convkb_d.py`), a
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

Saved analyses include uncalibrated and calibrated settings; the main paper emphasizes
the calibrated setting while the supplement provides the fuller comparison.

The trained model carries a near-binary "does this gene appear in the supervised
association edges" signal. It shifts a gene's scores up or down regardless of
which disease is queried, so it inflates ranking without reflecting anything the
model knows about the query. Per-gene calibration is designed to reduce this offset: for each candidate
gene we subtract that gene's own baseline, its mean score over the other queries,
and divide by the corresponding spread. The calibrated value expresses how far a
disease's score lies above or below that gene's query-wise norm.

The analysis applies the same per-gene transformation to learned and symbolic scores.
For learned models, model selection and evaluation remain aligned within each setting:

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
annotations**. Function and/or expression edges represent most genes; when a selected
modality is absent, the controlled main-benchmark variants retain one eligible phenotype
edge so the gene still receives an embedding. Those are the RQ2 variants (`-f`, `-s`, `-fs` in the
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

For a search cell, `--dual_arms` can evaluate both validation criteria along one
optimization path while `--skip_test` prevents test scoring:

```bash
python code/training/kge_transd.py --fold 0 \
    --use_phenotypes --use_functions --use_site --use_graph \
    --projector_name owl2vecstar \
    --embedding_dim 200 --batch_size 65536 --learning_rate 0.001 \
    --random_seed 0 --tolerance 15 \
    --dual_arms --skip_test --no_sweep
```

The same cell can be evaluated on test by omitting `--skip_test`:

```bash
python code/training/kge_transd.py --fold 0 \
    --use_phenotypes --use_functions --use_site --use_graph \
    --projector_name owl2vecstar \
    --embedding_dim 200 --batch_size 65536 --learning_rate 0.001 \
    --random_seed 0 --tolerance 15 \
    --dual_arms --no_sweep
```

This dual-arm example does not reproduce both reported `LinkGDA-pfs` settings, because
their search selected different hyperparameters. Run each selected configuration as
shown under [Hyperparameter search and 10-fold runs](#hyperparameter-search-and-10-fold-runs).
A dual-arm run writes four score files under `data/results/`, one per arm and aggregation:

```
kge_results_<identifier>_by_graph_bma.tsv
kge_results_<identifier>_by_graph_bmm.tsv
kge_results_<identifier>_calsel_by_graph_bma.tsv
kge_results_<identifier>_calsel_by_graph_bmm.tsv
```

`_calsel` marks the calibrated arm. `by_graph` becomes `inductive` when
`--use_graph` is omitted, so INDIGENA's files carry the other suffix — worth
knowing before globbing for results. The identifier encodes model/readout settings such
as tolerance, initialization seed, negative-sampling mode, and scored relation. It does
not encode every operational choice; in particular, `--val_seed` is absent, so isolate
runs with different validation splits in separate working directories.

Metrics are computed from these files rather than from anything the training
process reports, which keeps every reported number recomputable:

```bash
python code/link_gda/evaluate_sem_sim.py data/results/kge_results_<identifier>_by_graph_bma.tsv
python code/analysis/rq1_table.py --spec <spec>.tsv --reference INDIGENA
```

The spec is a tab-separated file with four columns and no header: label,
kind, raw filename template, and calibrated filename template. The `kind`
field is currently ignored. For example (replace these illustrative paths
with your saved files, retaining `{f}`):

```text
INDIGENA	kge	data/results/indigena_raw_fold{f}.tsv	data/results/indigena_calsel_fold{f}.tsv
LinkGDA	kge	data/results/linkgda_raw_fold{f}.tsv	data/results/linkgda_calsel_fold{f}.tsv
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
`--hidden_dropout_rate`, `--batch_size`, `--learning_rate`, `--tolerance`. Both reported
settings warm-start from the fold-matched, calibrated-selected TransD-pfs checkpoint
(dimension 200, batch 65,536, learning rate 0.001). ConvKB-D then selects its raw and
calibrated procedures separately.

```bash
python code/training/kge_convkb_d.py --fold 0 \
    --use_phenotypes --use_functions --use_site --use_graph \
    --projector_name owl2vecstar \
    --transd_dim 200 --transd_batch 65536 --transd_lr 0.001 \
    --transd_tolerance 15 --transd_arm _calsel \
    --batch_size 65536 --learning_rate 0.001 \
    --num_filters 200 --hidden_dropout_rate 0.0 \
    --random_seed 0 --tolerance 15 \
    --no_sweep

python code/training/kge_convkb_d.py --fold 0 \
    --use_phenotypes --use_functions --use_site --use_graph \
    --projector_name owl2vecstar \
    --transd_dim 200 --transd_batch 65536 --transd_lr 0.001 \
    --transd_tolerance 15 --transd_arm _calsel \
    --batch_size 65536 --learning_rate 0.0001 \
    --num_filters 100 --hidden_dropout_rate 0.0 \
    --random_seed 0 --tolerance 15 \
    --calibrated_selection --no_sweep
```

### Hyperparameter search and 10-fold runs

The current TransD search evaluates every combination of dimension
`{100, 200, 400, 800}`, batch size `{16384, 32768, 65536, 131072}`, and learning
rate `{0.0001, 0.001, 0.01}` on fold 0 at seed 0. It selects uncalibrated and
calibrated checkpoints independently by their corresponding validation mean rank.
Use `--skip_test` during search. The reported ten-fold evaluation then fixes the
selected configuration and uses seed 0 on folds 0--9.

The `sweeps/` directory also contains older three-fold and GDAProjector YAML files.
Their names do not identify the final paper workflow. New search runs can be launched
directly with `code/training/kge_transd.py` or registered with W&B, but the archived sweep registry is
not by itself provenance for the reported results.

The reported OWL2Vec* `LinkGDA-pfs` selections are:

```bash
python code/training/kge_transd.py --fold 0 --use_phenotypes --use_functions --use_site \
    --projector_name owl2vecstar --embedding_dim 100 --batch_size 16384 \
    --learning_rate 0.001 --tolerance 15 --use_graph --no_sweep

python code/training/kge_transd.py --fold 0 --use_phenotypes --use_functions --use_site \
    --projector_name owl2vecstar --embedding_dim 200 --batch_size 65536 \
    --learning_rate 0.001 --tolerance 15 --use_graph \
    --calibrated_selection --no_sweep
```

The phenotype-withheld `-f`, `-s`, and `-fs` variants inherit the `-pfs`
selections. `LinkGDA-p`, `-ps`, `-pf`, and INDIGENA have their own selected
configurations; consult the supplementary hyperparameter table rather than reusing the
example above.

## Significance testing

The reported RQ1 tests use ten fold-level mean-rank observations, deterministic
average-rank ties, the Nadeau–Bengio correction factor `1/10 + 1/9`, and a
one-sided alternative for INDIGENA minus LinkGDA-pfs. Recompute them from the
saved per-instance result TSVs with:

```bash
python code/analysis/rq1_stats.py \
    --results-dir /path/to/data/results \
    > /path/to/rq1_stats_results.json
```

The script checks query-key and true-gene alignment, records SHA-256 hashes and
file dimensions in its JSON manifest, and computes calibrated ranks directly
from the score matrices. It outputs the raw corrected p-value; it does not apply
multiple-testing adjustment. The manuscript's RQ1 raw p-value is
`0.004267755316762704`. Treating the one RQ1 and five supplied RQ2 tests as a six-test
family gives the reported Bonferroni-adjusted RQ1 value `0.025606531900576227`
(`0.0256`). Confirm that these six comparisons are the intended complete family before
release. The project review archive contains the independent verification record; it is
not required by this calculator.

The older pooled and uncorrected fold-level tests are preserved in the
[statistics archive](../code/archive/statistics/README.md). They use historical
configurations and tie handling and do not reproduce the reported p-values.

The five RQ2 comparisons are two-sided: LinkGDA-f versus LinkGDA-p, LinkGDA-fs
versus LinkGDA-p, LinkGDA-s versus ULTRA-s, LinkGDA-f versus ULTRA-f, and
LinkGDA-fs versus ULTRA-fs. This does not change the directional RQ1 test above.
Run the RQ2 calculator with an explicit JSON input:

```bash
python code/analysis/rq2_stats.py \
    --input /path/to/rq2_matched_fold_mean_ranks.json \
    > /path/to/rq2_stats_results.json
```

The input must contain non-empty provenance metadata and exactly the five named
comparisons. Each side is a map from fold IDs `"0"` through `"9"` to finite
mean ranks:

```json
{
  "metadata": {
    "source": "description of the current fold summaries",
    "paired_cases_verified_by": "description of the alignment check"
  },
  "comparisons": {
    "f_vs_p": {"left": {"0": 0.0}, "right": {"0": 0.0}},
    "fs_vs_p": {"left": {"0": 0.0}, "right": {"0": 0.0}},
    "s_vs_ultra_s": {"left": {"0": 0.0}, "right": {"0": 0.0}},
    "f_vs_ultra_f": {"left": {"0": 0.0}, "right": {"0": 0.0}},
    "fs_vs_ultra_fs": {"left": {"0": 0.0}, "right": {"0": 0.0}}
  }
}
```

The abbreviated maps illustrate the schema; every map must contain all ten fold
IDs. The calculator validates fold alignment at this summary level, records the
input SHA-256 and metadata, applies `1/10 + 1/9`, computes
`2 * scipy.stats.t.sf(abs(t), 9)`, and then applies `min(1, 6 * p)` before any
rounding. Fold means alone cannot prove that both methods used the same underlying
cases, so that alignment must be established by the supplied provenance.

Two manuscript values were converted from coauthor-supplied rounded one-sided
values: `0.0035` to `0.007` two-sided and `0.042` adjusted, and `0.0017` to
`0.0034` two-sided and `0.0204` adjusted. These are conversions of the supplied
rounded values, not independent full-precision recomputations. The three ULTRA
comparisons were already two-sided. Confirm all displayed values from current
full-precision output or matched fold vectors when those inputs are available;
archived vectors are not current provenance.

### Fold-correlation correction (Nadeau–Bengio)

The ten folds share training data, so the correction inflates the variance of
the paired fold differences. The current Python script applies the textbook
factor directly. The older vector extractor and R cross-check are in the
[statistics archive](../code/archive/statistics/README.md); their historical
`data/perfold_vectors.json` remains in place. The R `correctR` package is needed
only for that archived cross-check and is not included in the main environment.

## Data sources

| Source file              | Provider | Used for                                 |
|--------------------------|----------|------------------------------------------|
| `MGI_GenePheno.rpt`      | MGI      | gene → MP phenotype                      |
| `genes_to_disease.txt` | HPO      | human gene → OMIM disease (supervised signal)  |
| `phenotype.hpoa`         | HPO      | disease → HP phenotype                   |
| `goa_human.gaf.gz`       | GO       | gene → GO function (with MGI orthology mapping) |
| `tpmss.tsv`              | GTEx     | gene → UBERON expression site            |
| `upheno.owl`             | UPheno   | cross-species MP↔HP alignment + ontology |

The current benchmark contains 6,571 test associations pooled across the ten
disease-disjoint folds and ranks 4,399 candidate genes. The excluded benchmark contains
409 associations over 350 true genes and 402 diseases, ranked against a 4,749-gene
candidate pool. Source-table counts below are intermediate data-preparation diagnostics,
not the final benchmark size:

| Relation             | Entities            | Pairs   |
|----------------------|---------------------|---------|
| Gene–Phenotype       | 13 626 genes        | 213 988 |
| Disease–Phenotype    | 8 573 diseases      | 164 006 |
| Gene–Disease         | source rows before final benchmark filtering | 15,782 |
| Gene–Function        | (genes × GO terms)  | 321 532 |
| Gene–Expression site | (genes × UBERON)    | 576 060 |
| Phenotypes in UPheno | 14 387 MP + 18 546 HP |       |

## Compiling the Scala projector

The local Scala source is the version documented by this repository. It implements the
exploratory GDAProjector graph; the main reported graph uses standard OWL2Vec*. This
README does not rely on an unverified future mOWL release or installation URL.

```bash
# Override MOWL_LIB_DIR if mOWL is installed outside the default conda env.
./code/projector/compile_projector.sh
```

The script derives the checkout and build paths from its own location. Set
`MOWL_LIB_DIR` to the directory containing mOWL's jars when needed; set
`BUILD_DIR` or `JAR_OUT` to choose a different output location. It outputs
`build/OWL2VecStarGDAProjector.jar` by default. See
`code/projector/src/main/scala/org/mowl/Projectors/OWL2VecStarGDAProjector.scala`
for the source.

## Exomiser baseline

We compare against [Exomiser](https://github.com/exomiser/Exomiser) using
phenotype-only gene prioritisation (no VCF / variant analysis).

### Setup

- **Version:** Exomiser CLI 14.0.0
- **Phenotype data:** 2406_phenotype
- **Archived download date:** April 12, 2026
- **Java requirement:** Java 17+ (tested with OpenJDK 21)

### Installation

```bash
mkdir -p exomiser && cd exomiser

# Supply the archived Exomiser 14.0.0 distribution and 2406_phenotype bundle.
# The moving /latest endpoint is not a version pin and may return different files.

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
python code/baselines/exomiser_eval.py --folds 0
```

Runs all three prioritisers on the specified fold and writes
`data/results/exomiser_{prioritiser}_fold_{fold}.tsv`, using the same
metric definitions (MR, MRR, AUC, Hits@K) as the KGE pipeline.
