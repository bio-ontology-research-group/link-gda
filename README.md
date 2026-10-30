# PhenoQuery: phenotype-query learning and evaluation biases in gene–disease association prediction

PhenoQuery ranks candidate genes for an unseen disease from its phenotype set.
It combines ULTRA's relational graph network with UltraQuery's conjunctive
query formulation and typed Entity–Quality links extracted from phenotype
definitions. The repository also evaluates how training-association status,
phenotype overlap, calibration and repeated profiles affect gene ranking.

The [paper and supplement](manuscript/README.md) include the complete controlled
experiments, baseline comparisons and bias analyses. The
[reproduction guide](docs/reproduction.md) maps each result to its inputs,
environment and executable workflow. The [verification status](reproducibility/STATUS.txt)
identifies completed checks and outstanding independent-inference and release work.

**Data access update, 30 October 2026:** temporary bulk Release attachments have
been removed from GitHub at the authors' request. Verified server copies and
small manifests are retained. Archive download commands require those copies
until a replacement public archive is published. The committed table-generation
and manuscript-build workflows remain available. See [artifact storage and
access](docs/artifact-storage.md).

## Main experiments

- **Controlled phenotype queries:** four graph/readout arms, ULTRA and UltraQuery
  initialization, three seeds and ten folds. All 240 validation selections are
  frozen before outer-test inference for the 480 initial/selected checkpoints. Adding typed EQ
  links to conjunction improves calibrated Hits@10 by 1.55 percentage points
  under ULTRA initialization and 2.74 under UltraQuery initialization; both
  multiplicity-adjusted intervals exclude zero. See the
  [results and intervals](manuscript/tables/controlled-query-contrasts.tex) and
  [training protocol](reproducibility/protocols/controlled-query-20261028.md).
- **Bias analyses:** per-gene calibration, within-query candidate-status
  adjustment, phenotype-overlap strata, reference deduplication, co-causal
  filtering and profile weighting. Results retain the scoring rule and evaluation
  population for each comparison. Native Phrank and PhenoRank rankings are
  preserved alongside additional transformations. See the
  [query bias workflow](docs/controlled-query.md#bias-sensitivities-for-every-controlled-variant)
  and [baseline methods](docs/native-baselines.md).
- **Matched embedding comparators:** U1, T1, INDIGENA, ConvKB-D and P-pool,
  each trained with and without disease bridges under fixed within-family
  configurations. Ten folds and one seed give 100 training runs and 400 retained
  selections. Graph changes have different effects on mean rank and top-ten
  retrieval. See the [matched results](manuscript/tables/matched-kge-primary.tex)
  and [execution instructions](docs/kge-execution.md).

The benchmark was reused during method development. Confidence intervals
describe benchmark sampling variation conditional on the fitted models;
training-seed dispersion is reported separately for the query comparison.

## Start with the result tables

The following commands use Python's standard library and the committed,
verified production reports. Run them from the repository root with a new
output directory:

```bash
python3 reproducibility/generate_controlled_query_tables.py \
  --report-json reproducibility/results/controlled-query-report.json \
  --out-dir generated-results/query
python3 reproducibility/generate_matched_kge_tables.py \
  --report-json reproducibility/results/matched-kge-summary.json \
  --out-dir generated-results/matched
```

These regenerate three query tables and two matched-model tables in
[`manuscript/tables/`](manuscript/tables/). This is report-to-table reproduction.
For recomputation from predictions, follow the saved-score workflows below.

## Choose a reproduction workflow

| Task | Instructions |
|---|---|
| Recompute controlled-query statistics and bias results from retained scores | [Controlled-query score replay](docs/controlled-query-score-replay.md) |
| Generate query predictions from archived weights | [Controlled-query checkpoints](docs/controlled-query-checkpoints.md) |
| Train the controlled-query campaign | [Controlled queries](docs/controlled-query.md) |
| Recompute matched-model statistics and bias results | [Matched KGE score replay](docs/matched-kge-score-replay.md) |
| Generate matched-model predictions from archived weights | [Matched KGE checkpoints](docs/matched-kge-checkpoints.md) |
| Refit Phrank/PhenoRank or replay their saved results | [Baseline execution](docs/native-baselines.md), [baseline result replay](docs/final-baseline-results.md) |
| Recompute supporting embedding and excluded-cohort results | [Embedding archive replay](docs/embedding-archive-replay.md) |
| Replay supporting graph/query sensitivity experiments | [Historical comparator replay](docs/historical-comparator-replay.md) |
| Rebuild the manuscript PDFs and illustrations | [Manuscript build](manuscript/README.md) |

The controlled-query saved-score collection has passed a fresh public download
and full replay: 480 bundles, eight primary contrasts, 48 variants across 42
conditions, and 6,720 exact rank arrays. Matched native saved-score and checkpoint
verification has passed. The [matched score collection](https://github.com/bio-ontology-research-group/link-gda/releases/tag/matched-kge-scores-20261029-v1)
and [checkpoint collection](https://github.com/bio-ontology-research-group/link-gda/releases/tag/matched-kge-checkpoints-20261029-v1)
were published and verified before bulk-asset withdrawal. Fresh matched-score
download and full numerical replay passed all
40-by-42 conditions, 5,600 exact rank arrays, ten contrasts and primary/bias
exports. The incomplete public checkpoint download was stopped for the storage
change. Complete PhenoQuery
public-checkpoint inference and downstream verification have passed. All 100 matched runs have completed independent inference from retained
weights. All 400 fresh outputs have transferred. The complete destination CPU
numerical comparison remains in progress. See the [status record](reproducibility/STATUS.txt) for
the evidence supporting each stage.

## Inputs and environments

Use the [frozen inputs](docs/frozen-inputs.md) for the paper's prepared tables,
folds, ontologies and pretrained checkpoints. Reproduction starts from those
retained inputs because some original annotation-provider snapshots are
unavailable. Downloading current provider resources constructs a new dataset.

The computational stages use separate pinned environments:

| Stage | Environment |
|---|---|
| Saved-score analysis and figures | [CPU analysis, Python 3.13.5](reproducibility/environments/cpu-analysis/README.md) |
| ULTRA/UltraQuery training and inference | [Archived query environment](docs/frozen-inputs.md#reconstruct-the-ultra-environment) |
| Matched embedding training and inference | [Matched KGE Pixi environment](reproducibility/environments/matched-kge/README.md) |
| CPU regression suite | [Testing setup](docs/testing.md) |

Archive workflows check downloaded bytes, extracted files and scientific
identities. Use fresh output directories and project-local scratch. The
[artifact recipes](docs/reproduction.md) specify storage requirements and
which source revision belongs to each released experiment.

## Software and prior methods

The main drivers are `code/baselines/run_controlled_query.py` and
`code/training/run_matched_kge.py`. Shared scoring and evaluation code is under
`code/link_gda/` and `code/analysis/`; artifact downloaders and table generators
are under `code/reproduce/` and `reproducibility/`.

PhenoQuery builds on ULTRA and UltraQuery. Phrank uses conditional phenotype
information content, and PhenoRank uses simulated queries to address
annotation-related effects. Their author implementations, citations, resource
choices and adaptations are documented in [native baselines](docs/native-baselines.md)
and the paper. Retained third-party components carry their upstream license notices.

For the method and evaluation design, cite the working manuscript by Fernando
Zhapa-Camacho, Naeema Muhammed and Robert Hoehndorf, with the title above. Record
the code commit and artifact release tags used for an analysis.
