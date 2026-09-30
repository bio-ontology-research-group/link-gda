# Code layout

The active tools are moving out of the repository root into `code/` in bounded
batches. This map records where each tool lives and where it moved from. The
root keeps the training, ontology-projection, and baseline-scoring pipeline
until its own batch; the migration never changes data/output paths, which remain
resolved from the caller's working directory.

## Moved so far

Batch 1 — results analysis and reporting, from the root into `code/analysis/`:

| New location | Old location (root) |
|---|---|
| `code/analysis/aggregated_sem_sim_metrics.py` | `aggregated_sem_sim_metrics.py` |
| `code/analysis/analyze_excluded_seeds.py` | `analyze_excluded_seeds.py` |
| `code/analysis/check_data_leakage.py` | `check_data_leakage.py` |
| `code/analysis/compare_calibration_panels.py` | `compare_calibration_panels.py` |
| `code/analysis/excluded_table.py` | `excluded_table.py` |
| `code/analysis/gen_overlap_tables.py` | `gen_overlap_tables.py` |
| `code/analysis/make_overlap_labels.py` | `make_overlap_labels.py` |
| `code/analysis/rank_cdf_median.py` | `rank_cdf_median.py` |
| `code/analysis/ultra_excluded_table.py` | `ultra_excluded_table.py` |
| `code/analysis/ultra_metrics.py` | `ultra_metrics.py` |
| `code/figures/make_rankcdf_fig.py` | `make_rankcdf_fig.py` |

Batch 2 — overlap-stratified analysis and the data-preparation pipeline, from
the root:

| New location | Old location (root) |
|---|---|
| `code/analysis/leakage_overlap.py` | `leakage_overlap.py` |
| `code/analysis/leakage_overlap_perfold.py` | `leakage_overlap_perfold.py` |
| `code/analysis/leakage_overlap_verify.py` | `leakage_overlap_verify.py` |
| `code/analysis/popularity_controls.py` | `popularity_controls.py` |
| `code/analysis/sem_sim_overlap.py` | `sem_sim_overlap.py` |
| `code/analysis/strata_from_labels.py` | `strata_from_labels.py` |
| `code/analysis/stratified_metrics.py` | `stratified_metrics.py` |
| `code/data/build_association_files.py` | `build_association_files.py` |
| `code/data/build_excluded_benchmark.py` | `build_excluded_benchmark.py` |
| `code/data/download_data.py` | `download_data.py` |
| `code/data/generate_folds.py` | `generate_folds.py` |

Batch 3 — the trainers and the shared analysis/scoring modules, into the new
`link_gda` and `analysis` packages and a new `code/training/` directory
(`code/` itself deliberately has no `__init__.py`):

| New location | Old location |
|---|---|
| `code/link_gda/data.py` | `data.py` |
| `code/link_gda/evaluation.py` | `evaluation.py` |
| `code/link_gda/evaluate_sem_sim.py` | `evaluate_sem_sim.py` |
| `code/link_gda/pykeen_utils.py` | `pykeen_utils.py` |
| `code/link_gda/negative_sampling.py` | `negative_sampling.py` |
| `code/analysis/rq1_table.py` | `rq1_table.py` |
| `code/analysis/calibrate_scores.py` | `calibrate_scores.py` |
| `code/analysis/graph_statistics.py` | `graph_statistics.py` |
| `code/analysis/rq1_stats.py` | `analysis/rq1_stats.py` |
| `code/analysis/rq2_stats.py` | `analysis/rq2_stats.py` |
| `code/training/kge_transd.py` | `kge_transd.py` |
| `code/training/kge_convkb_d.py` | `kge_convkb_d.py` |

The moved scripts that import moved modules (`rq1_table`, `evaluate_sem_sim`,
`evaluation`, `data`) add the `code/` directory to `sys.path` from their own
location and import the `link_gda` and `analysis` packages, so they can be
launched by absolute path from any working directory (for example the
excluded-benchmark directory) with no change to how their data and output
paths resolve. The trainers in `code/training/` add `code/` the same way to
import `link_gda`; the root `exomiser_eval.py` adds `code/` to import
`link_gda.evaluate_sem_sim`. The scripts that import `leakage_overlap`
(`make_overlap_labels.py`, `leakage_overlap_perfold.py`) resolve it as a
same-directory sibling.

`tests/test_analysis_layout.py` pins the move: each moved Click/argparse CLI must answer
`--help` from the repository root and from a foreign working directory, and the
excluded-table tools must reproduce the same numbers from small fixtures
regardless of the working directory. `tests/test_data_analysis_layout.py` pins
the batch-2 move the same way, and additionally checks that `generate_folds.py`
still writes a deterministic, disease-disjoint 10-fold split from a small
fixture. The two CLI-less analysis scripts (`sem_sim_overlap.py`,
`leakage_overlap_verify.py`) were checked for unchanged calculations during the move; their full
analyses were not rerun. `tests/test_training_layout.py` pins the batch-3
move the same way: both trainers answer `--help` (which imports the full
`link_gda` stack but does not train) from the repository root and from a
foreign working directory, and the analysis CLIs answer `--help` from a
foreign working directory.

## Still at the root

Everything else is unchanged: `project_ontologies.py`, the Ultra/Exomiser
tools (`exomiser_eval.py`, `score_ultra.py`, `prepare_ultra_data.py`, the Ultra
shell scripts), `make_calibration_fig.py`, `compile_projector.sh`, and the
remaining baseline and scoring tools.

The rank-CDF pair retains older GDAProjector configurations and embedded plot
values. Its move preserves historical behavior; it does not verify those values
against the current manuscript. Regenerate and verify the data before using it
for current paper figures. The test suite renders both PDFs in a temporary
directory; it does not overwrite manuscript figures.

`leakage_overlap_verify.py` also retains older hardcoded model configurations.
Its relocation preserves that diagnostic; it does not make its outputs current
manuscript results.
