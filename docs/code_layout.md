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

The moved scripts that import repository-root modules (`rq1_table`,
`evaluate_sem_sim`, `evaluation`) add the repository root to `sys.path` from
their own location, so they can be launched by absolute path
from any working directory (for example the excluded-benchmark directory) with
no change to how their data and output paths resolve. `leakage_overlap.py`
imports the root `data.py` the same way; the scripts that import
`leakage_overlap` (`make_overlap_labels.py`, `leakage_overlap_perfold.py`)
resolve it as a same-directory sibling.

`tests/test_analysis_layout.py` pins the move: each moved Click/argparse CLI must answer
`--help` from the repository root and from a foreign working directory, and the
excluded-table tools must reproduce the same numbers from small fixtures
regardless of the working directory. `tests/test_data_analysis_layout.py` pins
the batch-2 move the same way, and additionally checks that `generate_folds.py`
still writes a deterministic, disease-disjoint 10-fold split from a small
fixture. The two CLI-less analysis scripts (`sem_sim_overlap.py`,
`leakage_overlap_verify.py`) were checked for unchanged calculations during the move; their full
analyses were not rerun.

## Still at the root

Everything else is unchanged: `rq1_table.py`, `calibrate_scores.py`,
`analysis/rq1_stats.py` and `analysis/rq2_stats.py`, `graph_statistics.py`, the
trainers and shared modules, `project_ontologies.py` and the Ultra/Exomiser
tools, `make_calibration_fig.py`, and the remaining baseline and scoring tools.

The rank-CDF pair retains older GDAProjector configurations and embedded plot
values. Its move preserves historical behavior; it does not verify those values
against the current manuscript. Regenerate and verify the data before using it
for current paper figures. The test suite renders both PDFs in a temporary
directory; it does not overwrite manuscript figures.

`leakage_overlap_verify.py` also retains older hardcoded model configurations.
Its relocation preserves that diagnostic; it does not make its outputs current
manuscript results.
