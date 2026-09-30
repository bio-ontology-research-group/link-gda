# Code layout

Active tools live under `code/`. This map records their current and former
locations. Data and output paths retain each tool's existing conventions,
described below.

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
import `link_gda`; `code/baselines/exomiser_eval.py` adds `code/` to import
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

Batch 4 — the external-baseline, ontology-projection, and W&B-extraction
tools, from the root and `projector/`/`wandb_scripts/`:

| New location | Old location |
|---|---|
| `code/baselines/exomiser_eval.py` | `exomiser_eval.py` |
| `code/baselines/prepare_ultra_data.py` | `prepare_ultra_data.py` |
| `code/baselines/score_ultra.py` | `score_ultra.py` |
| `code/baselines/run_all_sem_sim.sh` | `run_all_sem_sim.sh` |
| `code/baselines/run_ultra_export.sh` | `run_ultra_export.sh` |
| `code/baselines/run_ultra_score.sh` | `run_ultra_score.sh` |
| `code/baselines/setup_ultra_env.sh` | `setup_ultra_env.sh` |
| `code/baselines/validate_ultra_env.sh` | `validate_ultra_env.sh` |
| `code/baselines/semantic_similarity.groovy` | `semantic_similarity.groovy` |
| `code/baselines/semantic_similarity_simgic.groovy` | `semantic_similarity_simgic.groovy` |
| `code/projector/compile_projector.sh` | `compile_projector.sh` |
| `code/projector/project_ontologies.py` | `project_ontologies.py` |
| `code/projector/src/main/scala/org/mowl/Projectors/OWL2VecStarGDAProjector.scala` | `projector/src/main/scala/org/mowl/Projectors/OWL2VecStarGDAProjector.scala` |
| `code/analysis/wandb/extract_metrics_from_folds.py` | `wandb_scripts/extract_metrics_from_folds.py` |
| `code/analysis/wandb/extract_metrics_from_sweep.py` | `wandb_scripts/extract_metrics_from_sweep.py` |
| `code/analysis/wandb/extract_metrics_from_sweep_per_projector.py` | `wandb_scripts/extract_metrics_from_sweep_per_projector.py` |
| `code/analysis/wandb/best_config_from_sweep.py` | `wandb_scripts/best_config_from_sweep.py` |
| `code/analysis/wandb/best_config_cv.py` | `wandb_scripts/best_config_cv.py` |
| `code/analysis/wandb/sweep_ids.yaml` | `wandb_scripts/sweep_ids.yaml` |

Path-awareness preserved by the move. The Exomiser resources stay at
`exomiser/exomiser-cli-14.0.0` under the repository root, and
`exomiser_eval.py` reaches them from `code/baselines/`; its data and result
paths remain caller-relative. `project_ontologies.py` keeps its
input/output/build-JAR paths caller-relative; only its build-command error
message moved. `compile_projector.sh` determines the repository root as
`scriptdir/../..`, keeps the default build at `root/build`, and compiles the
source now at `code/projector/src/...`; the `BUILD_DIR`/`JAR_OUT`/
`MOWL_LIB_DIR`/`PYTHON_BIN` overrides are unchanged. `run_all_sem_sim.sh`
cds to the repository root (not `code/baselines/`) and launches the Groovy
drivers as `code/baselines/semantic_similarity*.groovy`; the `-r data`
argument and all log/result paths are unchanged. The ULTRA launchers
reference the drivers as `$SCRATCH/code/baselines/prepare_ultra_data.py` and
`$SCRATCH/code/baselines/score_ultra.py`; the trainer and metrics paths
already pointed into `code/`. `setup_ultra_env.sh` locates
`environment-ultra.yml` at the repository root (the file stays there). The
W&B helpers derive the repository root from `parents[3]` instead of
`parents[1]`, and `sweep_ids.yaml` remains a sibling in
`code/analysis/wandb/`. The two W&B sweep readers keep their historical
caller-relative `config.toml` reads unchanged; they select historical sweep
configs, so they are not the current numerical provenance. In particular,
`extract_metrics_from_sweep_per_projector.py` reads `../config.toml`: launch it
from a directory directly below the repository root, such as `code/`, using
`python analysis/wandb/extract_metrics_from_sweep_per_projector.py`.

## Still at the root

`code/figures/make_calibration_fig.py` is the calibration schematic generator,
moved from the repository root. Its embedded values are illustrative.
No executable Python scripts remain at the root. `environment-ultra.yml`, `requirements.txt`, and `config.toml`
(`.example`) are dependency/configuration files, and `sweeps/` holds the
historical sweep definitions.

`tests/test_external_layout.py` pins the move: the Ultra drivers
(`prepare_ultra_data.py`, `score_ultra.py`) answer `--help` from a foreign
working directory (they load the real ULTRA only lazily, past `--help`), and
the Scala compiler and the semantic-similarity launcher are exercised with
fake `scalac`/`jar`/`groovy` executables in a temporary copied checkout (its
path contains spaces). Those stub executions verify invocation and path
wiring only — no real Scala/Groovy compilation, inference, Exomiser, or W&B
API call happens, and the stubs never write into the real `build/`, `data/`,
or `paper/` resources. `exomiser_eval.py` starts its JVM at import, so it is
not `--help`-tested. `tests/test_external_resource_paths.py` checks its
resource paths with a mocked JVM and checks the ULTRA environment path
with a stubbed installer. Neither test installs packages or runs inference.

The rank-CDF pair retains older GDAProjector configurations and embedded plot
values. Its move preserves historical behavior; it does not verify those values
against the current manuscript. Regenerate and verify the data before using it
for current paper figures. The test suite renders both PDFs in a temporary
directory; it does not overwrite manuscript figures.

`leakage_overlap_verify.py` also retains older hardcoded model configurations.
Its relocation preserves that diagnostic; it does not make its outputs current
manuscript results.
