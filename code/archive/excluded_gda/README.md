# Archived excluded-benchmark GDAProjector campaign

`run_excluded_seeds.sh` is the historical launcher for LinkGDA-f and LinkGDA-fs
on the excluded benchmark, using fold 0 and seeds 0–9. It selects GDAProjector,
dimension 100, batch size 16,384, learning rate 0.001, and raw validation mean
rank. It was moved unchanged from the repository root on 29 September 2026.

This campaign is not used for the current paper. The retained supplementary
GDAProjector comparison uses the main RQ1 benchmark and full pfs models across
ten folds. The current excluded-benchmark results use OWL2Vec*.

The launcher's comments and embedded usage examples describe the historical
layout. After this move, its default CODE path points to this archive rather
than the trainer. To rerun the historical campaign, set CODE explicitly to the
directory containing kge_transd.py and launch from the prepared excluded-benchmark
working directory. Set PYTHON to the intended interpreter if needed. The script
launches variants on GPU indices 0 and 1; inspect those assignments before use.

No historical experiment was rerun during archiving. This script is not a
reproduction command for the reported OWL2Vec* results.

## Historical diagnostics

The following scripts were also archived unchanged on 29 September 2026:

- compare_tolerance_arms.py compares the old tol5/tol15 campaign outputs.
- diagnose_early_stopping.py relates historical training logs to test ranks.
- variance_decomposition.py examines per-instance variation across those seeds.

All three hardcode the old GDAProjector, dimension-100 f/fs configuration and
use strictly-greater tie handling. They are not generators for the current
excluded-set results. Their paths resolve from the supplied arguments or working
directory. The active excluded_table.py and analyze_excluded_seeds.py remain
outside this archive.
