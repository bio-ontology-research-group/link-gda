# Archived significance-testing scripts

These scripts use historical run configurations and do not reproduce the current
paper's reported p-values. They were moved unchanged from the repository root on
29 September 2026.

- `p_value.py` pools test pairs as independent observations and uses strictly-greater
  tie handling rather than deterministic average ranks.
- `p_value_per_fold.py` uses older configurations and strictly-greater tie handling.
  Its fold-level paired test does not apply the Nadeau–Bengio correction.

For the current corrected RQ1 comparison, use `analysis/rq1_stats.py` and the
multiplicity instructions in the root README. These archived scripts retain
historical comments and claims; those are not descriptions of the current paper.

Their data/results paths resolve relative to the working directory, not this
archive. Historical invocation requires the matching old result files and a
working directory containing data/results. No analysis was rerun during archiving.

## Historical corrected-test cross-check

- `dump_perfold_vectors.py` extracts fold means for hardcoded historical
  configurations using strictly-greater tie handling. Its output path is
  `data/perfold_vectors.json`, relative to the working directory.
- `nb_corrected_ttest.R` compares naive, Nadeau–Bengio, correctR, and sign tests
  using historical vectors embedded in the script. It does not read the JSON
  dynamically. The correctR dependency is needed only for this historical check.

Both files were archived unchanged on 29 September 2026. The existing
`data/perfold_vectors.json` remains in its original location as a historical
artifact. These configurations and vectors are not the current paper's tests.

## Historical RQ2 and seed check

`verify_rq2_and_seeds.py` checks old GDAProjector f/p configurations, uses
strictly-greater ties and an uncorrected paired t-test, and prints a historical
paper claim. It was archived unchanged on 29 September 2026. It does not
reproduce the current RQ2 results or the supplementary RQ1 projector comparison.
Its input paths remain relative to the working directory.
