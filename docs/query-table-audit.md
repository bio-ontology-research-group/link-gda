# Historical query table audit

`code/analysis/summarize_query_tables.py` recomputes every historical matched
ULTRA/UltraQuery and PhenoQuery table cell from candidate scores. It verifies
250 bundles across ten folds, including all 500 score/rank file hashes,
explicit candidate and query identities, test-pair order, training-status
labels and exact equality of recomputed raw/calibrated ranks to retained ranks.
The saved report records input and executing-source hashes.

```bash
python code/analysis/summarize_query_tables.py \
  --base "$INPUTS/benchmark" --matched "$QUERY_OUTPUTS/matched" \
  --eq "$QUERY_OUTPUTS/eq" --output "$WORKSPACE/query-table-report.json"
python reproducibility/generate_query_tables.py \
  --report-json "$WORKSPACE/query-table-report.json" --out-dir "$WORKSPACE/tables"
```

The `query-outputs-20261028-v1` release is a separate artifact from the
prepared-input release. Its eleven checksum-bound archives contain all 250
score/rank bundles and retained validation-selection metadata (799 files,
12.44 GB compressed). Allocate another 30 GB for download/extraction staging.
Download them from a checkout containing the output manifest:

```bash
python3 code/reproduce/artifacts.py fetch \
  --manifest reproducibility/artifacts/query-outputs-20261028-v1.json \
  --root "$WORKSPACE/query-outputs" --cache "$WORKSPACE/query-downloads"
export QUERY_OUTPUTS="$WORKSPACE/query-outputs"
```

All archives passed local extraction and checksum verification. A complete
fresh-public-download score-to-table replay is tracked separately.
The controlled multi-seed experiment is also separate: this report describes
the historical seed-zero campaigns.

The generated comparison table reproduces the previous equal-fold overall MR,
MRR and Hits metrics. Its seen/unseen and diagnostic cells correct values in
the manually assembled tables that did not match retained scores and identities.
For example, original-pool adapted ULTRA's calibrated unseen-gene Hits@10 is
6.02%, not 9.01%; its calibrated zero-overlap seen-gene Hits@10 is 3.39%, not
0.85%. These are corrections to the historical table, not new trained results.
The original hand-table generation is unavailable, so the cause of those
incorrect cells is not assigned to an unverified rounding or ordering error.

MR/MRR/Hits columns use equal-fold means; MR dispersion is the sample standard
deviation of fold means. Median and status/overlap-stratum summaries pool test
associations. Within-status ranks compare the true gene only with candidates
of the same training status. The exact wrong-query control averages its rank
under every other test-disease query in that fold. Profile identity can match
any training gene; it is not restricted to a different training gene.

The full supplementary table includes both raw and calibrated results for all
25 arms, including frozen and adapted stages. In the matched campaign, Primary
marks validation-selected scoring. The historical PhenoQuery component study
reports calibrated scoring without claiming that this mode was selected by
that study. All of its arms initialize from general ULTRA (`ultra_4g`).

The full results are split by campaign across three landscape tables for
readability. Include `\usepackage{pdflscape}` and `\usepackage{adjustbox}` in
the supplement preamble. The manuscript's method overview is editable TikZ
source in `reproducibility/figures/phenoquery-method.tex`; it illustrates the
pipeline and contains no simulated performance measurements.
