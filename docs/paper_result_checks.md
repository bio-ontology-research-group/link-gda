# Paper prediction regression checks

`code/reproduce/paper_results.py` recomputes two paper tables from **saved
prediction TSV files** — the per-instance score rows written out at evaluation
time. It never retrains a model and never runs checkpoint inference; it only
re-runs the production load, leave-one-out calibration, and metric code
(`rq1_table.py`, `analysis/rq1_stats.py`) on those artifacts.

## The two checks

| check | paper source | setting | files |
|---|---|---|---|
| `rq1` | main.tex Table 2 (overall, calibrated) + main text / supplementary Nadeau–Bengio test | OWL2Vec*, calibrated-selected, seed 0, folds 0–9, LinkGDA vs INDIGENA, 4,399 candidates, 6,571 pairs | 20 |
| `excluded` | main.tex Table 3 (LinkGDA-fs, full pool, calibrated) | OWL2Vec*, full 4,749-gene pool, fold 0, seeds 0–9, 409 pairs / 350 genes / 402 diseases | 10 |

The expected values are **declared constants in the code**
(`DEFAULT_EXPECTED`, labeled with their paper table), not derived from the
computed result. `--expected overrides.json` (keyed by check name) can override
them. For `rq1`, the full-precision regression reference
(`paper/reviews/point36_multiplicity_2026-09-29/rq1_recomputed.json`) is used
automatically when present and checked at 1e-12 relative tolerance, separate
from the fixed-point display checks; pass `--precision-reference none` to
skip it.

Display checks round the recomputed value the way the tables print (MR/SD to
2 digits, p to 4 digits) and compare against the printed value. Structural
checks enforce candidate counts, pair counts, per-fold (rq1) or per-seed
(excluded) case-key order and true-gene index agreement, and unique case keys.

## Usage

```bash
.venv/bin/python code/reproduce/paper_results.py --check all \
    --rq1-results /path/to/main-benchmark/results \
    --excluded-results /path/to/excluded-benchmark/results \
    --output .reproducibility/paper-results/results.json
```

`--rq1-results` is required with `--check rq1|all`; `--excluded-results` is
required with `--check excluded|all`. The report JSON (stdout or `--output`)
carries: overall status, scientific scope, full-precision per-fold/per-seed
values, aggregates, expected-value comparisons, all 30 file paths with
SHA-256, byte size, case counts and candidate counts, query-alignment checks,
and source-code hashes of `rq1_table`, `analysis/rq1_stats`, and the checker
itself (the checker hash is `null` when executed via stdin without `__file__`;
a launcher may re-record input hashes itself).

## Exit codes

| status | exit code | meaning |
|---|---|---|
| `pass` | 0 | every selected check verifies |
| `fail` | 1 | a value, structural property, or input file is rejected |
| `unverified` | 2 | a required artifact is missing |

A mixed run is `fail` if any check fails, else `unverified` if any check is
unverified. Malformed rows, non-finite scores, out-of-range true indices,
unequal candidate counts, and duplicate case keys are rejected loudly; they
never produce a silent skip or deduplication.

## Bound

The TSVs carry no candidate-identifier column (only `gene`, `disease`,
`true_idx`, scores), so candidate-set identity across methods or seeds cannot
be verified from these artifacts — only candidate counts, case-key order, and
true-gene indices. This is recorded in the report under `limitations`.

## Tests

`tests/test_paper_results.py` covers the display rounding against the real
paper numbers, pins `DEFAULT_EXPECTED` to the declared paper values, and
exercises pass / mismatch / missing-artifact / altered-pairing / duplicate-key
paths on small synthetic files with parameterized fold/seed counts. The
production CLI keeps the exact 10-fold (rq1) and 10-seed (excluded) contract;
the tests never require the 30 real files.

## Verified reference run

On 30 September 2026, both checks passed against the 30 saved prediction files.
The returned mean ranks, sample standard deviations, excluded-set Hits@10, and
RQ1 adjusted p-value match the manuscript at its displayed precision. Full-precision
outputs, input SHA-256 checksums, and the executed source snapshot are retained
locally under gitignored `.reproducibility/paper-results/`. The two RQ1 mean ranks
and p-values also match the earlier independent full-precision recomputation.

Run the same command after cleanup against the same inputs. A check can run where
the data already reside; there is no need to download large prediction archives.
Check input hashes and retain the full-precision report as well as the paper-display
pass/fail result. These checks validate saved-prediction analysis, not retraining
or checkpoint inference.
