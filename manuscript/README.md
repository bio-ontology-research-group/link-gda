# Working manuscript source

**Data access update, 30 October 2026:** bulk GitHub Release attachments have
been withdrawn. The snapshot's data-availability paragraph predates that
storage change and must be revised to the final archive destination before
submission. See [artifact storage and access](../docs/artifact-storage.md).

This directory contains the paper and supplement as of 30 October 2026,
including the complete three-seed controlled-query comparison, matched KGE
comparison, baseline results and bias analyses. The
[reproduction status](../reproducibility/STATUS.txt) records the remaining
independent-inference and public-artifact checks.

The source is copied explicitly from the separate manuscript repository, with
file hashes in `source-manifest.json`. No nested Git repository or internal
review files are included. The snapshot must be updated when manuscript results
or prose change.

## Build the PDFs

Use a TeX Live installation with pdfLaTeX, BibTeX, latexmk, the OUP template
dependencies and the standalone/TikZ packages. The checked build uses pdfTeX
1.40.26 (TeX Live 2025/dev/Debian) and latexmk 4.86. From this directory:

```bash
export SOURCE_DATE_EPOCH=1793232000
export FORCE_SOURCE_DATE=1
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

The second pair of commands resolves cross-document references from a clean
checkout. Outputs are `main.pdf` and `supplementary.pdf`. Inspect the final logs
for unresolved references or citations. PDF byte identity across different TeX
installations is not asserted; score/table identity is checked separately.

The [availability revision build](../reproducibility/verification/manuscript-availability-build.json)
verifies all 49 source files and resolves references in the 12-page main paper
and 46-page supplement. Main pages 1–10 render identically to the previous
verified public build; the changed final two pages have been visually reviewed.
The supplementary PDF is byte-identical to the previously reviewed artifact.
A [fresh public-source build](../reproducibility/verification/manuscript-availability-public-build.json)
downloads all 49 files without credentials and reproduces both revised PDFs
byte for byte in the same TeX environment.
The earlier [clean build](../reproducibility/verification/manuscript-controlled-clean-build.json),
[complete rendered review](../reproducibility/verification/manuscript-controlled-rendered-review.json)
and [fresh public build](../reproducibility/verification/manuscript-controlled-public-build.json)
retain their own source identities and scopes.

## Regenerate the exhibits

From the repository root, the current saved-report table command is:

```bash
PYTHON_BIN="$ANALYSIS_ENV/bin/python" \
  bash reproducibility/run_pipeline.sh generate-tables \
  --out-dir "$GENERATED_TABLES"
```

It writes the main/subgroup bias tables, isolated-factor table, historical query
tables and complete native/adapted baseline tables. Compare generated `.tex`
files with the matching `manuscript/tables/` files before replacing them.
The [baseline archive recipe](../docs/final-baseline-results.md) starts from the
actual saved scores. The [embedding replay recipe](../docs/embedding-archive-replay.md)
rebuilds the other ten result tables, prepared-input counts and rank-CDF figure
from retained scores and annotations. Both archive collections are public and pass their full fresh-download replays.
The [embedding verification](../reproducibility/verification/legacy-embedding-fresh-public-replay.json)
also checks all eleven generated tables and both CDF artifacts against the
current manuscript.
A source-only LaTeX build does not establish training reproduction.

The controlled-result tables have separate reproducible exporters. Follow the
[controlled-query score replay](../docs/controlled-query-score-replay.md) for
the query metrics, eight primary contrasts and six bias panels. The
[matched KGE score replay](../docs/matched-kge-score-replay.md) regenerates its
primary and bias tables; the complete verified summary and table exports are in
`../reproducibility/results/matched-kge-summary.json` and
`../reproducibility/results/matched-kge-tables/`.

`fig/phenoquery-method.tex` draws the main method diagram directly in LaTeX.
The supplementary graph schematic is built from `fig/overview_graph_clean.tex`:

```bash
mkdir -p "$GENERATED_ASSETS"
export SOURCE_DATE_EPOCH=1793232000 FORCE_SOURCE_DATE=1
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory="$GENERATED_ASSETS" manuscript/fig/overview_graph_clean.tex
"$ANALYSIS_ENV/bin/python" \
  code/figures/make_calibration_fig.py "$GENERATED_ASSETS/fig_calibration"
```

Set `GENERATED_ASSETS` to a new output directory before running these commands.
Both rebuilt illustrations render identically to the supplied PDFs at an
1800-pixel longest dimension. The [illustration verification](../reproducibility/verification/manuscript-illustration-replay.json)
records generator, PDF and rendered-image hashes, tool versions, and an
independent check of the calibration calculation. PDF bytes differ because the
supplied illustrations were created with earlier tool versions and timestamps.

The calibration illustration uses fixed hypothetical scores, not experimental
measurements. Its numerical construction is defined in the generator. Rendered
layout can vary with Matplotlib versions; the pinned analysis environment is
in `reproducibility/requirements-embedding-replay.txt`.

The OUP class retains its copyright and LPPL notice. The renamed
`oup-abbrvnat-short.bst` retains its original license notice and documents the
local bibliography-name modification in its header.
