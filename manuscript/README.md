# Working manuscript source

This directory is a source snapshot of the revised paper and supplement. It
includes the corrected baseline results and historical comparisons. The new
controlled-query and matched-KGE results are still being produced; this is not
the final paper release. [Reproduction status](../reproducibility/STATUS.txt)
records the remaining scientific and artifact requirements.

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
actual saved scores. The [embedding replay recipe](../docs/embedding-table-replay.md)
rebuilds the other ten result tables, prepared-input counts and rank-CDF figure
from retained scores and annotations. The latter archive publication is tracked
separately; a source-only LaTeX build does not establish training reproduction.

`fig/phenoquery-method.tex` draws the main method diagram directly in LaTeX.
The supplementary graph schematic is built from `fig/overview_graph_clean.tex`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=manuscript/fig manuscript/fig/overview_graph_clean.tex
SOURCE_DATE_EPOCH=1793232000 "$ANALYSIS_ENV/bin/python" \
  code/figures/make_calibration_fig.py "$GENERATED_ASSETS/fig_calibration"
```

The calibration illustration uses fixed hypothetical scores, not experimental
measurements. Its numerical construction is defined in the generator. Rendered
layout can vary with Matplotlib versions; the pinned analysis environment is
in `reproducibility/requirements-embedding-replay.txt`.

The OUP class retains its copyright and LPPL notice. The renamed
`oup-abbrvnat-short.bst` retains its original license notice and documents the
local bibliography-name modification in its header.
