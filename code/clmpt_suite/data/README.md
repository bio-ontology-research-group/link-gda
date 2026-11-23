# clmpt_suite/data: fold, modality and readout preparation

CPU-only preparation of the inputs for the CLMPT experiment suite: the ten supplied clinical folds, each with seven
native-annotation modalities (P, F, S, PF, PS, FS, PFS). This package also provides an audit of the excluded no-MGI
cohort and a paired raw/calibrated validation readout. Nothing here trains a model, reads TEST scores or changes an
existing dataset. Every output directory is write-once and bound by a `COMPLETE.json`.

## What a prepared fold and modality contains

| item | rule |
|---|---|
| clinical folds | the supplied `benchmark/data/fold{0..9}` partitions and profiles, unchanged |
| vocabulary | identical across modalities of a fold: the 234,644 shared non-disease names plus the fold's TRAIN diseases (239,552 entities in fold 0, 239,553 in fold 9), 624 relations; held-out diseases absent |
| ontology background | unchanged; uPheno, GO and UBERON projections only, no new projection |
| TRAIN supervision | `associated_with` equals the fold's `training.tsv`; `has_symptom` equals its TRAIN profiles |
| modality | keeps the named native gene annotations (P `has_phenotype`, F `has_function`, S `expressed_in`) and removes the others with their inverses |
| phenotype fallback (F, S, FS) | a gene with a native phenotype edge but no edge of an active relation keeps one phenotype edge: the first represented phenotype in a pinned order source. Support is decided from native annotations only, before TRAIN associations. Expected: 418, 1,175 and 257 genes; 29, 426 and 15 of the 6,571 outer TEST pairs |
| `causes_phenotype` | absent from every graph |
| query bank | one per fold, shared by all modalities: the audited fold0 axiom13 original bank (13 patterns, 175,906 represented classes) remapped through IRIs, plus the fold's TRAIN profiles and validation cases |
| candidates | the original 13,625 genes, same order in every fold |

The PFS graph of each fold is the fold base itself, linked with an unchanged `COMPLETE.json`, so the identity of an
existing PFS initializer is preserved. Folds 0 to 2 adopt existing bases; folds 3 to 9 are built by
`fold_builder.py`, a byte-identical copy of the executed three-fold builder (SHA-256 `befaaea6…`, checked before
use). It in turn imports `code/compositional_gda/prepare.py` (SHA-256 `0a77cf2f…`). Files that are
identical across modalities (vocabulary, candidates, union audit, banks, TEST cases) are hardlinked when possible.

## Fallback order source

`init` pins one order policy for the whole root:

- `prepared-csv`: the prepared MGI gene–phenotype CSV (columns `Gene`, `Phenotype`) in its row order, which is the
  historical rule. Pass `--order-source` and `--order-sha256`.
- `raw-train-line-order`: the order of `has_phenotype` rows in each fold's `raw/train.txt`, hashed through the
  benchmark manifest. This is deterministic but has not been shown to equal the prepared-CSV order.

The builder refuses if the order source lists a represented phenotype that is not a graph edge for that gene, if a
fallback gene has no eligible row, if the counts differ from the expected values, or if a fold's fallback edges differ
from fold 0's.

## Commands

Run from the repository root with Python 3.11 or later and NumPy; no other dependency. Peak memory is a few GB (one
bank JSON and one graph in memory).

```bash
D=code/clmpt_suite/data
ROOT=data/<new suite input directory>

python $D/prepare_suite.py init --root $ROOT --order-policy prepared-csv --order-source <csv> --order-sha256 <sha256>
python $D/prepare_suite.py base --root $ROOT --fold 0
python $D/prepare_suite.py verify-base --root $ROOT --fold 0
python $D/prepare_suite.py qualify --root $ROOT
python $D/prepare_suite.py plan --root $ROOT --output $ROOT/PLAN.json
python $D/prepare_suite.py modality --root $ROOT --fold 0 --modality FS
python $D/prepare_suite.py manifest --root $ROOT --output $ROOT/MANIFEST-<label>.json
```

`prepare_suite.py all --root $ROOT` runs every step below in order and resumes after an interruption;
`excluded_cohort.py all --root $ROOT` does the same for the excluded cohort.

`qualify` writes fold 0 metadata for the six derived modalities and audits the fallback counts against all ten TEST
partitions without writing any graph. For folds 1 to 9, run `base` and `verify-base` for each fold. `verify-base`
also requires the fold's ontology bank, with every identifier replaced by its IRI, to equal fold 0's. Then run
`modality` for each needed fold and modality. Each `modality` call writes `graphs/fold{k}/{M}/`, links `banks/fold{k}`
and `tests/fold{k}`, and records `entries/fold{k}-{M}.json` with the triple count and the steps per epoch for the
pretrainer.

Excluded no-MGI cohort (after `init`, `base --fold 0` and `qualify` on the same root):

```bash
python $D/excluded_cohort.py build --root $ROOT
python $D/excluded_cohort.py verify --root $ROOT
python $D/excluded_cohort.py modality --root $ROOT --modality FS
```

`build` reads the hash-pinned frozen sources in `data/clmpt-suite-20261123/immutable-sources/`. TRAIN is the
`train.csv` rows of the 5,454 archived reference diseases. Every other row is held-out validation. Selection uses
the validation rows whose gene is in the full GO pool and has no TRAIN association (333 pairs, 324 diseases). No split
is resampled. It writes:

- `excluded/cohort/`: the exported partitions, profiles, pools and audit;
- `excluded/base/`: shared ontology and native background, with the exact excluded TRAIN associations and profiles;
- `excluded/bank/`: the fold0 ontology bank remapped through IRIs. `candidate_ids` is the full GO pool;
  `training_gene_ids` also covers the TRAIN answer genes outside that pool;
- `excluded/test/`: 425 associations ranked against `candidate_ids` (the full GO pool), with a no-MGI candidate view
  and the expression-supported subset. The bank's `COMPLETE.json` declares `training_gene_domain`.

`modality` derives the F, S or FS arm with the same first-eligible fallback for main MGI genes. Excluded TEST genes
have no MGI phenotype and get no fallback. `verify` re-reads every artifact against the exported cohort and the fold0
ontology digest.

Paired raw and calibrated readouts:

```bash
python $D/readout_pairs.py --decision <SELECTION-DECISION.json> --metrics code/clmpt_boundary_queries/query/code/metrics.py \
    --trajectory <training dir> [--trajectory ...] [--recompute --bank <bank.json>] --output <readout.json>
```

For every evaluated step, `readout_pairs.py` pairs the raw and calibrated metrics of the same saved scores. It
recomputes each trajectory's raw-selected and calibrated-selected checkpoints with the executed selector and crosses
the selection view with the scoring view in a two-by-two table. It ranks the selected checkpoints on both leaderboards
by Hits@10, Hits@1, MRR and lower mean rank. The leaderboard named in the decision file gives the common-recipe
candidate; the other winner is reported separately. `--recompute` re-derives every metric from
`fits/fit-step-*.npz` and requires agreement to 1e-12.

## Fallback order used

The suite uses `prepared-csv` with the frozen prepared MGI table (`gene_phenotypes.csv`, 213,988 rows, SHA-256
`94829ab6…`). Fold 0 qualification reproduces 418, 1,175 and 257 fallback genes and 29, 426 and 15 affected TEST
pairs.
