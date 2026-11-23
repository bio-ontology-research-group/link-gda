# CLMPT suite: query-stage engine

Trains the CLMPT query model on top of a pretrained ComplEx initializer, evaluates on validation
cases at fixed intervals, and keeps raw and calibrated checkpoint selections separate. No result
in this repository has been produced by this package yet.

## Setup

Python with PyTorch (CUDA) and NumPy. Place the four official CLMPT source files in `references/`
(`official-clmpt.py`, `official-complex.py`, `official-transformer.py`, `official-training.py`).
They are not distributed here because the upstream repository has no license; the trainer checks
their sha256 against `protocols/suite-query-v1.json`.

Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=0`; the trainer refuses to start without
them.

## Conditions

| condition | `--encoder` | `--curriculum` |
|---|---|---|
| Full, shared encoder | shared | full13 |
| Core (1p, pip, ppip) | shared | core3 |
| Clinical-only (no ontology loss; clinical loss summed and divided by 128) | shared | aux_off |
| Independent phenotype branches (mean branch cosine) | independent | full13 |
| Plain mean aggregation instead of the Transformer | mean | full13 |

`--seed` sets model construction and global generators; `--sampler-seed` (default: the seed)
sets seven independent sampling streams, so clinical draws are identical across curricula and
encoders that share a sampler seed.

## Commands

```
cd code
python make_config.py build --protocol ../protocols/suite-query-v1.json --purpose hpo_screen ... --output run.json
python make_config.py validate --config run.json --output run-validation.json
python train.py --config run.json --output RUN_DIR [--allocation-stop-at UNIXTIME]
python leaderboard.py promote --protocol P --index screens.json --output promotion.json
python leaderboard.py freeze  --protocol P --index continuations.json --decision DECISION.json --output freeze.json
python test_eval.py --config test.json --output TEST_DIR
python qualify.py {streams,construction,scores,resume} ...
```

`train.py` returns 0 when complete, 75 when paused with a consistent saved state (rerun the same
command to resume), and writes `ATTENTION.json` on error. Outputs per run: `identity.json`,
`progress.json`, `updates.jsonl` (per-update loss and draws), `curve.json`, `selection.json`,
`fits/fit-step-*.npz` (saved validation score matrices and representations) with metrics JSON,
`models/` (initial, raw-best and calibrated-best weights), and `COMPLETE.json`.

Score semantics: shared and mean scores are cosine(representation, gene). Independent-branch
representations are the mean of unit branch vectors, not renormalized; their score is the dot
product with the unit gene embedding, so cosine replay of that representation is wrong.

Clinical training genes: a bank may list `training_gene_ids`, the gene universe from which clinical
TRAIN positives and negatives are drawn; without it the ranking candidates are used. Validation and
TEST always rank `candidate_ids` only. `python qualify.py domain --bank DIR` reports a bank's counts
and gene-universe identity for its protocol record.

Across configurations, leaderboards order by hits@10, hits@1, MRR, lower mean rank, then trial id.

TEST is scored only by `test_eval.py`, which requires a release file naming the completed run.
