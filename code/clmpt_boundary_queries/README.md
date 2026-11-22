# clmpt_boundary_queries: query stage of the original-taxonomy boundary sweep

This package runs the CLMPT query stage on the eight new ComplEx initializers produced by
`code/clmpt_boundary`. It screens each initializer, promotes a shortlist to a longer budget and
selects the final checkpoints. It reuses the completed learning-rate 0.2, 633,450-update query
runs as the control and never reruns them. All selection uses the fold 0 VALIDATION split; the
TEST split is never read.

## Design

| step | trajectories | budget | rule |
|---|---|---|---|
| screen | 8 new initializers (learning rates 0.2, 0.3 and 0.4; 633,450, 844,600 and 1,055,750 updates; 0.2 at 633,450 excluded) plus the reused control screen | 2,500 updates | each trajectory selects its raw and calibrated checkpoints independently |
| promotion | the 9 screens | — | per view, rank by the selected checkpoint (H10, H1, MRR, −MR, then initializer budget, then identifier); top two raw union top two calibrated, deduplicated, 2–4, no fill |
| long | promoted new screens resume their exact terminal 2,500-update full state | 20,000 updates in total | a promoted control reuses its completed 20,000-update run |
| final | every promoted 20,000-update trajectory | — | per view, the same ranking; the first entry wins |

Within a trajectory, each view keeps its own step-0 hits@1 and hits@10 floor. A checkpoint is
eligible only when it reaches both floors. Among eligible checkpoints the selector maximizes
hits@10, then hits@1, then MRR, then −MR, and takes the earliest step on exact ties. Ranks are
unfiltered deterministic average ranks over 13,625 candidate genes for 545 validation queries and
586 associations. Calibration is the leave-one-out per-gene standardization of the executed
evaluator. Real, shuffled and empty-profile controls are scored at every evaluation.

Fixed query settings: learning rate 1e-4, temperature 0.05, rank 400, two Transformer layers with
8 heads, dropout 0.1, AdamW with weight decay 1e-4, batch of 96 ontology queries (all 13 axiom
patterns, square-root group weighting) plus 32 clinical profiles, 128 negatives, evaluation every
500 updates, seed 0, FP32 without TF32 and deterministic algorithms. `protocol.json` freezes these
values, the admitted bank, the pretraining lineage pins, the eight expected initializers and the
control.

New initializers were pretrained on H100 GPUs; the control initializer was pretrained on V100/A100
GPUs. Lineage is recorded by hash; results are not claimed to be bitwise identical across
hardware.

## Files

| file | role |
|---|---|
| `query/code/*.py` | query trainer, sampler, model, evaluator, metrics and state comparison; byte-identical to the sources that produced the control (hashes in `protocol.json`) |
| `analysis/select_stage.py` | the frozen stage selector, byte-identical |
| `select_boundary.py` | promotion and final selection; calls the frozen selector's trajectory verification, ranking key and union rule, and admits the control trained under its own protocol |
| `campaign.py` | immutable writes, manifest checks and the query/continuation config builders (same keys and values as the control configs) |
| `controller.py` | one Slurm allocation of one trajectory: saves an exact full state before the allocation deadline and submits a successor only after a clean pause; the start-time query and every submission run with `SLURM_TIME_FORMAT` removed and `TZ=UTC`, so a custom shell time format cannot shift the deadlines |
| `advance.py` | `tick` admits delivered initializers, creates trials, gates promotion and submits jobs; `select` runs a selection inside its own CPU job; `release` and `status` |
| `stage.py` | writes the campaign directory and its write-once frozen manifest |
| `bridge.py` | polls the pretraining campaign, transfers each new milestone model with its lineage metadata and runs the advancer |
| `job-template.sh`, `select-job-template.sh` | Slurm scripts rendered by `stage.py` |
| `protocol.json` | frozen protocol `clmpt-boundary-original-queries-v1` |

The model loads three official CLMPT source files at run time (`official-clmpt.py`,
`official-complex.py`, `official-transformer.py`) from `query/references/` next to the worker,
and `official-training.py` is kept with them as provenance. They are not tracked here.
`stage.py` copies them from a local directory and refuses any file whose SHA-256 differs from
`protocol.json`.

## Running

Requirements: Python 3.11 with `torch==2.6.0` (CUDA 12.4 build) and `numpy==1.26.4`
(`requirements-query.txt`) for the worker. The other scripts use the standard library and Slurm
commands. Run everything from a clean checkout of the pushed commit; the stager refuses an
uncommitted or modified package.

```bash
CODE=<checkout>/code/clmpt_boundary_queries
CAMPAIGN=<new empty campaign directory>
PY=<python with the requirements above>

python "$CODE/stage.py" --campaign "$CAMPAIGN" --code-revision "$(git -C "$CODE" rev-parse HEAD)" \
    --python "$PY" --bank-dir <admitted bank directory> --control-campaign <completed control campaign> \
    --references-dir <directory with the four reference files>
python -I "$CODE/advance.py" release --campaign "$CAMPAIGN" --approve --authority "<who approved and what>"
python -I "$CODE/advance.py" status --campaign "$CAMPAIGN"
python -I "$CODE/advance.py" tick --campaign "$CAMPAIGN"
```

The bridge runs on a machine that can reach both clusters, with a JSON config naming the remote
commands, campaigns and local storage (keys: `state_dir`, `store_dir`, `pretrain`, `query`,
`poll_seconds`, `transport_attempts`, `transport_backoff_seconds`, `transport_failure_limit`,
`transfer_timeout_seconds`, `no_progress_hours`):

```bash
python3 -I "$CODE/bridge.py" inventory --config bridge-config.json
python3 -I "$CODE/bridge.py" loop --config bridge-config.json
```

`inventory` is read-only. `loop` holds a lock, runs one cycle every `poll_seconds` and exits on
completion, on a halt, on an `OPERATOR_STOP` file in `state_dir`, after repeated transport
failures, on any milestone that fails verification, or after `no_progress_hours` without progress
while no job is queued or running.

## Outputs

Each trajectory writes, under `trials/<stage>/<id>/training/`, the update log, the evaluation
curve, both views' selections, every scheduled evaluation's scores (`fits/fit-step-*.npz` with the
real, shuffled and empty-profile score matrices) and its metadata, the latest full state, and
model-only exports of the initial, raw-best and calibrated-best checkpoints. Decisions are written
once to `decisions/`: `screen-index.json`, `screen.json` (leaderboards and promoted set),
`S5-plan.json`, `final-index.json` and `final.json` (leaderboards and winners per view). Any metric
can be recomputed from the saved scores.

Nothing is retried automatically. A trial error, an ambiguous submission that cannot be matched
to exactly one scheduler job, a lineage mismatch or a failed selection stops further submission
until a person resolves it. Full states of screens that are not promoted are kept.
