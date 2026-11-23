# clmpt_suite: campaign orchestration

This directory runs CLMPT suite campaigns on a Slurm cluster. The query engine lives in `query/`,
the ComplEx engine in `pretrain/` and the dataset preparation in `data/`, each with its own README.
The root files here stage a campaign, run each trajectory in resumable three-hour allocations,
select checkpoints and record every decision. No result has been produced by this package yet.

The implemented campaign is Stage 0 qualification and the Stage 1/2 second-stage hyperparameter
search. The production suite (Stages 3 to 6) is planned by `production.py` but not yet launchable.

## Hyperparameter search design

| step | trajectories | budget | rule |
|---|---|---|---|
| qualification | one reference and two resumed copies of one configuration | 1,000 updates | resumed copies pause on their first allocation (at step 250, and 600 s after start) and continue in an automatic successor; every metric, score array, update-log row and the terminal model, optimizer, sampler and RNG state must equal the reference exactly |
| engine qualification | the query worker's own qualifiers (construction and score parity on CUDA; exact 20-step pause/resume for the five conditions and seed 1; a 40-step pretraining resume) | — | run as a separate GPU job; `advance.py admit-engine-qualification` verifies the receipts against this campaign's worker sources; cannot be waived |
| screen | 2 ComplEx initializers × learning rates 5e-5, 1e-4, 2e-4 × temperatures 0.025, 0.05, 0.1 = 18 | 5,000 updates | starts only after a passed qualification and an admitted engine qualification |
| continuation | top two raw union top two calibrated screens, deduplicated (2–4), no fill | to 20,000 updates | resumes the exact terminal 5,000-update state; starts only when all 18 screens are complete |
| final | every continuation | — | raw and calibrated winners selected separately |
| recipe | — | — | the common attribution recipe is the winner of the leaderboard named in a recorded selection decision |

Fixed settings: PFS fold 0, shared encoder, Full13 curriculum, seed 0, sampler seed 0, evaluation
every 500 updates on 545 VALIDATION cases (586 associations, 13,625 candidate genes). TEST is
never read.

Selection, per trajectory and per view (raw, calibrated): the checkpoint must reach its own
step-0 hits@1 and hits@10; among those, maximize hits@10, then hits@1, then MRR, then −MR, and take
the earliest step on exact ties. Across trajectories: hits@10, hits@1, MRR, −MR, then the trial id
and nothing else. `select_hpo.py` recomputes every selection from the
saved curves and requires it to equal the worker's `selection.json`. The query worker's
`leaderboard.py` runs on the same index as an independent check; a disagreement stops the campaign.

Every decision reports, for each trajectory, raw and calibrated metrics at the same checkpoint
(initial, raw-selected, calibrated-selected and terminal) and the separately selected raw and
calibrated winners. The calibrated-minus-raw differences are descriptive; nothing assumes that
calibration improves ranking.

## Files

| file | role |
|---|---|
| `hpo-protocol.json` | frozen campaign protocol: grid, initializer candidates, qualification, selection, storage and runtime policy |
| `suite.py` | shared helpers: write-once files, hashing, manifest verification, Slurm commands, config builders |
| `stage_hpo.py` | writes a campaign directory once: copies the worker, its protocol and the references, builds all 21 planned configs with the copied worker's `make_config.build_config`, validates each with its `validate_config`, writes `frozen-manifest.json` |
| `worker_configs.py` | builds and validates configs through the worker's `make_config` (used by the stager and by the screen-selection job for continuations) |
| `preflight.py` | CPU-only checks before staging (see Preflight) |
| `ops/watch.py` | workstation watcher that runs `tick` over SSH (see Monitoring) |
| `controller.py` | one Slurm allocation of one trajectory (see Runtime) |
| `advance.py` | `tick`, `status`, `gate`, `release`, `record-decision`, `select`, `qualify`, `resume`, `stop`, `ack` |
| `select_hpo.py` | screen promotion, final winners and recipe freeze from saved VALIDATION artifacts (standard library only) |
| `qualify.py` | qualification comparison and measured update time, evaluation time and artifact sizes |
| `capacity.py` | user job cap and storage admission |
| `production.py` | production plan (109 ComplEx and 185 query trajectories) and its readiness report |
| `ops/job-template.sh`, `ops/cpu-job-template.sh` | Slurm scripts rendered at staging |

## Running

Requirements: the query interpreter (Python with `torch==2.6.0` and `numpy==1.26.4`) for staging,
the GPU jobs and the CPU decision jobs; the other commands use only the standard library and the
Slurm client commands. Stage from a pushed checkout; the stager refuses if any executed file
(the root orchestration files, the query worker and its protocol) is uncommitted or modified. The
four official CLMPT references are copied from `references_dir` by hash and are never committed
(`.gitignore`).

Write an inputs file:

```json
{
  "schema": "clmpt-suite-hpo-inputs/1",
  "bank_dir": "<admitted fold-0 original bank directory>",
  "initializers": {
    "lr0.4-fresh-u0633450": "<path>/update-0633450.pt",
    "lr0.2-resume633450-u1055750": "<path>/update-1055750.pt"
  },
  "references_dir": "<directory with the four official CLMPT files>",
  "python": "<query interpreter>",
  "protected_trees": ["<every historical campaign directory the new one must not overlap>"]
}
```

```bash
CODE=<checkout>/code/clmpt_suite
CAMPAIGN=<new empty campaign directory>
python3 "$CODE/stage_hpo.py" --campaign "$CAMPAIGN" --code-revision "$(git -C "$CODE" rev-parse HEAD)" --inputs inputs.json
python3 -I "$CODE/advance.py" status --campaign "$CAMPAIGN"
python3 -I "$CODE/advance.py" release --campaign "$CAMPAIGN" --approve --authority "<who approved what>"
python3 -I "$CODE/advance.py" record-decision --campaign "$CAMPAIGN" --decision-file SELECTION-DECISION.json --authority "<who decided>"
python3 -I "$CODE/advance.py" tick --campaign "$CAMPAIGN"
```

After the engine qualification job has written its receipts (`qualify.py construction`, `scores`,
and `resume` for each required condition and for pretraining, every `--config` given as an absolute
path, only final receipts in the directory):

```bash
python3 -I "$CODE/advance.py" admit-engine-qualification --campaign "$CAMPAIGN" --receipts-dir <receipts> \
    --worker-code <the query/code directory that produced them> --authority "<who reviewed them>"
```

Run `tick` periodically from a monitor; it is idempotent and holds a lock. `status` and `gate`
are read-only and print JSON. Exit codes: 0 ok, done, busy or operator stop; 1 halted or a refused
CPU action; 2 refusal (message on stderr).

`release` options: `--user-active-job-cap N` (lower the cap of 24 jobs of this user, all projects
counted), `--free-bytes-command '["cmd", "arg"]'` (a quota command printing available bytes; the
smaller of it and `df` is used), `--reserve-bytes N`, and `--waive-qualification` (recorded with
its authority).

## Preflight

```bash
<query interpreter> code/clmpt_suite/preflight.py --inputs inputs.json --decision SELECTION-DECISION.json \
    --scratch <new empty directory> --output preflight.json
```

It compiles the root files, checks the campaign protocol against the worker protocol, builds and
validates the 21 planned configs and one continuation config through the worker, runs both
selectors on synthetic trajectories with exact ties and requires identical results, tests the
engine-qualification admission on synthetic receipts, and checks the deadline arithmetic. Synthetic
trajectories test selection logic only. To rehearse staging without a commit, stage with
`--local-check-uncommitted` (the campaign can never be released) and run
`preflight.py --rehearse-campaign <that campaign>`.

## Monitoring

`ops/watch.py` runs on a machine that can SSH to the cluster login node. It calls only `tick` and
`gate`, logs every cycle to `<state-dir>/cycles.jsonl`, keeps `last.json` and `last-gate.json`, and
writes `ATTENTION.json` on a halt, a refusal, unacknowledged trial attention, a storage stall or
repeated SSH failures. It never cancels, resumes, releases or acknowledges anything. One watcher per
state directory; a `STOP` file in it ends the watcher.

```bash
nohup python3 code/clmpt_suite/ops/watch.py --ssh-target <login node> --remote-code <checkout>/code/clmpt_suite \
    --campaign <campaign> --state-dir <local state dir> --interval 600 >> <local state dir>/watch.out 2>&1 &
```

## Runtime

Each allocation requests three hours on one V100 or A100. The controller reads the job's own
TimeLimit and the cluster OverTimeLimit at run time: with a grace of at least 60 minutes the worker
saves at 3 h 40 and the allocation exits by 3 h 45; with a shorter or unreadable grace both
deadlines move inside the request. The worker runs under `srun`; SIGTERM is forwarded to the step,
which saves and exits within 65 s. A successor allocation is submitted only after a clean pause
whose committed state verifies, after real progress, within the trajectory's allocation bound
(qualification 3, screen 3, continuation 8), and never while another running job carries the
trajectory's name. Errors are never retried. A stop marker (`OPERATOR_STOP.json` or the trial's
`STOP.json`, written by `advance.py stop`) ends the trajectory as STOPPED; a clean pause while the
campaign is halted or stopped is HELD and resubmitted by `tick` once nothing holds it.

`tick` submits a trial only if the user's queued and running jobs stay below the cap and the free
space, minus the reserve and minus the growth still owed by every unfinished trial, covers the new
trial's peak (three full states, four model-only exports and its new score archives, times 1.25).
Peaks use the qualification's measured sizes once it has passed.

## Outputs

Each trajectory writes, under `trials/<Q|S|C>/<id>/`: `config.json`, `runtime.json`, submission
receipts, `allocations/wave-NNN/` (allocation record with the deadlines and validated grace, GPU
inventory and telemetry, worker log, and one of `ALLOCATION_COMPLETE`, `CLEAN_DEADLINE` with
`successor`, `HELD`, `STOPPED` or `ATTENTION`), and `training/` (the worker's outputs, including
every evaluation's saved scores). Decisions are written once to `decisions/`: the
qualification, screen, final and recipe indexes and decisions, the cross-check records, the
validated continuation configs and the recorded selection decision. `decisions/recipe.json` holds
`production_recipe` in the format the production query protocol expects.

After the CPU job that verifies a decision, full-state checkpoints that decision makes unnecessary
are deleted: qualification trials after a pass, screens that were not promoted, continuations that
won neither view, and promoted screens whose continuation has completed. Each deletion has a
per-file receipt in `state/pruned/` and a `PRUNED.json` in the training directory. Model-only
exports (initial, raw-best, calibrated-best), score archives, curves and logs are never deleted.

## Production plan

```bash
python3 "$CODE/production.py" plan --output production-plan.json
python3 "$CODE/production.py" readiness --hpo-campaign "$CAMPAIGN" --data-manifest <file> --pretrain-interface <file> --query-protocol <file>
```

The plan lists every trajectory with its dependency: a query trajectory is released by the
delivered COMPLETE record of its own ComplEx trajectory, so there is no barrier across all
pretraining. `readiness` always reports the parts that are not implemented (production stager,
dependency-aware advancer, cross-cluster delivery, production qualifications) and never claims a
launchable suite.
