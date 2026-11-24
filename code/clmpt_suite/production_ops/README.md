# production_ops: production execution of the CLMPT suite

Runs the production units after the query HPO has frozen its common recipe. There are 109 ComplEx
pretraining units, run on a GPU cluster with H100s, and 185 query units, run on a second Slurm
cluster. A workstation bridge moves model-only exports from the pretraining site to the query site
and archives the query results back. No result has been produced by this package yet.

## Units

`units.py` takes the enumeration from `../production.py` and only normalizes condition names to the
worker's `encoder` (`shared`, `independent`, `mean`) and `curriculum` (`full13`, `core3`, `aux_off`)
values.

| stage | pretraining | query |
|---|---:|---:|
| 3: PFS, five conditions; folds 0–9 at seed 0 plus seeds 1–9 on fold 0 | 19 | 95 |
| 4: P, PF, PS; folds 0–9, seed 0 | 30 | 30 |
| 5: F, S, FS; folds 0–9, seed 0 | 30 | 30 |
| 6: excluded cohort F, S, FS; seeds 0–9 | 30 | 30 |

Each query unit depends only on its own pretraining unit. Summaries are kept in separate groups:
tenfold seed 0 (110 units), fold-0 ten seeds (50) and the excluded cohort ten seeds (30).

## Flow

1. `bridge.py fetch-hpo` copies the HPO decision files with hashes computed on the cluster.
2. `freeze.py check-hpo` refuses anything that is not a DONE, frozen, cross-checked and non-synthetic
   HPO outcome whose common leaderboard is the recorded calibrated choice.
3. `freeze.py protocols` runs `query/code/make_protocol.py generate` and `validate`, FROZEN only.
4. `freeze.py reuse-audit` (optional) compares the existing PFS fold-0 initializer and the HPO
   trajectory field by field. Missing evidence or any difference means the unit trains from scratch.
5. `freeze.py bundle` pins the committed source and writes `bundle/BUNDLE.json` together with the
   commands needed to check out the source and transfer the inputs.
6. On each site:
   - `campaign.py stage --bundle --deploy` verifies the checkout and every input's COMPLETE hash at
     the host paths;
   - `campaign.py release` re-reads the live partition, GPU type, QoS, grace and `sbatch --test-only`.
7. `bridge.py loop` runs short scheduling cycles: both ticks, both inventories, then transfer
   reconciliation. Transfers never run inside a cycle (see Transfers).

Configs are built just in time by the workers' own builders: `pretrain/make_config.py`, and
`query/code/make_config.py` through `build_query_config.py`. A query unit's config pins the hash of
the admitted export of its own parent.

## Runtime rules

| | pretraining site | query site |
|---|---|---|
| request | 3 h, typed H100 | 3 h, V100 or A100 |
| deadlines | save at 2 h 50, exit by 2 h 57 (no grace assumed) | 3 h 40 / 3 h 45 only with a live-validated 60-minute grace; inside the request otherwise |
| cap | `--max-active` (default 8, counted on the partition) | 24 jobs of this user, all projects |
| storage | reserve on the data area | admission uses free space minus reserve minus the growth owed by unfinished units |

- **Allocation chaining.** Each allocation runs the worker under `srun`. A successor is submitted
  (afterok) only after a clean deadline pause with real progress, inside the allocation bound,
  without a stop marker or halt, and when no other running job carries the unit's name.
- **Stops, signals and preemption.**
  - A stop marker (`campaign.py stop`) makes a pause STOPPED.
  - A clean worker pause on an unmarked signal (return 75; PAUSED, progress, checkpoint and log
    verified; progress made) is only a `SIGNAL_PAUSE_CANDIDATE`, and no successor is submitted from
    the signalled job.
  - Once the allocation is gone, the tick reads its terminal `sacct` state. Only `PREEMPTED`, with
    no stop marker and inside the per-unit bound, resumes, as a fresh submission with no dependency
    on the preempted job.
  - `CANCELLED`, `FAILED`, `TIMEOUT` or any worker error raises attention and is never revived.
- **Pretraining utilization guard.** The controller samples GPU utilization every 5 s. It warns
  after one hour below the partition threshold (75%, or 50% on the PI partition). It makes a clean
  stop if utilization stays low for 30 more minutes, or on gross underutilization or a stalled
  heartbeat.

## Transfers

Each delivery (pretraining export → query site → `campaign.py admit`) and each archive (query
results or HPO campaign → archive host → `verify-archive` → acknowledgement) runs in its own detached
worker process (`bridge.py transfer`).

- **Concurrency.** The worker holds an exclusive `flock` on its unit for the whole transfer, so a
  second worker for the same unit exits at once. It writes only its own state file under
  `state/transfers/`, so workers never share writable state. The kernel releases the lock if a
  worker dies.
- **Scheduling.** The scheduler holds its own lock only while cycling, never during a transfer.
  Each cycle it:
  - launches workers in two bounded pools, `max_delivery_workers` and `max_archive_workers`;
  - runs deliveries first, ordered by how many query units they unblock, and never starves them
    behind archives;
  - keeps every tick independent of transfer duration.
- **Failures and restarts.**
  - A failed attempt backs off exponentially (`transfer_backoff_seconds`).
  - After `transfer_attempt_limit` failures the unit raises attention and is not relaunched.
  - A killed worker is relaunched next cycle and resumes from its transit copy and the remote
    `.partial-*` directory (`rsync --partial`; files already matching their hash are skipped).
- **Admission.** Nothing is admitted or acknowledged unless every hash matches on the receiving
  host.
- **STOP.** A `STOP` file in the bridge state stops the scheduler and makes workers stop at their
  next file. The bridge never cancels a Slurm job.
- **Compression.** `rsync_compress` adds `--compress`.
- **Drain.** `bridge.py drain` cycles until nothing is running or launchable.

## Artifacts

- **Reportable artifacts** are `COMPLETE`, identity, progress, curve, selection, update and commit
  logs, sampler description, `fits/` (validation score matrices) and `models/` (initial, raw-best
  and calibrated-best weights).
- **Archive.** The bridge copies them to the archive area, hashes them locally, re-hashes them on
  the archive host (`verify-archive`, atomic rename) and records an acknowledgement.
- **Pruning.** After the acknowledgement, the query site removes only `checkpoints/` (full optimizer
  states) and writes the worker's `PRUNED.json`. The query site's copies of `fits/` and `models/` are
  removed only if a recorded storage decision (`clmpt-suite-storage-decision/1`, via
  `release --storage-decision` or `record-storage-decision`) sets
  `release_new_production_reportables_after_verified_archive` to true. Without one, a storage stall
  raises `storage_permission_required` instead of deleting anything.
- **Partial transfers.** Deliveries and archives land in `.partial-*` directories and are renamed
  only after every hash matches.
- **HPO campaign.** With `archive_current_hpo_reportables`, the bridge archives the bundle's own HPO
  campaign (records and every trial) to the archive host. `release-hpo-copies` (with `--dry-run`)
  deletes archived HPO score archives and model-only exports only with
  `release_current_hpo_reportables_after_verified_archive`. It never touches any other campaign or
  any directory production reuses.
- **Deferred analyses.** Robert replay and native baseline replay are recorded as not started. They
  are later analyses of the archived TEST matrices.

## Per-unit TEST inference

TEST is released for one query unit when both of these hold:

- the bundle carries a passed, non-synthetic HPO check, so the common calibrated recipe is globally
  frozen;
- the unit's training is complete, with `selection.json` and `curve.json` bound by its COMPLETE
  record and the raw and calibrated pointers equal to the frozen selection.

Other units do not need to have finished. The tick then:

1. freezes the selection (`SELECTION-FROZEN.json`);
2. writes a per-unit `TEST-RELEASE.json` and the `test_eval.py` config (views initial, raw,
   calibrated, terminal);
3. submits a short GPU job (`inference_job.py`, 1 h) inside the same personal-job cap.

The job runs three steps:

1. `export_terminal.py` saves the terminal model-only weights from the terminal full state, because
   the terminal checkpoint is not necessarily a selected one.
2. `query/code/test_eval.py` scores and saves the matrices.
3. `readout.py` produces every pool and label mask from those same matrices:
   - calibration (worker `calibrate`) is computed once on the full TEST population;
   - pools are column subsets ranked by the worker's `measure`;
   - the excluded S mask restricts only which associations are summarized.

A unit is archived only after training and inference are both verified, and pruned only after the
archive is acknowledged. No TEST result feeds any decision.

## FROZEN_HPO_ARTIFACT reuse

`freeze.py reuse-audit` (`reuse.py`) admits only two units:

- `main-PFS-f0-ps0`: the calibrated HPO winner's exact initializer, checked against the HPO
  manifest, its original delivery admission, graph, dimensions, numerics, recipe, budget and seed 0;
- `main-PFS-f0-qs0-full13-shared`: the winner's completed trajectory, only if its scientific
  config, core sources and references equal what production would run.

Legacy trainer and protocol hashes stay as provenance. Nothing is relabelled or continued, and old
files are only read. Reused units are counted separately from the 109/185 planned units.

## Orchestration

`orchestrate.py --config <reviewed config>` waits for the real HPO, then runs check-hpo, protocols,
reuse, bundle, a source/input gate, stage, release and the bridge. It stops at `GATE.json` (exit
code 3) when source synchronization or release approval is needed; re-running continues from there.

## Campaign paths

Campaign directories must be absolute and symlink-free (`campaign.py` refuses otherwise). For a
storage location reached through a symlink, put its resolved path in `DEPLOY.json`.

## Checks

`rehearse.py --scratch DIR --python <interpreter with torch> --output R.json` runs the whole
pipeline on CPU with fake Slurm and fake SSH, under a scratch directory, using the real v3 inputs.
It also uses a synthetic HPO campaign for the reuse, TEST, preemption and storage paths.
Rehearsal protocols are DRAFT relabelled in scratch only, and the bundle is marked rehearsal, so no
site releases it for GPU work.
