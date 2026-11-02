# Slurm checkpoint-resume chain for compositional GDA training

These files run compositional GDA training runs on a Slurm cluster as a chain of
GPU segments. A segment that reaches its time budget saves a checkpoint and exits;
the next segment resumes it in a new job. They are portable templates: nothing in
them names a cluster, account or user.

| file | role |
|---|---|
| `chain.py` | controller (standard library only): freezes a campaign, submits waves, classifies finished tasks, arms the next wave |
| `train-segment.sbatch` | GPU array task: one segment of one run |
| `continue.sbatch` | CPU job that runs `chain.py controller` after each wave |
| `campaign.example.json` | campaign spec template |

## Trainer contract

Each task runs

```
python code/compositional_gda/train.py --config CONFIG --run-dir RUN_DIR \
    --device cuda --deadline-seconds 11400 [trainer_args...]
```

and the chain relies on this behaviour of the trainer:

- exit 0 with `RUN_DIR/COMPLETE.json` written last when the fixed update budget is done;
- exit 75 after committing `RUN_DIR/resume.pt` when it pauses at its deadline or on SIGUSR1;
- `RUN_DIR/progress.json` holds the committed step as an integer `step` (and, optionally, the
  validation curve as a list `curve`). It is written only after the matching `resume.pt` is
  committed, so its `step` never runs ahead of the checkpoint. The chain cannot read
  `resume.pt` itself; it relies on this ordering to avoid resuming from a stale checkpoint;
- started again on the same `RUN_DIR`, it resumes from `resume.pt` without extra flags, and it
  exits non-zero (classified `failed`, never resumed) when the configuration, trainer source,
  package versions or prepared data differ from those recorded in the run directory.

## Resources and timing

The GPU template requests one GPU with `--constraint=v100|a100` and `--time=03:30:00`
(12600 s). The trainer's internal deadline (11400 s by default) leaves 1200 s to finish the
current step chunk and commit `resume.pt`. Slurm sends SIGUSR1 600 s before the limit, after that
deadline, as a backup. The batch shell `exec`s `chain.py exec-task`, which `exec`s the trainer,
so the trainer keeps the shell's PID and receives the signal. CPU count and memory in the
template (4 CPUs, 32 GB) are starting values; adjust them to the measured footprint.

`deadline_seconds` may not exceed 11400 (the limit less the 1200 s margin).
`chain.py init` refuses specs that try to override `--time`, `--constraint`, `--gres`,
`--signal` or the options the controller sets itself. Site options such as partition or account
go in `sbatch_gpu_args` and `sbatch_cpu_args`.

## Campaign layout

```
CAMPAIGN/
  campaign.json            frozen spec: runs, config SHA-256s, code root, python, cap
  configs/<run>.json       frozen config copies passed to the trainer
  runs/<run>/              trainer run directories
  logs/<token>-<array>_<task>.out, logs/<token>-ctl-<job>.out
  chain/wave_NNN.json      one record per wave: intent, submissions, outcomes
  chain/events.jsonl       append-only event log
  chain/segments/<run>/    per-segment record: host, GPU, start progress, argv
  chain/locks/<run>.lock   held by the running trainer; blocks duplicate segments
  chain/failures/          controller failures recorded outside a wave record
  status.json              campaign state, recomputed after every action
```

## Usage

```
python scripts/compositional/chain.py init SPEC.json --campaign CAMPAIGN
python scripts/compositional/chain.py start --campaign CAMPAIGN --dry-run
python scripts/compositional/chain.py start --campaign CAMPAIGN
python scripts/compositional/chain.py status --campaign CAMPAIGN
```

Relative config paths in the spec are resolved from the spec's directory. `init` refuses a
non-empty campaign directory; campaigns are never reused. After `start`, the chain runs without
intervention until every run is complete or needs attention.

## How a wave is handled

1. `controller` takes the campaign lock and verifies that the finished array matches the wave
   record. It polls `sacct` (25 polls, 20 s apart, so up to 8 minutes) until every task is terminal; a failing
   `sacct` call counts as one incomplete poll. It submits nothing if accounting stays
   incomplete.
2. Each task is classified:

   | outcome | condition | resumed |
   |---|---|---|
   | `completed` | `COMPLETE.json` present | no |
   | `resume` | exit 75, or `TIMEOUT`/`PREEMPTED`/`NODE_FAIL`/`BOOT_FAIL`, with `resume.pt`, readable progress, and progress beyond the wave's start | yes |
   | `stalled` | as `resume` but no new step or validation | no |
   | `no_checkpoint`, `progress_unreadable` | paused or recoverable state without `resume.pt` or readable `progress.json` | no |
   | `failed_oom` | `OUT_OF_MEMORY`, or a CUDA out-of-memory message in the task log | no |
   | `cancelled` | `CANCELLED` | no |
   | `exit0_without_complete` | exit 0 without `COMPLETE.json` | no |
   | `failed` | anything else | no |
   | `cap_reached` | would resume, but the next wave would exceed `max_waves` | no |

3. The outcomes are written to the wave record before anything is submitted. Runs classified
   `resume` form the next wave. The wave is marked handled only after the next wave and its
   controller are both submitted.

Out-of-memory failures and ordinary errors are never retried, so a deterministic failure
cannot loop.

## Controller failures are visible

The controller never leaves a wave looking `running` after it stops. Any exception (incomplete
accounting, a Slurm client error, a lock timeout, an unreadable record, a mismatched array ID)
and a `SIGTERM` at the controller's time limit are written as `problem` on the wave record
under the campaign lock. When the lock or the record is unavailable, the failure goes to
`chain/failures/` instead. Either way `status.json` becomes `needs_attention` with the wave in
`problems`, and the controller exits 1. A controller killed outright (`SIGKILL`, node failure,
cancellation) cannot record anything; `chain.py status` therefore asks `sacct` about the
controller job of every unhandled wave and flags the wave when that job has ended. Use
`status --no-slurm` to skip that query. A later successful `controller` run clears the
problem.

## Idempotent submission

- The record of wave N+1 is written to a temporary file and hard-linked into place, so it is
  published complete and only once; it carries a unique job-name token. A second controller run
  finds the record and resumes it instead of creating another. The campaign filesystem must
  support hard links; `init` fails at once if it does not.
- The attempt counter is saved before each `sbatch` call. Before any further call the
  controller looks up the token in `squeue` and `sacct`. One match is adopted. More than one
  match stops the chain as `duplicate`.
- What happens when no match is found depends on how the previous attempt ended:
  - `sbatch` could not be executed, or it reported a definite refusal (an invalid option,
    partition, account or feature, or a policy limit): a new attempt with the same token is
    allowed, up to three in total.
  - Anything else is ambiguous: a time-out, a lost connection, an unparseable reply, or a crash
    mid-call. Slurm may have accepted the job even though `squeue`/`sacct` do not show it yet.
    The controller searches over the whole reconciliation window (6 polls, 20 s apart). If the
    job is still not found, it marks the submission `unresolved` and stops with
    `needs_attention`. It does not resubmit.
  - A failed lookup is never taken as absence.
- A wave whose array was submitted but whose controller was not is reported in `status.json`
  as `needs_attention`. `chain.py arm --campaign CAMPAIGN --wave N` completes it.
- `exec-task` holds `chain/locks/<run>.lock` across `exec`, so the trainer holds it as well.
  A second segment of the same run exits 3 instead of training.

`sbatch` is called with the caller's `SLURM_*` variables removed, so a controller job does not
pass its own allocation to the jobs it submits.

## Manual recovery

| `status.json` shows | action |
|---|---|
| `running` / `running_with_attention` | wait; read `attention` runs' logs |
| `needs_attention` with `wave not fully submitted` | `chain.py arm --campaign C --wave N` |
| `submission unresolved` | check `squeue --name TOKEN` and `sacct --name TOKEN`; `chain.py arm --campaign C --wave N` adopts a job that has appeared; once it is certain none exists, add `--confirm-absent` |
| `controller failed: ...` or `controller job ... ended` | fix the cause, then `chain.py controller --campaign C --wave N --after ARRAY_JOB_ID` |
| `next wave not armed` | resolve the next wave with `arm` as above, then rerun `controller` for wave N so it is marked handled |
| `cap_reached`, `stalled`, `failed*` | inspect the run; start a new campaign if a fix is needed |

These commands are light (no training) and can run where `sbatch` and `sacct` are available.

## Tests

```
python -m pytest tests/test_compositional_chain.py
```

The tests use a fake Slurm and a stub trainer; they submit no jobs.
