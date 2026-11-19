# clmpt_boundary: ComplEx pretraining for the original-taxonomy boundary expansion

This package pretrains the ComplEx initializers used by the CLMPT query stage. It extends the
learning-rate and update-budget boundaries of the original-taxonomy recipe. Taxonomy closure is
not used.

## Design

| learning rate | start | milestones (updates) | epochs |
|---|---|---|---|
| 0.2 | exact full state of the completed 633,450-update run | 844,600; 1,055,750 | 200; 250 |
| 0.3 | fresh, seed 0 | 633,450; 844,600; 1,055,750 | 150; 200; 250 |
| 0.4 | fresh, seed 0 | 633,450; 844,600; 1,055,750 | 150; 200; 250 |

All runs use rank 400, batch size 1,000, FP32 without TF32, deterministic algorithms, Adagrad
and N3 weight 0.05 on the fold 0 original graph (4,222,290 training triples, 4,223 updates per
epoch). The learning-rate 0.2 run at 633,450 updates already exists and is reused as a control,
so the campaign adds eight new initializers. `protocol.json` freezes this design together with
the admitted graph and parent hashes.

## Files

| file | role |
|---|---|
| `complex_model.py` | ComplEx scorer and N3, byte-identical to the source of the reused parent |
| `state_compare.py` | recursive exact comparison of saved and restored states, byte-identical |
| `train.py` | pretraining worker: exact full-state checkpoints, resume and parent import |
| `controller.py` | one Slurm allocation: deadline pause, bounded continuation, utilization and stall guards |
| `stage.py` | writes the campaign protocol copy, configs, batch script, release template and manifest |
| `submit.py` | one-time release (with `sbatch --test-only`) and wave-0 submission |
| `monitor.py` | read-only status of queue, progress, guard markers and GPU utilization |
| `protocol.json` | frozen protocol |

### What `train.py` keeps from the parent's trainer

The model, cross-entropy over all entities, N3, Adagrad update, seed-0 initialization, CPU
permutation of the training triples, batch slicing and finite checks are those of the trainer
that produced the parent (sha256 `af8240656e139a6a52d87e85b23078741c2102c3addc733a9d49e9e77929f95c`).
Only runtime behaviour differs:

- The update log is a buffered stream. It is flushed and fsynced before every checkpoint, pause
  and completion, so each checkpoint still records an exact log prefix length and hash.
- The heartbeat is written every 10 seconds instead of after every update.
- The periodic checkpoint interval (`checkpoint_every_seconds`) and the per-process time bound are
  configuration values.
- An exact import of a completed parent's terminal full state is allowed only for a parent listed
  in `protocol.json`. The import checks the parent's completion record, identity, progress,
  checkpoint and log-prefix hashes, restores model, Adagrad accumulators, epoch order and
  position and all RNG states, and proves the restored state equal to the saved one before any
  update.

Restoring a full state saved on a different GPU model keeps the optimizer, RNG and batch order
exact, but floating-point kernels differ between GPU models. The continued trajectory is
therefore not bitwise identical to a continuation on the original hardware.

## Requirements

Python 3.11 with `torch==2.6.0` (CUDA 12.4 build) and `numpy==1.26.4`; see `requirements-pretrain.txt`.
The controller, stager, submitter and monitor use only the standard library and Slurm commands.

## Running

Stage a campaign directory that already holds the graph and the parent run:

```bash
python code/clmpt_boundary/stage.py --campaign CAMPAIGN \
  --graph-dir CAMPAIGN/inputs/graph-original-fold0 \
  --parent-dir CAMPAIGN/parent/S1-original-plr0.2-n3r0.05 \
  --python PYTHON --code-revision "$(git rev-parse HEAD)" \
  --partition PARTITION --qos QOS --account ACCOUNT \
  --gpu-request GPU_GRES --gpu-name-substring GPU_NAME
```

Staging refuses unless `--code-revision` equals `HEAD`, `HEAD` equals its upstream (the pushed
commit), and this package directory has no modified, untracked or ignored files. The manifest
records the Git blob ID of every package file, and each allocation rechecks the file hashes. Do
not pull or edit the checkout while a campaign runs: a changed package makes every successor
stop.

Release once and submit the three trajectories. Approval runs `sbatch --test-only` for both the
initial and the long time limit, so partition and QoS limits are checked by the scheduler, and
writes nothing if either is rejected:

```bash
python code/clmpt_boundary/submit.py --campaign CAMPAIGN --approve --authority "WHO APPROVED WHAT" \
  --trial CAMPAIGN/trials/lr0.2-resume633450 --trial CAMPAIGN/trials/lr0.3-fresh --trial CAMPAIGN/trials/lr0.4-fresh
```

Check status:

```bash
python code/clmpt_boundary/monitor.py --campaign CAMPAIGN
```

Each trajectory writes to `CAMPAIGN/trials/NAME/training/`:

| output | content |
|---|---|
| `updates.jsonl` | one line per update: step, epoch, position, batch index hash, fit, N3, loss, timing |
| `checkpoints/step-*.pt` | the two most recent exact full states |
| `models/update-*.pt` | model-only milestone states for the query stage |
| `progress.json`, `commits.jsonl` | current and historical checkpoint records with hashes |
| `IMPORT.json` | parent import proof (learning rate 0.2 only) |
| `restore-proof-*.json` | exact restore comparison at every resume |
| `COMPLETE.json` | final record with hashes of all retained files |

Each allocation writes `CAMPAIGN/trials/NAME/allocations/wave-NNN/` with the GPU telemetry
(`gpu-telemetry.csv`), `utilization-summary.json` and any guard marker.

## Allocation policy

The controller derives the pause time from the job's own start time and time limit: the worker
commits at the limit minus 10 minutes, and the controller exits by the limit minus 3 minutes.
After a clean pause it verifies the checkpoint and log hashes and submits one successor with an
`afterok` dependency, up to the trajectory's allocation bound. A signal, a real error, an
operator stop file (`OPERATOR_STOP.json` in the campaign or trial directory) or a guard stop
leaves the last exact state in place and submits nothing. Stops reach the worker as SIGTERM
through its Slurm step, or through the `srun` process when the step ID is unknown, so it can
commit. Shutting down the telemetry sampler never changes the controller's exit status, so a
queued `afterok` successor cannot be stranded by cleanup.

The guards read GPU utilization sampled every 5 seconds over the whole allocation, including
startup and checkpoint pauses:

- a full 60-minute window below 75% writes `UTILIZATION_WARNING.json`;
- if utilization stays below 75% for the next 30 minutes, the worker commits and stops, and
  `UTILIZATION_STOP.json` records why;
- a 20-minute mean below 60% after the first 25 minutes stops the run the same way;
- a worker heartbeat older than 15 minutes stops the run (`STALL_STOP.json`).

A successor requests the longer time limit only if `LONG_CONTINUATION.json` approves it and the
finished allocation met every criterion in `successor-policy.json`: allocation mean and every
complete hour at or above 75%, rolling-hour minimum at or above 75%, at least 90% sample
coverage, no warning or telemetry gap in the trajectory, and a partition maximum time that
admits the request. Otherwise the successor keeps the initial limit.
