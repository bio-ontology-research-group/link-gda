# Sharing a GPU between matched KGE trajectories

`scripts/run_matched_kge_shared.py` runs three independent training trajectories
on one allocated GPU. Launch it outside the frozen scientific source checkout:

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/run_matched_kge_shared.py \
  --source-root "$FROZEN_SOURCE" --campaign-root "$CAMPAIGN" \
  --data-root "$BENCHMARK" --slots 3
```

Use a Slurm GPU allocation with `SLURM_JOB_ID`, `SLURM_RESTART_COUNT` (if
requeued), and `CUDA_VISIBLE_DEVICES`. The validated H200 allocation has 24
CPUs and 128 GB host memory. The batch wrapper must verify the frozen source
checksum manifest before launch. Stop older sequential workers first; all
concurrent workers must use the shared scheduling leases. Keep the launcher's
Python file outside `FROZEN_SOURCE/code`, because the trainer hashes that tree.

Each worker visits the fixed model/fold/graph order, claims unowned tasks, and
fills vacant slots. ConvKB and P-pool wait for their corresponding U1 training
completion. The frozen trainer still validates warm-start artifacts, scientific
identity, resumed optimizer/RNG states and selected checkpoints. Scheduling
uses `TRAINED.json` only as an eligibility marker; the separate global freeze
must validate all 100 trajectories before any test scoring.

In its default training phase, the scheduler preserves the original 1,000-epoch
budget, 20 profile epochs, validation/checkpoint frequency of 20, and every
trainer default. It does not choose checkpoints or freeze the campaign. Training
mode does not score tests. Task leases remain open in child processes if the supervisor dies. A terminal child failure or
exit without `TRAINED.json` records a persistent failure and stops that worker;
investigate the failure before explicitly clearing its failure marker.

Per-worker logs retain commands, launcher hash, child PIDs, outcomes, and GPU
utilization. `no_unclaimed_work` means the worker has no available task; another
worker may still own unfinished work. It is not a campaign completion claim.

The real allocation pilot ran three trajectories for 390 seconds with mean
sampled GPU utilization 87.7%, including startup and shutdown. It resumed T1
from epoch 760 to 800 and advanced two U1 trajectories to epoch 40 with complete
scheduled validation. This measures allocation feasibility, not model quality.
Long production runs must still be monitored for resource utilization.

## Shared inference after global freeze

After all 100 training trajectories finish, run the original trainer's separate
freeze command from [KGE execution](kge-execution.md). A training worker's
`no_unclaimed_work` status is insufficient: another worker may still own runs.
The freeze must cover all ten folds, five model families and both graphs.

Use the same immutable scientific source, prepared inputs, annotation table and
[pinned environment](../reproducibility/environments/matched-kge/README.md).
Launch scoring with the operational scheduler outside the frozen source tree:

```bash
python scripts/run_matched_kge_shared.py --phase score \
  --source-root "$FROZEN_SOURCE" --campaign-root "$CAMPAIGN" \
  --data-root "$BENCHMARK" --slots 3
```

The scheduler requires the complete global freeze before launching any child.
It verifies all 100 frozen training-record hashes and checks that the freeze
remains unchanged throughout execution. Each child invokes the original
trainer's `--phase score` command with the original budgets; that trainer checks
its scientific inputs, source, environment, selected weights and freeze before
inference. Scoring has separate leases, failure records and worker logs under
`shared-score-workers/`. A child must write `SCORED.json` before its task is
counted as finished. Already marked tasks are skipped for scheduling; the final
manifest builder must verify every score file and completion record before any
result is accepted.

The launcher supports one to three concurrent scoring processes. Its lifecycle,
freeze rejection, ownership and completion behavior have passed CPU subprocess
checks. Production GPU inference throughput, memory use and numerical replay
remain unverified until the full campaign is frozen. Monitor the scoring
allocation's recorded `gpu.csv` and resource use before scaling workers. The
training utilization pilot does not establish scoring utilization.

After workers finish, run the complete 400-score manifest builder, evaluator
and prespecified statistical summary from [KGE execution](kge-execution.md).
A worker completion file alone never certifies the full result collection.

The submitted [ORIX batch wrapper](../scripts/orix-matched-kge-score.sbatch)
records the actual campaign paths, environment, source checks and allocation.
It is queued after successful completion of every production training worker;
see [submission evidence](../reproducibility/verification/matched-kge-score-scheduled.json).
It creates the global freeze only when absent and reuses it on restarts.
These deployment paths are specific to this campaign; use the portable command
above with your own paths for an independent reconstruction. The queued job
must be cancelled if its parent campaign is invalidated or replaced.

Three additional workers were submitted using the unchanged training launcher
from commit `a16c946`, bringing the scheduled total to five allocations (up to
15 trajectories if all receive GPUs). Each new worker was submitted on hold;
the scoring dependency was extended and verified before release. See
[expansion evidence](../reproducibility/verification/matched-kge-expanded-training.json).
The three never-started query scoring tasks were deferred to free submission
slots. Their unchanged array must be resubmitted as query training jobs finish.
Any further training worker must be added to the scoring dependency before it
is released; the complete 100-run freeze remains mandatory.
