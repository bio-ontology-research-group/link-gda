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

The scheduler preserves the original 1,000-epoch budget, 20 profile epochs,
validation/checkpoint frequency of 20, and every trainer default. It does not
choose checkpoints, freeze the campaign, or score tests. Task leases remain
open in child processes if the supervisor dies. A terminal child failure or
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
