#!/bin/bash
# Submit with afterok dependencies on both shared production training workers.
# All 100 selections are frozen before any test-score command is started.
#SBATCH --job-name=matched-kge-score
#SBATCH --partition=freecycle-h200
#SBATCH --qos=freecycle
#SBATCH --account=pi-hohndor
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/mnt/data/u/hohndor/matched-kge-20261029/production-v1/logs/%x-%j.out
set -euo pipefail
root=/mnt/data/u/hohndor/matched-kge-20261029/production-v1
export PATH=/home/hohndor/linkgda-env-20261013/.pixi/envs/default/bin:$PATH
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$root/scratch/$SLURM_JOB_ID"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
exec 9>"$root/scoring-allocation.lock"
flock -n 9
cd "$root/source"
sha256sum -c "$root/source.sha256" > /dev/null
(cd "$root" && sha256sum -c <<'CHECKSUM'
2bd5d15005352f9a744184c37fa1d09093761d31ad74f99843e3ea07c0e57593  run_matched_kge_shared_scoring.py
CHECKSUM
)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# A restart must reuse the immutable freeze once test scores exist.
if [ ! -e "$root/runs/FROZEN.json" ]; then
  python -u code/training/run_matched_kge.py --phase freeze --root "$root/runs"
fi
python -u "$root/run_matched_kge_shared_scoring.py" --phase score \
  --source-root "$root/source" --campaign-root "$root" \
  --data-root /mnt/data/u/hohndor/ultra-top-hits-20261023/data --slots 3
