#!/bin/bash
# Three seed jobs each score both initializations using the original 240-run freeze.
# Submit with afterok dependencies on all six production training jobs.
#SBATCH --job-name=phenoquery-score-parallel
#SBATCH --partition=freecycle-h200
#SBATCH --qos=freecycle
#SBATCH --array=0-2
#SBATCH --account=pi-hohndor
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/mnt/data/u/hohndor/phenoquery-controlled-20261028/pilot-v3/logs/%x-%A_%a.out
set -euo pipefail
project=/mnt/data/u/hohndor/phenoquery-controlled-20261028/pilot-v3
base=/mnt/data/u/hohndor/ultra-top-hits-20261023
export CUDA_HOME=/home/hohndor/linkgda-ultra-20261013/ultra-env
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc" CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$base/torch-extensions-gpu"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MAX_JOBS=8
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$project/scratch/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
(cd "$project" && sha256sum -c SNAPSHOT.sha256 > /dev/null)
(cd "$base/source/ULTRA" && sha256sum -c ../ULTRA-files.sha256 > /dev/null)
(cd "$base/source/ULTRA/ckpts" && sha256sum -c <<'CHECKPOINTS'
48a046e708adf5632d87c30eacae01f5f51466b2301effdc2cb42358d22854e0  ultra_4g.pth
9b6dc20801c35e7c9ebe65764acb4ed9cb5bffc6b0a0a727bd48935589e9388d  ultraquery.pth
CHECKPOINTS
)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
(cd "$project" && sha256sum -c <<'LAUNCHER'
42cbbf2beb67a6a6284bd51a89d17d3354d88af5b118e9e917459522f59a2dc9  run_controlled_query_scoring.py
LAUNCHER
)
seed=${SLURM_ARRAY_TASK_ID:?}
if (( seed < 0 || seed > 2 )); then exit 2; fi
for initialization in ultra_4g ultraquery; do
  python -u "$project/run_controlled_query_scoring.py" \
    --source-root "$project" --root "$project/runs" \
    --prepared-root /mnt/data/u/hohndor/ultra-eq-query-20261026 \
    --base "$base" --ultra-root "$base/source/ULTRA" \
    --initialization "$initialization" --seed "$seed"
done
