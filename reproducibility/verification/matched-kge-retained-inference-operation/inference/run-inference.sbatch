#!/bin/bash
# ORIX template; validated original matched runtime and eight-thread setting.
#SBATCH --job-name=matched-retained-full-h200
#SBATCH --partition=freecycle-h200
#SBATCH --qos=freecycle
#SBATCH --account=pi-hohndor
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/mnt/data/u/hohndor/matched-kge-20261029/retained-full-replay-v3/inference-%A_%a.log
set -euo pipefail
work=/mnt/data/u/hohndor/matched-kge-20261029/retained-full-replay-v3
export TMPDIR="$work/scratch/$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
export MPLCONFIGDIR="$work/cache/matplotlib/$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
unset PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for ((fold = SLURM_ARRAY_TASK_ID; fold < 10; fold += 3)); do
  /home/hohndor/linkgda-env-20261013/.pixi/envs/default/bin/python -u "$work/run-inference.py" --task "$fold"
done
