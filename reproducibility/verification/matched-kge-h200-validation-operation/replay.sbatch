#!/bin/bash
# ORIX template: original H200 matched runtime, eight allocated CPUs and threads.
#SBATCH --job-name=matched-public-val-h200
#SBATCH --partition=freecycle-h200
#SBATCH --qos=freecycle
#SBATCH --account=pi-hohndor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/u/hohndor/matched-kge-20261029/public-validation-v1/validation-%j.log
set -euo pipefail
work=/mnt/data/u/hohndor/matched-kge-20261029/public-validation-v1
export TMPDIR="$work/scratch/$SLURM_JOB_ID"
export MPLCONFIGDIR="$work/cache/matplotlib/$SLURM_JOB_ID"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
unset PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
/home/hohndor/linkgda-env-20261013/.pixi/envs/default/bin/python -u "$work/replay.py"
