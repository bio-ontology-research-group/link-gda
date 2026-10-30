#!/bin/bash
#SBATCH --job-name=matched-validation-inference
#SBATCH --account=pi-hohndor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029/release-workspace/interim-matched-artifact-check/replay-%j.log
set -euo pipefail
base=/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029
work="$base/release-workspace/interim-matched-artifact-check"
export TMPDIR="$work/scratch/$SLURM_JOB_ID"
export MPLCONFIGDIR="$work/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
unset PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$work/gpu-$SLURM_JOB_ID.txt"
"$base/inference-environments-v1/matched-kge/.pixi/envs/default/bin/python" -u "$work/replay.py"
