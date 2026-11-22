#!/bin/bash
#SBATCH --partition=batch
#SBATCH --constraint=[v100|a100]
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
set -uo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CAMPAIGN=$1
TRIAL=$2
WAVE=$3
exec @PYTHON@ -u @CODE_DIR@/controller.py --campaign "$CAMPAIGN" --trial "$TRIAL" --wave "$WAVE"
