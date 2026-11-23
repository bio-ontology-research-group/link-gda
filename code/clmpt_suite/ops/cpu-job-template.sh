#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -uo pipefail
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
CAMPAIGN=$1
ACTION=$2
TARGET=$3
exec @PYTHON@ -I -u @CODE_DIR@/advance.py "$ACTION" --campaign "$CAMPAIGN" --target "$TARGET"
