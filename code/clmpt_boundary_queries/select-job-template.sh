#!/bin/bash
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -uo pipefail
unset CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
CAMPAIGN=$1
STAGE=$2
exec @PYTHON@ -I -u @CODE_DIR@/advance.py select --campaign "$CAMPAIGN" --stage "$STAGE"
