#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100|a100
#SBATCH --time=03:30:00
#SBATCH --signal=B:USR1@600

set -euo pipefail
: "${CGDA_PYTHON:?set by chain.py}"
: "${CGDA_CODE_ROOT:?set by chain.py}"
: "${CGDA_CAMPAIGN:?set by chain.py}"
: "${CGDA_WAVE:?set by chain.py}"
: "${SLURM_ARRAY_TASK_ID:?run as an array task}"

threads=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

printf 'job=%s array=%s task=%s host=%s wave=%s start=%s\n' \
  "${SLURM_JOB_ID:-}" "${SLURM_ARRAY_JOB_ID:-}" "$SLURM_ARRAY_TASK_ID" "$(hostname)" \
  "$CGDA_WAVE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

exec "$CGDA_PYTHON" "$CGDA_CODE_ROOT/scripts/compositional/chain.py" exec-task \
  --campaign "$CGDA_CAMPAIGN" --wave "$CGDA_WAVE"
