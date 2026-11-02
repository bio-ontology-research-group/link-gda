#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00

set -euo pipefail
: "${CGDA_PYTHON:?set by chain.py}"
: "${CGDA_CODE_ROOT:?set by chain.py}"
: "${CGDA_CAMPAIGN:?set by chain.py}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

exec "$CGDA_PYTHON" "$CGDA_CODE_ROOT/scripts/compositional/chain.py" controller \
  --campaign "$CGDA_CAMPAIGN" "$@"
