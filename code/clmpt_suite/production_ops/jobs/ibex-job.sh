#!/bin/bash
set -uo pipefail
unset SLURM_TIME_FORMAT
export PYTHONDONTWRITEBYTECODE=1
@ENVIRONMENT@
CAMPAIGN=$1
UNIT=$2
WAVE=$3
exec @PYTHON@ -u @CODE_DIR@/controller.py --campaign "$CAMPAIGN" --unit "$UNIT" --wave "$WAVE"
