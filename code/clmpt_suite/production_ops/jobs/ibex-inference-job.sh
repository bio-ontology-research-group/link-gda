#!/bin/bash
set -uo pipefail
unset SLURM_TIME_FORMAT
export PYTHONDONTWRITEBYTECODE=1
@ENVIRONMENT@
CAMPAIGN=$1
UNIT=$2
ATTEMPT=$3
exec @PYTHON@ -u @CODE_DIR@/inference_job.py --campaign "$CAMPAIGN" --unit "$UNIT" --attempt "$ATTEMPT"
