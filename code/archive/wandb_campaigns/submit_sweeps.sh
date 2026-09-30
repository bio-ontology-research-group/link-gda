#!/bin/bash
# Submit one sbatch array job per sweep ID stored in
# wandb_scripts/sweep_ids.yaml under the requested top-level key.
#
# Usage:
#   ./submit_sweeps.sh                       # defaults: hpo_rq1, array 0-35
#   ./submit_sweeps.sh hpo_rq1               # explicit key
#   ./submit_sweeps.sh hpo_rq1 0-39          # custom array range
#
# Each HPO sweep is a 2x3x3x2x1 = 36-config grid, so an array spanning
# 0-35 covers it exactly; widen the range to allow retries.

set -euo pipefail

KEY="${1:-hpo_rq1}"
ARRAY_SPEC="${2:-0-35}"
IDS_FILE="wandb_scripts/sweep_ids.yaml"

if [ ! -f "$IDS_FILE" ]; then
    echo "Missing $IDS_FILE. Run wandb_scripts/create_sweeps.py first." >&2
    exit 1
fi

if [ ! -x "run_sweep.sh" ]; then
    echo "run_sweep.sh not executable in $(pwd)." >&2
    exit 1
fi

mkdir -p out err

mapfile -t entries < <(python3 - "$IDS_FILE" "$KEY" <<'PY'
import sys, yaml
ids_file, key = sys.argv[1], sys.argv[2]
with open(ids_file) as f:
    data = yaml.safe_load(f) or {}
section = data.get(key) or {}
if not section:
    sys.stderr.write(f"No entries under '{key}' in {ids_file}\n")
    sys.exit(2)
for name, sid in section.items():
    print(f"{name}\t{sid}")
PY
)

if [ ${#entries[@]} -eq 0 ]; then
    exit 1
fi

echo "Submitting ${#entries[@]} sweep(s) under '$KEY' with --array=$ARRAY_SPEC"
for entry in "${entries[@]}"; do
    name="${entry%%$'\t'*}"
    sid="${entry##*$'\t'}"
    printf '  %-12s %s\n' "$name" "$sid"
    sbatch --array="$ARRAY_SPEC" run_sweep.sh "$sid"
done
