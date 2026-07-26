#!/bin/bash
# Ten-seed campaign on the excluded-gene benchmark.
#
# Trains ten seeds each of LinkGDA-f and LinkGDA-fs. If two GPUs are visible the two
# variants run in parallel, one per GPU; on a single GPU set CUDA_VISIBLE_DEVICES to
# pin both to it. Hyperparameters come from Supplementary Table 3 (dim 100, batch
# 16384, lr 1e-3) and are NOT retuned: the test set is held out, and fitting anything
# to it would spend the property that makes it worth running. See README,
# "Excluded-gene benchmark".
#
# Every run writes its own training log, and kge_transd.py writes per-instance ranks
# to data/results/, so metrics are recomputed from disk rather than scraped from stdout.
#
# Usage, from the excluded-benchmark working directory. Override CODE/PYTHON if the code
# checkout or interpreter are not the defaults:
#   bash /path/to/link-gda/run_excluded_seeds.sh
#   CODE=/path/to/link-gda PYTHON=/path/to/env/bin/python bash .../run_excluded_seeds.sh

set -uo pipefail

CODE=${CODE:-$(cd "$(dirname "$0")" && pwd)}   # link-gda code checkout (dir of this script)
PYTHON=${PYTHON:-python}                        # interpreter of the link-gda env
SEEDS=${SEEDS:-10}

mkdir -p logs data/models data/results

run_variant() {
    local gpu=$1 variant=$2 flags=$3
    for seed in $(seq 0 $((SEEDS - 1))); do
        local log="logs/train_${variant}_seed${seed}.log"
        echo "[gpu $gpu] LinkGDA-${variant} seed ${seed} -> ${log}"
        {
            echo "=== LinkGDA-${variant} seed ${seed} on GPU ${gpu} ==="
            echo "started $(date -Is)"
        } > "$log"

        CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$CODE/kge_transd.py" \
            --fold 0 \
            $flags \
            --projector_name owl2vecstar_gda \
            --embedding_dim 100 \
            --batch_size 16384 \
            --learning_rate 0.001 \
            --random_seed "$seed" \
            --use_graph \
            --no_sweep \
            --description "excluded-gene benchmark, LinkGDA-${variant}, seed ${seed}" \
            >> "$log" 2>&1

        local status=$?
        echo "finished $(date -Is) exit=${status}" >> "$log"
        if [ $status -ne 0 ]; then
            echo "[gpu $gpu] FAILED LinkGDA-${variant} seed ${seed} (exit ${status}), continuing"
        fi
    done
    echo "[gpu $gpu] done"
}

echo "campaign started $(date -Is)"
run_variant 0 f  "-func"          &
run_variant 1 fs "-func -site"    &
wait
echo "campaign finished $(date -Is)"
