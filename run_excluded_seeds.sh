#!/bin/bash
# Ten-seed campaign on the excluded-gene benchmark.
#
# Trains ten seeds each of LinkGDA-f and LinkGDA-fs. If two GPUs are visible the two
# variants run in parallel, one per GPU; on a single GPU set CUDA_VISIBLE_DEVICES to
# pin both to it. The tuned hyperparameters come from Supplementary Table 3 (dim 100,
# batch 16384, lr 1e-3) and are NOT retuned: the test set is held out, and fitting
# anything to it would spend the property that makes it worth running. See README,
# "Excluded-gene benchmark".
#
# VAL_SEED fixes the train/validation split. This benchmark has one fold, so without it
# every seed holds out different diseases and the spread mixes resampling with
# initialization.
#
# TOLERANCE is 15, not the trainer's default of 5. At 5 the runs stopped near epoch 360
# while validation mean rank was still falling, so the spread across seeds tracked the
# stopping rule rather than initialization. We read that off the validation curves, not
# the 409 test pairs. Give each value its own tol<N>/ directory (the checkpoint filename
# does not encode tolerance) and compare arms with compare_tolerance_arms.py.
#
# Every run writes its own training log, and kge_transd.py writes per-instance ranks
# to data/results/, so metrics are recomputed from disk rather than scraped from stdout.
#
# Usage, from the excluded-benchmark working directory. Override CODE/PYTHON if the code
# checkout or interpreter are not the defaults:
#   bash /path/to/link-gda/run_excluded_seeds.sh
#   CODE=/path/to/link-gda PYTHON=/path/to/env/bin/python bash .../run_excluded_seeds.sh
#   TOLERANCE=5 bash .../run_excluded_seeds.sh    # the other arm, into its own tol5/ dir

set -uo pipefail

CODE=${CODE:-$(cd "$(dirname "$0")" && pwd)}   # link-gda code checkout (dir of this script)
PYTHON=${PYTHON:-python}                        # interpreter of the link-gda env
SEEDS=${SEEDS:-10}
VAL_SEED=${VAL_SEED:-0}                         # fixed train/validation split, shared by all seeds
TOLERANCE=${TOLERANCE:-15}                      # early-stopping patience, in validation evaluations

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
            --val_seed "$VAL_SEED" \
            --tolerance "$TOLERANCE" \
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
