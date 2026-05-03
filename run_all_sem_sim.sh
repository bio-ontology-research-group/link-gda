#!/bin/bash
# Run all semantic similarity measures × 10 folds.
# Each measure iterates folds 0..9 sequentially in its own process;
# the 5 measures run in parallel.

set -u
cd "$(dirname "$0")"

mkdir -p logs/sem_sim data/baseline_results

run_pairwise () {
    local pw="$1"
    local gw="$2"
    local tag="resnik_${pw}_${gw}"
    for fold in 0 1 2 3 4 5 6 7 8 9; do
        local log="logs/sem_sim/${tag}_fold${fold}.log"
        echo "[$(date -Is)] START ${tag} fold=${fold}" >> "logs/sem_sim/${tag}.master.log"
        groovy semantic_similarity.groovy \
            -r data -ic resnik -pw "${pw}" -gw "${gw}" -fold "${fold}" \
            > "${log}" 2>&1
        echo "[$(date -Is)] END   ${tag} fold=${fold} rc=$?" >> "logs/sem_sim/${tag}.master.log"
    done
}

run_simgic () {
    local tag="resnik_simgic"
    for fold in 0 1 2 3 4 5 6 7 8 9; do
        local log="logs/sem_sim/${tag}_fold${fold}.log"
        echo "[$(date -Is)] START ${tag} fold=${fold}" >> "logs/sem_sim/${tag}.master.log"
        groovy semantic_similarity_simgic.groovy \
            -r data -ic resnik -fold "${fold}" \
            > "${log}" 2>&1
        echo "[$(date -Is)] END   ${tag} fold=${fold} rc=$?" >> "logs/sem_sim/${tag}.master.log"
    done
}

run_pairwise resnik bma &
PID_RBMA=$!
run_pairwise resnik bmm &
PID_RBMM=$!
run_pairwise lin    bma &
PID_LBMA=$!
run_pairwise lin    bmm &
PID_LBMM=$!
run_simgic              &
PID_SIMG=$!

echo "PIDs: rbma=${PID_RBMA} rbmm=${PID_RBMM} lbma=${PID_LBMA} lbmm=${PID_LBMM} simgic=${PID_SIMG}"

wait ${PID_RBMA} ${PID_RBMM} ${PID_LBMA} ${PID_LBMM} ${PID_SIMG}
echo "[$(date -Is)] all measures complete"
