#!/usr/bin/env bash
# Score every exported fold with the zero-shot ULTRA checkpoint and compute metrics.
#
# CUDA_HOME points at the env prefix because the host has no system toolkit; the
# rspmm kernel is JIT compiled on first use and cached in TORCH_EXTENSIONS_DIR,
# also inside the env, so nothing is written outside it. nvcc 12.1 refuses gcc 13,
# which is the system compiler here, so CC/CXX are the env's gcc 11.
#
# One GPU by design: the driver is a single-process scoring loop and the second
# card is left free.
#
# Zero-shot inference is deterministic -- no training, no sampling, a fixed
# checkpoint and a fixed graph -- so one pass per configuration is the whole result.
# seed_0 stays in the identifier to keep the filename shape of the trained runs, and
# carries no meaning here.
#
# -u is deliberately absent: the conda gcc_linux-64 activation hook dereferences
# SYS_SYSROOT unguarded, which aborts the shell under nounset before the env is
# usable.
set -eo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
SCRATCH="${SCRATCH:-$HOME/Git/link-gda-ultra}"
CKPT="${CKPT:-$SCRATCH/ULTRA/ckpts/ultra_4g.pth}"
BATCH="${BATCH:-8}"
BENCH="${BENCH:-main}"
VARIANTS="${VARIANTS:-pfs}"
PROJECTIONS="${PROJECTIONS:-owl2vecstar owl2vecstar_gda}"
METRICS="${METRICS:-1}"
ORDER="${ORDER:-forward}"
PEER_LOG="${PEER_LOG:-}"

variant_source() {
    case "$1" in
        pfs) echo "pheno_func_expr" ;;
        f)   echo "func" ;;
        s)   echo "expr" ;;
        fs)  echo "func_expr" ;;
        *)   echo "unknown variant $1" >&2; exit 1 ;;
    esac
}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ultra-linkgda

export CUDA_HOME="$CONDA_PREFIX"
export TORCH_EXTENSIONS_DIR="$CONDA_PREFIX/torch_extensions"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ "$BENCH" = "main" ]; then
    WORKDIR="$SCRATCH"
    FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
    PREFIX=""
    IDENT_BENCH=""
else
    WORKDIR="$SCRATCH/excluded"
    FOLDS="${FOLDS:-0}"
    PREFIX="excluded_"
    IDENT_BENCH="excluded_"
fi

cd "$WORKDIR"
mkdir -p logs data/results

if [ "$ORDER" = "reverse" ]; then
    FOLDS="$(echo $FOLDS | tr ' ' '\n' | tac | tr '\n' ' ')"
    VARIANTS="$(echo $VARIANTS | tr ' ' '\n' | tac | tr '\n' ' ')"
    PROJECTIONS="$(echo $PROJECTIONS | tr ' ' '\n' | tac | tr '\n' ' ')"
fi

for variant in $VARIANTS; do
    source_str="$(variant_source "$variant")"
    for proj in $PROJECTIONS; do
        for fold in $FOLDS; do
            tag="${PREFIX}${variant}_proj_${proj}_fold_${fold}_seed_0"
            dump="data/dumps/${tag}.tsv"
            name="linkgda_${PREFIX}${variant}_${proj}_fold_${fold}"
            ident="ultra4g_zeroshot_${IDENT_BENCH}fold_${fold}_seed_0_${source_str}_proj_${proj}_use_graph_True"
            if [ -s "data/results/kge_results_${ident}_by_graph_bma.tsv" ]; then
                echo "skip $ident (already scored)"
                continue
            fi
            # Two workers share one results directory and one ultra_kg cache, so a run
            # must be claimed before it starts or both could prepare the same dataset,
            # one deleting the processed cache the other is reading. A worker defers to
            # its peer's log, which announces an identifier before any work on it, and
            # then takes an atomic directory claim of its own.
            if [ -n "$PEER_LOG" ] && grep -qF "=== $ident ===" "$PEER_LOG" 2>/dev/null; then
                echo "skip $ident (peer worker has it)"
                continue
            fi
            if ! mkdir "data/results/.claim_${ident}" 2>/dev/null; then
                echo "skip $ident (claimed)"
                continue
            fi
            echo "=== $ident ==="
            python "$SCRATCH/prepare_ultra_data.py" --dump "$dump" --root data/ultra_kg --name "$name"
            python "$SCRATCH/score_ultra.py" \
                --ultra_root "$SCRATCH/ULTRA" \
                --data_root data/ultra_kg \
                --name "$name" \
                --dump "$dump" \
                --ckpt "$CKPT" \
                --identifier "$ident" \
                --out_dir data/results \
                --batch_size "$BATCH" \
                --log "logs/score_${ident}.log"
        done
    done
done

# Metrics run in the project env, not the ULTRA one. evaluate_sem_sim.compute_rank_roc
# calls np.trapezoid, which exists only from numpy 2.0, while torch 2.1 requires
# numpy < 2. Rather than fork the metric, the score files are read back by the same
# env every other method in this project is scored in.
if [ "$METRICS" = "1" ] && [ "$BENCH" = "main" ]; then
    conda activate multihopgda
    python "$SCRATCH/ultra_metrics.py" --results_dir data/results \
        --out data/results/ultra_zeroshot_metrics.tsv 2>&1 | tee logs/metrics.log
fi
echo "scoring stage complete for BENCH=$BENCH VARIANTS=$VARIANTS"
