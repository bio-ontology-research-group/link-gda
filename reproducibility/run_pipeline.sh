#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  read -r -a PY_CMD <<< "$PYTHON_BIN"
elif command -v uv >/dev/null 2>&1; then
  PY_CMD=(uv run python3)
else
  PY_CMD=(python3)
fi
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/results}"

usage() {
  cat <<'EOF'
Usage: ./reproducibility/run_pipeline.sh <command> [options]

End-to-end command-line runner for reproducing all experiments, model retrainings,
matched baseline refittings, and stepwise bias evaluations (S0 -> S4).

Commands:
  prepare                  Download/prepare association files, 10 disease-disjoint folds, and ontology projections
  train-kge                Train baseline LinkGDA (U1, T1, P-pool), ConvKB-D, and INDIGENA on full graphs (folds 0..9)
  train-nobridge-kge       Train LinkGDA (U1, T1, P-pool), ConvKB-D, and INDIGENA on disease-bridge-free graphs (folds 0..9)
  train-ultra              Staged ULTRA / UltraQuery campaign; requires --root, --base, --ultra-root
  train-phenoquery         PhenoQuery campaign; requires --root, --base, --ultra-root
  run-phrank-phenorank     Refit and evaluate Phrank and 1,000-simulation PhenoRank on UPheno v2 + MGI (folds 0..9)
  evaluate-stepwise-bias   Compute S0 -> S4 stepwise bias decomposition and corrections across all 66 variants
  generate-tables          Regenerate LaTeX tables (Table 9 and Suppl. Table S13) from saved JSON reports
  --help, -h               Show this help message and exit

Environment variables:
  PYTHON_BIN   Python executable (default: python3)
  DATA_ROOT    Prepared benchmark data root (default: ./data)
  WORKSPACE_ROOT Benchmark workspace containing data/foldN and baseline resources
  OUT_ROOT     Output directory for checkpoints and score matrices (default: ./data/results)
EOF
}

if [[ $# -eq 0 ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

CMD="$1"
shift

case "$CMD" in
  prepare)
    cd "$REPO_ROOT"
    "${PY_CMD[@]}" code/data/download_data.py
    "${PY_CMD[@]}" code/data/build_association_files.py
    "${PY_CMD[@]}" code/data/generate_folds.py
    "${PY_CMD[@]}" code/projector/project_independent.py --data-dir "$DATA_ROOT"
    ;;
  train-kge)
    cd "$REPO_ROOT"
    for fold in $(seq 0 9); do
      "${PY_CMD[@]}" code/training/kge_transd.py --fold "$fold" \
        --use_phenotypes --use_functions --use_site --use_graph \
        --projector_name owl2vecstar --graph_policy independent-unique \
        --candidate_pool phenotype --calibration_unit disease \
        --random_seed 0 --val_seed 0 --embedding_dim 200 \
        --batch_size 65536 --learning_rate 0.001 --tolerance 15 \
        --dual_arms --write_baselines --no_sweep "$@"
    done
    ;;
  train-nobridge-kge)
    cd "$REPO_ROOT"
    for fold in $(seq 0 9); do
      for cond in U1_INDIGENA_PPOOL T1 CONVKB; do
        "${PY_CMD[@]}" code/training/retrain_nobridge_kge.py \
          --fold "$fold" \
          --condition "$cond" \
          --data-root "$DATA_ROOT" \
          --code-root "$REPO_ROOT" \
          --out-root "${OUT_ROOT}/nobridge_runs" "$@"
      done
    done
    ;;
  train-ultra)
    cd "$REPO_ROOT"
    "${PY_CMD[@]}" code/baselines/run_query_campaign.py --kind matched "$@"
    ;;
  train-phenoquery)
    cd "$REPO_ROOT"
    "${PY_CMD[@]}" code/baselines/run_query_campaign.py --kind eq "$@"
    ;;
  run-phrank-phenorank)
    cd "$REPO_ROOT"
    "${PY_CMD[@]}" code/baselines/run_matched_phrank_phenorank.py \
      --base-dir "${WORKSPACE_ROOT:-${REPO_ROOT}/workspace}" \
      --out-root "${OUT_ROOT}/corrected_v3" "$@"
    ;;
  evaluate-stepwise-bias)
    cd "$REPO_ROOT"
    "${PY_CMD[@]}" code/analysis/evaluate_stepwise_bias.py \
      --base-dir "${WORKSPACE_ROOT:-${REPO_ROOT}/workspace}" \
      --out-json "${REPO_ROOT}/reproducibility/results/stepwise-bias-report-v4.json" "$@"
    ;;
  generate-tables)
    cd "$REPO_ROOT"
    "${PY_CMD[@]}" reproducibility/generate_bias_tables.py \
      --report-json "${REPO_ROOT}/reproducibility/results/stepwise-bias-report-v4.json" \
      --out-dir "${REPO_ROOT}/reproducibility/tables" "$@"
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    usage >&2
    exit 1
    ;;
esac
