#!/usr/bin/env bash
# Export per-benchmark, per-variant, per-fold LinkGDA training graphs for the ULTRA baseline.
#
# The graph is built by kge_transd.py itself, with the same flags the corresponding
# LinkGDA runs use, so the baseline sees the identical graph rather than a
# reconstruction of it. That matters most for the phenotype-free variants: when
# --use_phenotypes is off, a gene with no annotation from the included modalities
# still keeps its first gene->phenotype edge (kge_transd.py, the gene_phenotypes
# block), and that fallback is inherited here rather than reimplemented.
# --dump_triples returns before any model is built, so this is CPU only.
#
# data/ in each working directory is a directory of symlinks into the read-only
# checkouts, not a link to them: kge_transd.py creates data/results and data/models
# on startup and must not touch the published ones.
#
# BENCH=main      ten folds of the main benchmark, candidate pool 4,399
# BENCH=excluded  the single excluded-gene fold, candidate pool 4,749
set -eo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
SCRATCH="${SCRATCH:-$HOME/Git/link-gda-ultra}"
SOURCE_DATA="${SOURCE_DATA:-$HOME/Git/link-gda/data}"
EXCLUDED_DATA="${EXCLUDED_DATA:-$HOME/Git/link-gda-excluded/data}"
BENCH="${BENCH:-main}"
VARIANTS="${VARIANTS:-pfs}"
PROJECTIONS="${PROJECTIONS:-owl2vecstar owl2vecstar_gda}"

variant_flags() {
    case "$1" in
        pfs) echo "-pheno -func -site" ;;
        f)   echo "-func" ;;
        s)   echo "-site" ;;
        fs)  echo "-func -site" ;;
        *)   echo "unknown variant $1" >&2; exit 1 ;;
    esac
}

# Must match kge_transd.py's source_str, which is what the LinkGDA result filenames carry.
variant_source() {
    case "$1" in
        pfs) echo "pheno_func_expr" ;;
        f)   echo "func" ;;
        s)   echo "expr" ;;
        fs)  echo "func_expr" ;;
    esac
}

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate multihopgda

if [ "$BENCH" = "main" ]; then
    WORKDIR="$SCRATCH"
    FOLDS="${FOLDS:-0 1 2 3 4 5 6 7 8 9}"
else
    WORKDIR="$SCRATCH/excluded"
    FOLDS="${FOLDS:-0}"
fi

mkdir -p "$WORKDIR"/data/{results,models,dumps,ultra_kg} "$WORKDIR/logs"
cd "$WORKDIR"

for f in gene_diseases.csv disease_phenotypes.csv gene_functions.csv gene_phenotypes.csv \
         gene_site.csv upheno_edges.tsv upheno_edges_gda.tsv go_edges.tsv uberon_edges.tsv folds; do
    ln -sfn "$SOURCE_DATA/$f" "data/$f"
done
if [ "$BENCH" = "excluded" ]; then
    # The benchmark supplies its own candidate pool and its own single fold; every other
    # input is the shared one, exactly as link-gda-excluded/data symlinks them.
    ln -sfn "$EXCLUDED_DATA/gene_diseases.csv" data/gene_diseases.csv
    ln -sfn "$EXCLUDED_DATA/folds" data/folds
fi

for variant in $VARIANTS; do
    flags="$(variant_flags "$variant")"
    for proj in $PROJECTIONS; do
        for fold in $FOLDS; do
            if [ "$BENCH" = "main" ]; then
                tag="${variant}_proj_${proj}_fold_${fold}_seed_0"
            else
                tag="excluded_${variant}_proj_${proj}_fold_${fold}_seed_0"
            fi
            dump="data/dumps/${tag}.tsv"
            if [ -s "$dump.test_pairs.tsv" ]; then
                echo "skip $tag (already exported)"
                continue
            fi
            echo "=== exporting $tag ==="
            python "$SCRATCH/kge_transd.py" \
                --fold "$fold" $flags --use_graph \
                --projector_name "$proj" --random_seed 0 --no_sweep \
                --dump_triples "$dump" 2>&1 | tee "logs/export_${tag}.log"
        done
    done
done
echo "export stage complete for BENCH=$BENCH VARIANTS=$VARIANTS"
