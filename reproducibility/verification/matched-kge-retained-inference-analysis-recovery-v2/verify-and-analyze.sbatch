#!/bin/bash
#SBATCH --job-name=matched-inference-analysis
#SBATCH --partition=batch
#SBATCH --account=pi-hohndor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/ibex/project/c2014/hohndor/multihop-remediation-20261029/orix-full-matched-retained-inference-analysis-v2/analysis-%j.log
set -euo pipefail
work=/ibex/project/c2014/hohndor/multihop-remediation-20261029/orix-full-matched-retained-inference-analysis-v2
export TMPDIR="$work/scratch/$SLURM_JOB_ID"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR"
/ibex/project/c2014/hohndor/multihop-remediation-20261029/analysis-environment-rebuild-v1/source/reproducibility/environments/cpu-analysis/.pixi/envs/default/bin/python -u "$work/verify-and-analyze.py"
