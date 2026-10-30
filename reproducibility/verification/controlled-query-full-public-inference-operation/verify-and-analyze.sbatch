#!/bin/bash
#SBATCH --job-name=query-h200-analysis
#SBATCH --partition=batch
#SBATCH --account=pi-hohndor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029/orix-full-query-inference-analysis-v1/analysis-%j.log
set -euo pipefail
work=/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029/orix-full-query-inference-analysis-v1
export TMPDIR="$work/scratch/$SLURM_JOB_ID"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR"
/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029/analysis-environment-rebuild-v1/source/reproducibility/environments/cpu-analysis/.pixi/envs/default/bin/python -u "$work/verify-and-analyze.py"
