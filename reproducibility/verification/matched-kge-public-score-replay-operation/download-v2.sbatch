#!/bin/bash
#SBATCH --job-name=matched-public-scores-v2
#SBATCH --partition=batch
#SBATCH --account=pi-hohndor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029/full-matched-score-public-replay-v1/download-v2-%j.log
set -euo pipefail
work=/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029/full-matched-score-public-replay-v1
export TMPDIR="$work/scratch/$SLURM_JOB_ID"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR"
/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029/analysis-environment-rebuild-v1/source/reproducibility/environments/cpu-analysis/.pixi/envs/default/bin/python -u "$work/download-v2.py"
