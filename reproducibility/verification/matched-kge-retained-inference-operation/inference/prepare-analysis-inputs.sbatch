#!/bin/bash
#SBATCH --job-name=matched-retained-inputs
#SBATCH --partition=batch
#SBATCH --account=pi-hohndor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029/orix-full-matched-retained-replay-v3/input-stage-%j.log
set -euo pipefail
work=/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029/orix-full-matched-retained-replay-v3
export TMPDIR="$work/scratch/$SLURM_JOB_ID"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
mkdir -p "$TMPDIR"
/ibex/scratch/projects/c2014/hohndor/multihop-remediation-20261029/analysis-environment-rebuild-v1/source/reproducibility/environments/cpu-analysis/.pixi/envs/default/bin/python -u "$work/prepare-analysis-inputs.py"
