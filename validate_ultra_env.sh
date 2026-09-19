#!/usr/bin/env bash
# Prove the environment by running ULTRA unmodified: zero-shot inference with the
# 4g checkpoint on CoDExSmall. This is the cheapest thing that exercises the whole
# stack, in particular the JIT build of the rspmm CUDA kernel, which is where a
# mismatched toolchain fails.
set -eo pipefail
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
SCRATCH="${SCRATCH:-$HOME/Git/link-gda-ultra}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ultra-linkgda
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_EXTENSIONS_DIR="$CONDA_PREFIX/torch_extensions"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDA_VISIBLE_DEVICES=0
echo "nvcc: $(which nvcc)"; nvcc --version | tail -2
echo "CXX:  $CXX"; "$CXX" --version | head -1
cd "$SCRATCH/ULTRA"
python script/run.py -c config/transductive/inference.yaml --dataset CoDExSmall \
    --epochs 0 --bpe null --gpus "[0]" --ckpt "$SCRATCH/ULTRA/ckpts/ultra_4g.pth"
