#!/usr/bin/env bash
# Create the ULTRA environment on the host it is run on.
#
# Two pip packages are installed after the env exists rather than from the yml:
# torch-geometric's build backend and the torch-scatter wheel index both need the
# already-installed torch to resolve, and a single pip transaction resolves them
# before torch is on disk.
# -u is deliberately absent: the conda gcc_linux-64 activation hook dereferences
# SYS_SYSROOT unguarded, which aborts the shell under nounset before the env is
# usable.
set -eo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME="ultra-linkgda"
YML="$(dirname "$0")/environment-ultra.yml"

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "env $ENV_NAME already exists; skipping create"
else
    conda env create -f "$YML"
fi

conda activate "$ENV_NAME"
python -m pip install "torch-geometric==2.4.0"
python -m pip install "torch-scatter==2.1.2" \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

python - <<'PY'
import torch, torch_geometric, torch_scatter, sys
print("python        ", sys.version.split()[0])
print("torch         ", torch.__version__)
print("torch.cuda    ", torch.version.cuda, "available:", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
print("torch_scatter ", torch_scatter.__version__)
PY
