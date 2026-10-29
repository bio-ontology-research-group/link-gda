# Check reconstructed GPU environments

`reproducibility/verify_inference_gpu.py` exercises small synthetic graphs in
both reconstructed model environments. The recorded IBEX A100 run passes
TransD/ConvKB GPU scoring and gradient checks, and loads both released
ULTRA/UltraQuery initializations through the actual query backbone and CUDA
extension. CPU/GPU comparisons use `rtol=1e-4, atol=1e-5`, with TF32 disabled
for this check. Maximum observed absolute error was below `1.4e-6`.

See [native evidence](../reproducibility/verification/ibex-inference-gpu-readiness.json).
This checks runtime compatibility. Production checkpoint inference, benchmark
scores and training trajectories require their separate verification.

Run within a GPU allocation. `MATCHED_PYTHON` is the reconstructed matched-KGE
environment's Python; `MATCHED_TRAINING_SOURCE` and `QUERY_TRAINING_SOURCE` are
the frozen source checkouts described in the checkpoint-replay instructions.
`ULTRA` points to the prepared upstream checkout with both initial checkpoints.
Choose fresh output paths. The compiler settings below match the recorded
ULTRA environment and A100 GPU; other hardware requires its own architecture.

```bash
mkdir -p "$WORKSPACE/scratch" "$WORKSPACE/torch-extensions-a100"
export TMPDIR="$WORKSPACE/scratch"
"$MATCHED_PYTHON" reproducibility/verify_inference_gpu.py \
  --mode matched --source "$MATCHED_TRAINING_SOURCE" \
  --output "$WORKSPACE/matched-gpu.json"

export CUDA_HOME="$WORKSPACE/ultra-env"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"
export TORCH_EXTENSIONS_DIR="$WORKSPACE/torch-extensions-a100"
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4
"$CUDA_HOME/bin/python" reproducibility/verify_inference_gpu.py \
  --mode query --source "$QUERY_TRAINING_SOURCE" --ultra "$ULTRA" \
  --output "$WORKSPACE/query-gpu.json"
```

Allocate at least four CPU cores for the extension build. The verifier refuses
an existing output file, unavailable CUDA, non-finite GPU outputs/gradients or
CPU/GPU differences outside the stated tolerances. It records device, library
versions, initial-checkpoint hashes and its own source hash.
