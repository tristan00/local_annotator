#!/usr/bin/env bash
set -euo pipefail
CUDA_HOME=/root/local-annotator-cuda13/nvidia/cu13
REBUILD="/root/.local/bin/uv pip install --python /root/vllm/bin/python --no-deps --target /root/local-annotator-cuda13 nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-cuda-runtime==13.0.96 nvidia-nvvm==13.0.88 nvidia-cuda-cccl==13.0.85"
if [ ! -x "$CUDA_HOME/bin/nvcc" ] || [ ! -f "$CUDA_HOME/include/cuda.h" ] || [ ! -f "$CUDA_HOME/include/cccl/cuda/std/utility" ]; then
  echo "vllm_serve: incomplete CUDA toolkit at $CUDA_HOME (needs nvcc, cuda.h and CCCL headers); create it with: $REBUILD" >&2
  exit 1
fi
headers=$(sed -n 's/^#define CUDA_VERSION \([0-9]*\).*/\1/p' "$CUDA_HOME/include/cuda.h")
read -r major minor < <("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\),.*/\1 \2/p')
if [ "$headers" != "$((major * 1000 + minor * 10))" ]; then
  echo "vllm_serve: nvcc $major.$minor does not match CUDA headers $headers in $CUDA_HOME; FlashInfer JIT would fail. Rebuild with: $REBUILD" >&2
  exit 1
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:/root/vllm/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=6
echo "vllm_serve: CUDA_HOME=$CUDA_HOME nvcc=$major.$minor headers=$headers MAX_JOBS=$MAX_JOBS args=$*" >&2
exec /root/vllm/bin/vllm serve "$@"
