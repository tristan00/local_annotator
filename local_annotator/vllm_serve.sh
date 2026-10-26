#!/usr/bin/env bash
set -euo pipefail
CUDA_HOME=/root/local-annotator-cuda13/nvidia/cu13
REBUILD="bash $(dirname "$(readlink -f "$0")")/cuda_toolkit.sh"
for part in bin/nvcc include/cuda.h include/cccl/cuda/std/utility include/cublasLt.h lib64/libcudart.so lib64/libcublas.so lib64/libcublasLt.so \
            lib64/libnvrtc.so lib64/stubs/libcuda.so; do
  if [ ! -e "$CUDA_HOME/$part" ]; then
    echo "vllm_serve: $CUDA_HOME/$part is missing; prepare the toolkit with: $REBUILD" >&2
    exit 1
  fi
done
headers=$(sed -n 's/^#define CUDA_VERSION \([0-9]*\).*/\1/p' "$CUDA_HOME/include/cuda.h")
read -r major minor < <("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\),.*/\1 \2/p')
if [ "$headers" != "$((major * 1000 + minor * 10))" ]; then
  echo "vllm_serve: nvcc $major.$minor does not match CUDA headers $headers in $CUDA_HOME; FlashInfer JIT would fail. Rebuild with: $REBUILD" >&2
  exit 1
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:/root/vllm/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=6 VLLM_ENGINE_READY_TIMEOUT_S=3600
echo "vllm_serve: CUDA_HOME=$CUDA_HOME nvcc=$major.$minor headers=$headers MAX_JOBS=$MAX_JOBS args=$*" >&2
exec /root/vllm/bin/vllm serve "$@"
