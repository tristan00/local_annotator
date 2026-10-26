#!/usr/bin/env bash
set -euo pipefail
TARGET=/root/local-annotator-cuda13
CUDA_HOME=$TARGET/nvidia/cu13
/root/.local/bin/uv pip install --python /root/vllm/bin/python --no-deps --target "$TARGET" \
  nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-cuda-runtime==13.0.96 nvidia-nvvm==13.0.88 nvidia-cuda-cccl==13.0.85 \
  nvidia-cuda-nvrtc==13.0.88 nvidia-cublas==13.1.1.3
mkdir -p "$CUDA_HOME/lib64/stubs"
for library in libcudart libcublas libcublasLt libnvrtc; do
  ln -sfn "../lib/$library.so.13" "$CUDA_HOME/lib64/$library.so"
done
ln -sfn /usr/lib/wsl/lib/libcuda.so "$CUDA_HOME/lib64/stubs/libcuda.so"
echo "cuda_toolkit: ready at $CUDA_HOME"
