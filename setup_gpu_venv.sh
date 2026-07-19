#!/usr/bin/env bash
# Create the onnxruntime-gpu venv that run_gpu.sh uses.
#
# The nvidia-*-cu12 wheels are required because a system CUDA install may be
# missing the exact shared libs onnxruntime-gpu links (e.g. libcufft.so.11);
# the wheels ship them and run_gpu.sh puts them on LD_LIBRARY_PATH.
#
# Override the location with $TLR_GPU_VENV (must match run_gpu.sh).
set -euo pipefail
VENV="${TLR_GPU_VENV:-$HOME/.venvs/tlr_onnxgpu}"

echo "creating venv at $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install \
  onnxruntime-gpu==1.23.0 \
  numpy==1.26.4 opencv-python-headless==4.10.0.84 pyyaml==6.0.2 scipy==1.13.1 \
  nvidia-cufft-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 \
  nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12

echo "done. run_gpu.sh will use $VENV"
