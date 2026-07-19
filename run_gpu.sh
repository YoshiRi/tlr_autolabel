#!/usr/bin/env bash
# Run tlr_autolabel.py on the GPU via the dedicated onnxruntime-gpu venv.
# The venv holds onnxruntime-gpu 1.23 + matching CUDA 12 / cuDNN 9 libs, isolated
# from the system python so it can't disturb other tools.
#
# Usage: ./run_gpu.sh <image_or_dir> --out-dir ./labels [--viz] [any tlr_autolabel flag]
set -euo pipefail
# venv location is overridable; default keeps the original path.
VENV="${TLR_GPU_VENV:-$HOME/.venvs/tlr_onnxgpu}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "onnxruntime-gpu venv not found at $VENV" >&2
  echo "create it with: ./setup_gpu_venv.sh   (or set \$TLR_GPU_VENV)" >&2
  exit 1
fi
NV="$(echo "$VENV"/lib/python*/site-packages/nvidia)"
# make onnxruntime-gpu find libcufft/cublas/cudnn/... shipped by the nvidia wheels
export LD_LIBRARY_PATH="$(ls -d "$NV"/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$VENV/bin/python" "$HERE/tlr_autolabel.py" "$@"
