#!/usr/bin/env bash
# Run tlr_autolabel.py on the GPU via the dedicated onnxruntime-gpu venv.
# The venv holds onnxruntime-gpu 1.23 + matching CUDA 12 / cuDNN 9 libs, isolated
# from the system python so it can't disturb other tools.
#
# Usage: ./run_gpu.sh <image_or_dir> --out-dir ./labels [--viz] [any tlr_autolabel flag]
set -euo pipefail
VENV=/home/yoshiri/.venvs/tlr_onnxgpu
NV="$VENV/lib/python3.10/site-packages/nvidia"
# make onnxruntime-gpu find libcufft/cublas/cudnn/... shipped by the nvidia wheels
export LD_LIBRARY_PATH="$(ls -d "$NV"/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$VENV/bin/python" "$HERE/tlr_autolabel.py" "$@"
