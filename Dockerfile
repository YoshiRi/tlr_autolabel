# tlr_autolabel runtime image.
#
# Base: NGC TensorRT 25.01 == TensorRT 10.8.0.43 + CUDA 12.8 + cuDNN 9, which
# matches the validated host (libnvinfer 10.8.0.43-1+cuda12.8). This single
# image serves BOTH L1 GPU paths and replaces the setup_gpu_venv.sh +
# run_gpu.sh LD_LIBRARY_PATH juggling:
#   - int8 .engine  -> system libnvinfer via the trt_run helper (compiled here)
#   - fp32 .onnx    -> onnxruntime-gpu
#
# Models and .engine files are NOT baked in (large/licensed, and engines are
# GPU-arch + TensorRT-version specific build artifacts). Mount them at runtime.
# See docs/docker.md for build/run recipes.
FROM nvcr.io/nvidia/tensorrt:25.01-py3

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# opencv-python-headless still needs libglib at runtime; everything else
# (CUDA, cuDNN, TensorRT, g++, python3+pip) comes from the base image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/tlr_autolabel

# Pinned to the versions this pipeline was validated against (see requirements.txt).
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
      numpy==1.26.4 \
      opencv-python-headless==4.10.0.84 \
      pyyaml==6.0.2 \
      scipy==1.13.1 \
      onnxruntime-gpu==1.23.0

COPY . .

# Precompile the TensorRT runner at build time (the host compiles it lazily on
# first use). Linking needs only the headers/libs from the base image, not a
# GPU, so this works during `docker build`.
RUN mkdir -p build && \
    g++ -O2 tools/trt_run.cpp -o build/trt_run \
        -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
        -lnvinfer -lcudart && \
    test -x build/trt_run

# Package importable from any cwd (also makes `python -m tlr_autolabel.*` and
# the scripts/ wrappers resolve without relying on cwd).
ENV PYTHONPATH=/workspace/tlr_autolabel
# Default mount points for the model store; override to taste.
ENV TLR_MODEL_ROOT=/models \
    AUTOWARE_MLMODELS=/models

ENTRYPOINT ["python3"]
CMD ["scripts/tlr_autolabel.py", "--help"]
