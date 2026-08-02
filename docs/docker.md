# Docker image (L1 GPU runtime)

Reproducible environment for the GPU L1 path (detector + LampRecognizer). One
image covers **both** backends, replacing the `setup_gpu_venv.sh` +
`run_gpu.sh` `LD_LIBRARY_PATH` setup:

- int8 `.engine` → system `libnvinfer` via the `trt_run` helper (compiled into
  the image at build time)
- fp32 `.onnx` → `onnxruntime-gpu`

Base image `nvcr.io/nvidia/tensorrt:25.01-py3` pins **TensorRT 10.8.0.43 +
CUDA 12.8 + cuDNN 9**, matching the validated host (`libnvinfer
10.8.0.43-1+cuda12.8`, CUDA 12.8).

## What is / isn't in the image

- **In**: the package + scripts, pinned Python deps, the compiled `trt_run`
  binary, and the checked-in default classifier
  (`models/traffic_light_lamp_recognizer_comlops.onnx`).
- **Out** (mount at runtime): the detector/classifier model store, and any
  `.engine` files. Engines are **GPU-architecture + TensorRT-version specific
  build artifacts** — treat them as regenerated-per-machine, not portable.
  Datasets are mounted too.

## Build

```bash
docker build -t tlr_autolabel .
```

The build compiles `trt_run` from `tools/trt_run.cpp`; that only needs the base
image's headers/libs (no GPU during build).

## Run

Requires the NVIDIA container runtime (this host already has it: `docker info`
shows the `nvidia` runtime + `nvidia.com/gpu` CDI devices).

```bash
# --help (default CMD)
docker run --rm --gpus all tlr_autolabel

# L1 over a mounted dataset with the Autoware model store.
# Mount the model root to /models (matches TLR_MODEL_ROOT / AUTOWARE_MLMODELS).
docker run --rm --gpus all \
  -v /path/to/mlmodels:/models:ro \
  -v /path/to/dataset:/data \
  tlr_autolabel scripts/tlr_autolabel.py /data/CAM_FRONT \
    --preset yolox-1920-int8 --out-dir /data/labels
```

> **Symlinked model stores (Autoware `mlmodels`).** `/opt/autoware/mlmodels/*`
> are symlinks into `/opt/autoware/model-store/...`, so mounting only
> `mlmodels` leaves the links dangling inside the container ("model not
> found"). Mount the whole tree at the same path and point the env at it:
>
> ```bash
> docker run --rm --gpus all \
>   -v /opt/autoware:/opt/autoware:ro \
>   -e AUTOWARE_MLMODELS=/opt/autoware/mlmodels \
>   -v /path/to/dataset:/data \
>   tlr_autolabel scripts/tlr_autolabel.py /data/CAM_FRONT \
>     --preset autoware-mlmodels-960-onnx --out-dir /data/labels
> ```

The entrypoint is `python3`, so any script/module works:

```bash
docker run --rm --gpus all -v /path/to/dataset:/data tlr_autolabel \
  scripts/match_traffic_lights.py --dataset-root /data

# run the test suite inside the image
docker run --rm tlr_autolabel -m unittest discover -s tests
```

### First-run engine build

The first int8 run against a `.onnx` builds the `.engine` (~1–2 s deserialize
amortized by the `trt_run` serve mode). Persist engines by mounting a writable
model dir so the build is reused across `docker run` invocations.

## Verified

On RTX 3060 Laptop (driver 580), host TensorRT 10.8 / CUDA 12.8:

- `docker build` clean; `trt_run` compiled into the image.
- `-m unittest discover -s tests` → 66/66 inside the image.
- L1 on a real frame, both backends: onnx preset →
  `backend=['CUDAExecutionProvider', ...]`; `.engine` detector →
  `backend=tensorrt-engine` (trt_run + system libnvinfer). Both wrote valid
  `tlr_autolabel/v1` JSON.

## Notes / caveats

- **`ENV PYTHONPATH` is intentionally NOT set.** The `scripts/` wrappers
  self-bootstrap the repo root; setting `PYTHONPATH=/workspace/tlr_autolabel`
  leaves `scripts/` ahead of it, so `scripts/tlr_autolabel.py` shadows the
  `tlr_autolabel` package. See the note in the `Dockerfile`.
- **GPU visibility at build**: not needed — `trt_run` only links against TRT/CUDA
  libs. The GPU is required only at `docker run` (`--gpus all`).
- **onnxruntime-gpu ↔ cuDNN**: `onnxruntime-gpu==1.23.0` loads the base image's
  CUDA 12 / cuDNN 9 libraries. If a future base bump breaks the load, fall back
  to the `nvidia-*-cu12` wheels listed in `setup_gpu_venv.sh`.
- **CPU-only tools** (L2/L3/L4 review/eval) run in this image too, but don't need
  `--gpus`; omit it to run them anywhere.
