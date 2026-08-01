# TLR Autolabel Refactor Plan

Updated: 2026-07-31

## Goal

Clean up the repository without breaking the current command-line workflows.
Refactor only after the externally meaningful I/O contracts are covered by
tests, then move implementation into a package with stable CLI wrappers.

## Current Problems

- Top-level files mix CLI entrypoints, reusable library code, configs, generated
  outputs, local previews, model artifacts, and build artifacts.
- Large scripts such as `scripts/match_traffic_lights.py` and `scripts/tlr_autolabel.py` combine
  argument parsing, orchestration, domain logic, I/O, and backend-specific code.
- Pure logic (`state_tokens.py`, `temporal_association.py`) lives beside
  OpenCV/ONNX/TensorRT/ROS-dependent code.
- Generated data is mostly ignored by `.gitignore`, but local outputs under
  `data/`, `sample_preview/`, `__pycache__/`, and `trt_run` still make the
  working tree visually noisy.
- Some interfaces are still evolving. In particular, the current TIER B/B'
  contract should use `traffic_light.json` rows shaped as:

```json
{
  "token": "...",
  "instance_token": "...",
  "primitive_id": "..."
}
```

## Target Repository Shape

```text
tlr_autolabel/
  __init__.py
  cli/
    autolabel.py
    match.py
    to_object_ann.py
    evaluate.py
    review.py
  core/
    state_tokens.py
    schemas.py
    io.py
  inference/
    detector.py
    lamp_recognizer.py
    trt.py
  map/
    lanelet2.py
    projection.py
    association.py
  tracking/
    temporal.py
  t4/
    object_ann.py
    traffic_light.py
    adapters.py
  review/
    cvat.py
    re_timeline.py
  eval/
    signals.py
    l1_vs_t4.py
scripts/
  tlr_autolabel.py
  match_traffic_lights.py
  to_object_ann.py
  export_object_ann_to_t4dataset.py
  export_re_to_t4dataset.py
  ...
configs/
docs/
tests/
ros2_pipeline/
deprecated/
```

Compatibility rule: keep existing top-level commands working during the
transition. A current command such as:

```bash
python3 scripts/match_traffic_lights.py --dataset-root <dataset>
```

should keep working by delegating to the new package entrypoint.

## Refactor Sequence

### 1. Baseline and Hygiene

- Record the current dirty working tree and separate unrelated user changes
  from refactor changes.
- Run the current test suite and record pass/fail as the baseline.
- Confirm `.gitignore` covers generated outputs:
  - `__pycache__/`
  - `trt_run`
  - `sample_preview/`
  - `data/*` except small documentation/fixture files
- Decide whether local artifacts currently in the working tree should stay
  local-only, become test fixtures, or move to external storage.

### 2. Add I/O Contract Tests Before Moving Code

Add tests around interfaces that are meant to be stable:

- Tier A `tlr_autolabel/v1` per-frame JSON:
  - required metadata keys
  - `signals[]` shape
  - optional `raw_detections[]` shape
  - deterministic IDs for synthetic input
- A' `traffic_signal_2d/v2` sidecar:
  - annotation row shape
  - `attributes.state`, visibility, review fields, map cache fields
  - v1 fallback where still supported
- TIER B/B' t4dataset output:
  - `object_ann.json`, `category.json`, `attribute.json`, `instance.json`
  - `traffic_light.json` must use `{token, instance_token, primitive_id}`
  - no deprecated `traffic_light_map_association.json` unless explicitly asked
  - no `instance_token` carrying lanelet2/TLR identity
- CVAT round trip:
  - A' sidecar -> CVAT XML/ZIP metadata -> A' sidecar
  - state token validation fails loudly
  - map ID validation against synthetic lanelet2 map
- RE review:
  - template generation
  - accepted/fixed/rejected/unchecked behavior
  - propagation back into reviewed A'
- CLI smoke tests:
  - synthetic dataset only
  - no real model or GPU required
  - write only to temporary directories

### 3. Move Pure Core Modules First

- Move `state_tokens.py` to `tlr_autolabel/core/state_tokens.py`.
- Move `temporal_association.py` to `tlr_autolabel/tracking/temporal.py`.
- Leave old top-level modules as import compatibility shims initially.
- Update tests to import the package path.
- Keep a short backward-compatibility test for the old import path until all
  scripts are migrated.

### 4. Split T4 Annotation I/O

- Extract shared JSON/table helpers to `tlr_autolabel/core/io.py`.
- Move B/B' writing logic into:
  - `tlr_autolabel/t4/object_ann.py`
  - `tlr_autolabel/t4/traffic_light.py`
  - `tlr_autolabel/t4/adapters.py`
- Keep CLI wrappers for:
  - `scripts/to_object_ann.py`
  - `scripts/export_object_ann_to_t4dataset.py`
  - `scripts/export_re_to_t4dataset.py`
- Add golden tests for the generated annotation rows.

### 5. Split L3 Map Enrichment

Refactor `scripts/match_traffic_lights.py` in smaller pieces:

- lanelet2 parsing -> `tlr_autolabel/map/lanelet2.py`
- frame/camera indexing -> `tlr_autolabel/t4/index.py`
- projection -> `tlr_autolabel/map/projection.py`
- detection-to-map association -> `tlr_autolabel/map/association.py`
- temporal tracking integration -> uses `tlr_autolabel/tracking/temporal.py`
- report writing -> `tlr_autolabel/core/io.py` or dedicated report module

This is the highest-risk move. Do it only after synthetic integration tests
cover normal match, low-score tracking, propagation, duplicate prevention, and
map-missing behavior.

### 6. Split L1 Inference

Refactor `scripts/tlr_autolabel.py` into:

- detector backends:
  - ONNX YOLOX
  - CoMLOps detector
  - TensorRT engine wrapper
- lamp recognizer backends:
  - ONNX
  - TensorRT engine wrapper
- image orchestration:
  - tiling
  - NMS/containment handling
  - crop classification
  - Tier A JSON writer
- CLI wrapper:
  - preset resolution
  - argument parsing
  - run metadata

Unit tests should use mock detector/classifier backends. Real model smoke tests
should be optional and skipped unless model paths are available.

### 7. Split Review, Evaluation, and Visualization

- CVAT import/export -> `tlr_autolabel/review/cvat.py`
- RE timeline template/render/apply -> `tlr_autolabel/review/re_timeline.py`
- signal evaluation -> `tlr_autolabel/eval/signals.py`
- map-less T4 evaluation -> `tlr_autolabel/eval/l1_vs_t4.py`
- rendering/video helpers can remain CLI-oriented, but reusable parsing should
  live under the package.

### 8. Clean Artifact Boundaries

- Keep committed fixtures small and intentional under `tests/fixtures/`.
- Keep generated dataset snapshots out of git under `data/`.
- Keep local visual inspection outputs out of git under `sample_preview/`.
- Treat compiled binaries such as `trt_run` as build artifacts unless there is a
  strong reason to commit them.
- Treat large model files as external dependencies unless they are deliberately
  small enough and license-compatible to track.

## Suggested Commit / PR Order

1. Test and hygiene baseline:
   - `.gitignore` cleanup
   - test command documentation
   - current baseline note
2. I/O contract tests:
   - Tier A
   - A'
   - B/B' including `primitive_id`
   - CVAT and RE review minimal round trips
3. Package skeleton and pure module moves:
   - `core/state_tokens.py`
   - `tracking/temporal.py`
4. T4 annotation package extraction.
5. L3 map enrichment extraction.
6. L1 inference extraction.
7. Review/eval/render extraction.
8. Remove temporary compatibility shims after all callers and docs use package
   paths.

## Portability to Another PC

Refactor-only work is possible on another PC if it has:

- this git repository
- Python 3.10+
- dependencies from `requirements.txt`
- `numpy`, `scipy`, `yaml`/PyYAML, and other normal test dependencies
- no real dataset or model files for synthetic unit tests

Full L1 inference smoke tests need model files. Current presets reference:

- `${TLR_MODEL_ROOT}/traffic_light_detector_testL/yolox-sPlus-opt-Co_MLOps-traffic_light-1920x1280_20260706_best.engine`
- `${TLR_MODEL_ROOT}/TLRtest/traffic_light_detector/yolox-sPlus-opt-Co_MLOps-traffic_light-1280x1280_20260703_best.engine`
- `${TLR_MODEL_ROOT}/traffic_light_detector_testS/yolox-sPlus-opt-Co_MLOps-traffic_light-960x960_20260629_best.engine`
- `${TLR_MODEL_ROOT}/traffic_light_classifier/CoMLOps-Large-Detection-Model-v1.0.1.onnx`
- `${AUTOWARE_MLMODELS}/traffic_light_detector/yolox_s_car_ped_tl_detector_960_960_batch_1.onnx`
- `${AUTOWARE_MLMODELS}/traffic_light_classifier/traffic_light_lamp_recognizer_comlops.onnx`
- `${AUTOWARE_MLMODELS}/traffic_light_classifier/lamp_recognizer_ml.param.yaml`

Important portability note: TensorRT `.engine` files are not portable across
GPU/TensorRT/CUDA environments. On another PC, prefer ONNX smoke tests or rebuild
the engine locally from the matching ONNX.

Full L3/L4/L6 end-to-end smoke tests need a T4-style dataset with:

- `annotation/sample_data.json`
- `annotation/sample.json` where required
- `annotation/ego_pose.json`
- `annotation/calibrated_sensor.json`
- `annotation/sensor.json`
- `map/lanelet2_map.osm`
- camera images under `data/<CHANNEL>/`
- optional existing `tlr_autolabel/<CHANNEL>/*.json`
- optional A' and RE review sidecars for review/evaluation tests

ROS2 parity work needs more:

- ROS 2 Humble
- Autoware workspace sourced
- `autoware_ml_model_launchers`
- `tier4_perception_msgs`
- `sensor_msgs`
- `cv_bridge`
- relevant rosbag input
- production model store under `/opt/autoware/mlmodels` or equivalent

## What Must Be Shared for Other-PC Work

Minimum for safe refactor and most tests:

- repository source
- small synthetic fixtures added under `tests/fixtures/`
- `requirements.txt`

Needed for realistic non-GPU integration:

- a small redacted T4 dataset fixture, preferably with only a few frames
- matching lanelet2 map snippet
- sample `tlr_autolabel/v1` predictions
- sample A' sidecar and RE review JSON

Needed for real inference:

- detector ONNX files
- classifier ONNX files
- `configs/model_params/lamp_recognizer_ml.param.yaml`
- `configs/model_params/comlops_large_detector_ml.param.yaml` if using CoMLOps detector
- environment variables:
  - `TLR_MODEL_ROOT`
  - `AUTOWARE_MLMODELS` when not using `/opt/autoware/mlmodels`

Usually not worth sharing:

- TensorRT `.engine` files, unless the other PC has a matching GPU/CUDA/TRT
  stack
- large generated dataset snapshots under `data/`
- local visual previews under `sample_preview/`
- `__pycache__/`
- compiled `trt_run`

## Handoff Policy

When another PC or another engineer takes over this refactor, hand off the work
as a source-first task. The recipient should be able to run synthetic tests and
perform package restructuring without access to local production datasets or GPU
model artifacts.

### Handoff Package

Share:

- the git repository, including this `REFACTOR_PLAN.md`
- current branch name and commit SHA
- a clean summary of uncommitted changes, grouped by topic
- exact baseline test command and result
- small synthetic fixtures under `tests/fixtures/` once they are added
- any newly added package/test files
- `requirements.txt`
- `configs/`
- `docs/`

Do not share by default:

- full local `data/` snapshots
- `sample_preview/`
- `__pycache__/`
- compiled `trt_run`
- local TensorRT `.engine` files
- machine-specific absolute paths under `/home/yoshiri/...`
- rosbag files unless the task explicitly requires ROS2 parity

Share only when the recipient is expected to run realistic end-to-end tests:

- a minimal redacted T4 dataset fixture with a few frames
- matching `map/lanelet2_map.osm`
- sample `tlr_autolabel/v1` predictions
- sample A' sidecar
- sample RE review sidecar
- expected output golden JSON files

Share only when the recipient is expected to run real inference:

- detector ONNX files
- classifier ONNX files
- classifier param YAML
- detector param YAML for CoMLOps-style models
- instructions for setting `TLR_MODEL_ROOT` and `AUTOWARE_MLMODELS`

Avoid sharing TensorRT `.engine` files unless the target PC intentionally mirrors
the source machine's GPU, CUDA, TensorRT, and driver stack. Prefer ONNX tests or
local engine rebuilds on the target PC.

### Required Handoff Notes

Every handoff should include:

- objective: which refactor phase is being attempted
- allowed scope: files/modules the recipient may move or edit
- compatibility expectations: which old commands must still work
- known dirty files that are unrelated and must not be reverted
- baseline command output summary
- missing optional dependencies, if any
- whether model-backed or ROS2-backed tests are expected
- current TIER B/B' contract: `traffic_light.json` uses
  `{token, instance_token, primitive_id}`

Recommended handoff text template:

```text
Objective:
  Continue refactor phase <N>: <short description>.

Branch / base:
  <branch> at <commit SHA>.

Must preserve:
  python3 scripts/tlr_autolabel.py ...
  python3 scripts/match_traffic_lights.py --dataset-root <dataset>
  python3 scripts/to_object_ann.py ...

Baseline:
  <test command>
  <pass/fail summary>

Do not touch:
  <unrelated dirty files or local artifacts>

Available local assets:
  <models/datasets/fixtures present on this PC>

Not available:
  <GPU/TRT/ROS2/production dataset/etc.>

Current IF contract:
  annotation/traffic_light.json rows are {token, instance_token, primitive_id}.
```

### File Inventory for Handoff

Required source files:

- `README.md`
- `STATUS.md`
- `PLAN.md`
- `REFACTOR_PLAN.md`
- all tracked `*.py`
- `configs/`
- `docs/`
- `tests/`
- `requirements.txt`
- `make_cvat_review.sh`
- `run_gpu.sh`
- `setup_gpu_venv.sh`
- `ros2_pipeline/` if ROS2 parity or bag conversion remains in scope

Optional small files:

- `configs/model_params/lamp_labels.txt`
- `configs/model_params/lamp_recognizer_ml.param.yaml`
- `configs/model_params/comlops_large_detector_ml.param.yaml`
- `models/traffic_light_lamp_recognizer_comlops.onnx` if license and storage policy
  allow sharing it

Large or local-only files:

- `data/`: currently local generated/reference snapshots; do not assume portable
- `sample_preview/`: visual inspection output
- `trt_run`: compiled local binary
- `.engine` files under model roots: machine-specific
- production T4 datasets under external paths such as
  `/home/yoshiri/.webauto/...`: share only as a minimized fixture or via the
  project's approved data channel

### First Commands on the Target PC

The recipient should start with:

```bash
git status --short
python3 -m unittest discover -s tests
```

If dependency installation is needed:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests
```

For real inference work, also confirm:

```bash
echo "$TLR_MODEL_ROOT"
echo "$AUTOWARE_MLMODELS"
python3 scripts/tlr_autolabel.py --help
```

Do not start by running full dataset jobs or ROS2 parity jobs. First prove the
synthetic tests and CLI help still work after the move.

## Immediate Next Step

Phase 1.8 is now in progress on `refactor/phase1-8-review`:

- top-level CLI entrypoints moved under `scripts/`
- reusable review/eval logic moved under `tlr_autolabel.review` and
  `tlr_autolabel.eval`
- L1 orchestration moved behind `tlr_autolabel.cli.autolabel`; the
  `scripts/tlr_autolabel.py` command is a thin wrapper
- T4 A->B/B' converters moved behind `tlr_autolabel.t4.convert`,
  `tlr_autolabel.t4.object_ann_export`, and `tlr_autolabel.t4.re_export`; the
  corresponding `scripts/` files are thin wrappers
- L3 map-enrichment orchestration moved behind `tlr_autolabel.cli.match`; the
  `scripts/match_traffic_lights.py` command is a thin wrapper
- pure compatibility shims removed after callers were migrated to package paths
- model params moved under `configs/model_params/`
- tracked model artifact moved under `models/`
- TensorRT helper source moved under `tools/`, binary output under ignored
  `build/`

Remaining refactor work after this phase:

1. Continue splitting `tlr_autolabel.cli.match` until remaining report writing,
   fill/backfill orchestration, and sidecar emission are independently reusable.
2. Decide whether `data/` and `sample_preview/` should stay as local-only
   ignored folders or become explicit fixture/artifact locations.
3. Add direct package-level tests for the new `tlr_autolabel.t4.convert` and
   `tlr_autolabel.cli.autolabel` entrypoints where the current coverage still
   exercises only wrapper-level behavior.

The original first step for this plan was:

1. Add the package skeleton.
2. Add I/O contract tests for `traffic_light.json` using `primitive_id`.
3. Move only `state_tokens.py` and `temporal_association.py` behind compatibility
   shims.
4. Run the current unit tests.

Only after that should larger CLI modules be split.
