# TLR autolabeling

Tooling to **evaluate traffic-light-recognition models and efficiently produce
the evaluation data** for it. Autolabel over camera images with the ML models
Autoware can run, refine into review-ready GT via manual correction, and
evaluate. Training is out of scope. Output conforms to the **standard t4dataset
`object_ann.json`**, so downstream conversions (AWML / Deepen / CVAT / COCO) are
delegated to existing t4devkit / webauto tooling rather than maintained here.

> **Where things stand** — capabilities, goals, and maturity live in
> [STATUS.md](STATUS.md); remaining tasks in [PLAN.md](PLAN.md). This README is
> the contract (how it works, IF specs, schemas).

## Architecture: processing layers

The work is organized in layers. Lower layers never depend on higher ones.
**The solid interface is the standard t4dataset `object_ann.json` (Tier B)**:
once labels are in that format the existing t4devkit / webauto tooling produces
AWML info files, Deepen, CVAT, COCO etc. — we do not maintain those converters.
Our owned surface shrinks to L1, L3, the A→B converter, and the review UI.

The prime (`'`) mark means only one thing: **whether the map / regulatory-element
identity is carried**. The box+state+occlusion/truncation(+2D instance) core IF
is common across A↔B; the prime adds the map-signal identity (A'/B'). The RE→lane
relation is never persisted — it is re-derived from the lanelet2 map at eval time.

| layer | role | input -> output | where |
|-------|------|-----------------|-------|
| **L1 inference** | detect + classify signals on an image directory | images -> Tier A (`tlr_autolabel/v1` per-frame JSON) | `tlr_autolabel.py` |
| **L2 standardize (A→B)** | convert autolabel to the **standard t4 `object_ann.json`** (the solid IF): bbox + db_tlr `category` + `occlusion`/`truncation` attributes + optional empty 2D `instance`; the optional map-signal id goes to a separate sidecar | Tier A (or A') -> Tier B (`object_ann.json` + `category`/`attribute`) [+ `traffic_light_map_association.json` for B'] | `to_object_ann.py` |
| **L3 map enrichment (A→A')** | map association (lanelet2 way + RE) + multi-camera/head fusion; an **internal** enrichment used for review/QA, eval-time re-derivation, and to fill B' | Tier A + T4 map -> Tier A' (`traffic_signal_2d/v2` sidecar) + `traffic_signal_re/v1` | `match_traffic_lights.py`, `aggregate_regulatory_signals.py`, `render_re_timeline.py` |
| **L4 review UI** | human correction that turns provisional A' into a confirmed Tier B: per-frame box / state / occlusion / truncation (CVAT), and RE state-intervals (timeline) | Tier A'/B -> reviewed Tier B | CVAT pair (`export_cvat_signal_task.py`/`import_cvat_signal_annotations.py`, `docs/cvat_interop.md`) + RE review (`make_re_review_template.py`, `render_re_review_timeline.py`, `apply_re_review.py`, `docs/re_timeline_review.md`) |
| _(downstream)_ | AWML info / Deepen / CVAT / COCO **from** standard `object_ann` | Tier B -> external formats | **existing t4devkit / webauto tooling — not maintained here** |
| **L5 ros2 verification** | score the live Autoware node against our GT (detection + classification) | node rosbag -> `tlr_autolabel/v1` -> eval | `ros2_pipeline/`, `bag_to_labels.py`, `docs/eval_design.md` |
| **L6 evaluation** | metrics vs GT: detection P/R/IoU + classification accuracy + confusion (two-source), plus the ledger profiles; RE level via driving_log_replayer_v2 | pred + GT -> `tlr_eval*` reports | `eval_vs_gt.py`, `evaluate_signals.py` |

Design rules:

- **L1 is inference; everything after is conversion / review / analysis.** No
  layer after L1 re-runs inference (regenerate from stored labels).
- **L1 is model- and dataset-agnostic.** Detector variants interchangeable;
  T4 linkage is optional via `--t4-dataset`, never required.
- **Tier B is strictly standard.** `object_ann` carries only the standard keys;
  the map-signal id lives in a separate optional sidecar, absent when there is
  no map (contract: optional). The RE/lane relation is re-derived, never stored.
- **L3 adds fields, does not transform**, and is now an internal aid (feeds B',
  review, and eval-time map re-derivation) rather than a delivered tier.
- **L5 is out of the dependency graph.** Nothing in L1-L4 imports `ros2_pipeline/`.
- **L6 is pure analysis** — never writes annotations, never re-runs inference.

## Annotation tiers

Core IF (common to A and B): `{sample_data_token, box, state, occlusion,
truncation, 2D instance}`. Prime adds `{map_traffic_light_id}` (optional).

| tier | spec | carries map/RE? | notes |
|------|------|-----------------|-------|
| **A** | `tlr_autolabel/v1` (frozen 2026-07-18) | no | raw autolabel per frame; **richest** — full per-lamp color/shape/arrow + confidence; source of truth |
| **A'** | `traffic_signal_2d/v2` sidecar + `traffic_signal_re/v1` | yes | L3 internal enrichment: canonical `state`, `map_traffic_light_id`, review fields, RE fusion. Used for review/QA + eval-time re-derivation; **not a delivered format** |
| **B** | standard t4 `object_ann.json` (+ `category`/`attribute`) | no | **the solid interface.** bbox + db_tlr `category` + `occlusion_state`/`truncation_state` + optional 2D `instance`. Consumed by AWML/Deepen/CVAT/COCO via existing tooling |
| **B'** | B + `traffic_light_map_association.json` | yes | B plus the optional map-signal id (separate from the 2D instance); absent with no map. RE relation re-derived from the map, not stored |
| **B-review** | `traffic_signal_re_review/v1` | — | human RE state-interval decisions; propagates into the reviewed annotation |

> **Lossy projection is deliberate.** Tier A keeps full lamp detail; the A→B
> `category` is the db_tlr projection (per `configs/state_vocab/db_tlr.yaml`) and
> is always re-derivable from A. `visibility` (A') maps to `occlusion_state`;
> `truncation_state` defaults to non-truncated until a reviewer sets it.

> **Transitional (standardization in progress, 2026-07-23):** `export_awml.py`
> and `export_labels.py` (COCO) are **superseded** by L2 (`to_object_ann.py`) +
> external tooling and will be deprecated; the CVAT pair stays as the review UI
> for now. `traffic_signal_2d/v2` is the L3 internal form (A'), no longer the
> delivered "Tier B".

Rules across tiers: the **canonical state vocabulary (lamp tokens
`{color}-{shape}[-{direction}]`, sorted, comma-joined — see the state spec
below) is shared by every tier and every exporter**, so autolabels and human
corrections stay diffable in the same CVAT task. Derived strings (`state`,
`lamps[].label`) are always re-derivable from decomposed fields; decomposed
fields are the source of truth. Every produced file records provenance
(`meta.run_id`, models, thresholds) and a `schema_version`; schema changes bump
the version and keep a read fallback for one version.

## Detector model matrix (multiple ONNX patterns)

| model family | format to use | why |
|--------------|---------------|-----|
| YOLOX traffic_light (testS/M/L, 960/1280/1920x1280) | **int8 `.engine`** via `trt_run` | fp32 ONNX fires high-confidence FPs on motion blur that the deployed int8 engine does not; int8 = parity with the launched pipeline |
| CoMLOps-Large-Detection-Model | `.onnx` via onnxruntime (GPU: `run_gpu.sh`) | no engine exists; fp32 verified clean; TRAFFIC_LIGHT class filter |

The family is auto-detected (1 ONNX output = YOLOX, 3 = CoMLOps darknet,
`.engine` = TensorRT). Named presets under `configs/detectors/*.yaml` bundle
model path + recommended thresholds + tiles default; the preset name is
recorded in `meta.preset`, so a run is reproducible from one word:

```bash
python3 tlr_autolabel.py <dir> --preset yolox-1920-int8 --out-dir ./labels
# available: yolox-1920-int8 (recommended), yolox-1280-int8, yolox-960-int8, comlops-large
```

A detector must be chosen explicitly (`--preset` or `--detector`); explicit CLI
flags always override preset values (`--no-tiles` cancels a preset's
`tiles: true`). testM exists as a model directory but is empty — no preset.

### Portability: model paths and environment

Presets reference models by `${TLR_MODEL_ROOT}/...`, never an absolute path.
The root resolves as: `$TLR_MODEL_ROOT` if set, else the first existing of
`~/autoware_data`, `/opt/autoware/mlmodels` (the last matches Autoware launch's
`data_path`). `--detector` given directly also expands `~` / `$VARS` /
`${TLR_MODEL_ROOT}`. The resolved root is recorded in `meta.model_root`, and a
missing model fails fast printing the root and how to set it.

Environment setup:

- Python deps are pinned in `requirements.txt` (CPU/system profile: the active
  lines; GPU profile: the commented onnxruntime-gpu block).
- `setup_gpu_venv.sh` builds the onnxruntime-gpu venv that `run_gpu.sh` uses;
  its location is `$TLR_GPU_VENV` (default `~/.venvs/tlr_onnxgpu`).
- The `.engine` build uses `$CUDA_HOME` (default `/usr/local/cuda`) and the
  system libnvinfer (TensorRT 10.x).

> **`.engine` files are not portable**: a TensorRT engine is specific to the
> GPU architecture + TRT version it was built on. Presets that name a
> `.engine` assume it was built on this machine. On a different machine, either
> rebuild the engine from the ONNX, or point the preset at the `.onnx` and run
> via `run_gpu.sh`. A portable engine cache (build-on-demand keyed by
> GPU/TRT/model hash) is backlog — see PLAN "model management".

### Trying a new model (試走)

L1 doubles as a model test bench. A bare `--detector <path>` run uses the
plain single-pass configuration (tiles OFF by default — tiles are opt-in via
presets or `--tiles`), so a quick eyeball run is:

```bash
python3 tlr_autolabel.py <a_few_images> --detector new_model.onnx --viz --out-dir ./try
```

Flexibility contract (what happens when model specs change):

- **Input size** is read from the model (static NCHW required; dynamic dims
  exit with a clear message rather than misbehaving).
- **yolox head**: `num_class` is taken from the output shape (`4+1+C`);
  multi-class variants score as obj × best class. A wrong grid count (input
  size / stride mismatch) raises with a diagnostic instead of decoding garbage.
- **Family detection** is by output signature (1 output = yolox head, 3 =
  CoMLOps darknet); anything else exits telling you to add a decode — extend
  `Detector.__init__/detect` for new families.
- **Presets** may use `$ENV_VARS` and `~` in paths; a missing model path fails
  fast, naming the preset it came from.
- Per-model decode constants never live in code: yolox needs none, CoMLOps
  reads `comlops_large_detector_ml.param.yaml` (`--comlops-param`).
- The **classifier** is swappable the same way (`--classifier` /
  `--classifier-param`); a different classifier architecture needs its decode
  added alongside the LampRecognizer one.

## AWML training input (investigated 2026-07-18)

AWML (tier4/AWML) trains both TLR models — YOLOX_opt fine detector and
MobileNetv2 classifiers — from **T4dataset native annotations, not COCO**:
`tools/detection2d/create_data_t4dataset.py` reads `object_ann.json` /
`sample_data.json` / `category.json` / `attribute.json` via t4dev-kit and emits
mmdet info JSONs. The classifier uses the same script/annotation source
("annotations for traffic light in t4dataset are in 2d object detection
format").

In the db_tlr_v1..v7 datasets the **signal state is the per-box category
name** (not an attribute): a 32-entry vocabulary mixing legacy hyphen style
(`left-red`, `red-straight`, `leftdiagonal-red`) and current underscore style
(`red_left`, `red_straight`, `green_straight`, `crosswalk_red`, ...), collapsed
by config `class_mappings` to `BACKGROUND / traffic_light /
pedestrian_traffic_light` for the detector and kept fine-grained for the
classifiers.

**Design decision (2026-07-18): AWML is an adapter target, never the canonical
format.** The db_tlr shape is a lossy, legacy-constrained projection
(state-as-category conflates detection class and state; per-lamp detail,
confidence, and down-arrows cannot be represented; the 32-entry vocabulary
already mixes hyphen/underscore styles — historical drift). Canonical data
stays in Tier A/B; AWML output is produced one-way by an adapter:

- The adapter (`export_awml.py`) generates a **derived training dataset
  directory**: images and untouched annotation tables are symlinked from the
  source T4 dataset; only `object_ann.json` (+ merged `category.json` with the
  db_tlr state categories) are generated. The canonical dataset is never
  edited in place — stuffing every annotation into `object_ann.json` is
  exactly what we avoid.
- The canonical->db_tlr vocabulary mapping lives in **one shared file**
  (`configs/state_vocab/db_tlr.yaml`); every exporter that needs db_tlr names
  reads it. If AWML modernizes or a non-MobileNet training target appears,
  only an adapter changes.
- Names not in the db_tlr vocabulary must NOT be emitted (AWML's
  `class_mappings` raises on unmapped names); the adapter falls back to
  `unknown` and warns.
- Acceptance test: AWML `create_data_t4dataset.py` runs cleanly on the derived
  directory.
- COCO export stays a review/generic-tool view only; AWML never reads it for
  T4 training.

Canonical -> db_tlr mapping:

| canonical | db_tlr |
|---|---|
| `red-circle` / `amber-circle` / `green-circle` | `red` / `yellow` / `green` |
| arrow direction `up` / `left` / `right` / `up_left` / `up_right` | `straight` / `left` / `right` / `leftdiagonal` / `rightdiagonal` |
| combo (e.g. `green-arrow-up,red-circle`) | color first + arrows, underscore-joined: `red_straight` |
| `red-ped` / `green-ped` | `crosswalk_red` / `crosswalk_green` |
| `unknown` (no lamps) | `unknown` |
| arrow `down` / `down_left` / `down_right` | no db_tlr equivalent — map to `unknown` (open) |

## Open items

> Task tracking has moved to **PLAN.md** (single source for remaining work);
> the list below is kept as context for each item.
- **down/down_left/down_right arrows**: no db_tlr category; adapter maps them
  to `unknown` for now — confirm treatment.
- **AWML acceptance test**: run `create_data_t4dataset.py` against a derived
  dataset (needs an AWML checkout; not yet executed).
- **Dataset-side tools still read the pre-v1 IF** (`signal` key, paren arrow
  tokens): add the v1 fallback there or regenerate their inputs at integration.
- **deepen format mapping** (Tier C): owned by the converter repo; align labels.
- **L5 verification**: `ros2_pipeline/` unverified; acceptance test = parity
  with the launched int8 pipeline on the same frames.

---

## Pipeline

```
full image ──> tlr_detector_onnx.py ──> ROI crops ──> tlr_lamp_recognizer_onnx.py ──> color+shape per lamp
              (YOLOX detector)                        (YOLOX-based classifier = LampRecognizer)
```

`tlr_autolabel.py` runs the whole chain (detector -> per-ROI classifier) and
writes per-image JSON (+ optional annotated `--viz` PNG).

```bash
python3 tlr_autolabel.py <image_or_dir> --out-dir ./labels [--viz] [--drop-unknown]
```

### GPU (recommended — ~15-30x faster)

The system `onnxruntime` is the CPU-only build (~4 s/image at 1280). A dedicated
venv holds `onnxruntime-gpu` 1.23 + matching CUDA 12 / cuDNN 9 wheels, isolated
from the system python. Use the wrapper, which sets `LD_LIBRARY_PATH` to the
venv's bundled CUDA libs and runs the same script on the GPU (RTX 3060, ~44 ms
detector infer, ~0.3-0.4 s/image end-to-end):

```bash
./run_gpu.sh <image_or_dir> --out-dir ./labels [any tlr_autolabel flag]
```

venv: `/home/yoshiri/.venvs/tlr_onnxgpu` (rebuild: `python3 -m venv` it, then
`pip install onnxruntime-gpu numpy opencv-python-headless pyyaml
nvidia-cufft-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 nvidia-cuda-runtime-cu12
nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12`). The `nvidia-*-cu12` wheels are
required because the system CUDA 12.8 install is missing `libcufft.so.11`.

> Note: this venv runs the **fp32 ONNX** on GPU. Per the detector model matrix
> above, for the YOLOX detectors prefer the int8 `.engine` (`--detector *.engine`,
> also GPU) — fp32 has the motion-blur FP problem regardless of CPU/GPU. This
> venv is the right backend for CoMLOps-Large, which has no engine.

### Detector variations

`--detector` accepts two model families; the type is auto-detected from the ONNX
output count (1 output = YOLOX, 3 outputs = CoMLOps darknet). For YOLOX it also
accepts the TensorRT int8 `.engine` (run on GPU via the `trt_run` helper,
compiled from `trt_run.cpp` on first use).

**Prefer the int8 `.engine` over the fp32 `.onnx` for the YOLOX detectors.**
The fp32 ONNX fires high-confidence (0.9+) false positives on motion-blurred
road texture / bushes that the deployed int8 engine (EntropyV2 calibration)
does not — this is why offline ONNX results look worse than the launched
Autoware pipeline. Verified on identical letterboxed blobs: frame 00250 of the
c1af6a38 dataset gives 4 FPs at fp32 vs 0 at int8, while true positives on
frame 00000 are identical (scores slightly lower at int8, e.g. 0.99 -> 0.93).

```bash
# YOLOX traffic_light detector (any input size; 960/1280/1920x1280 variants)
python3 tlr_autolabel.py <dir> --detector \
  /home/yoshiri/autoware_data/traffic_light_detector_testL/yolox-sPlus-opt-Co_MLOps-traffic_light-1920x1280_20260706_best.onnx

# same model, int8 TensorRT engine (recommended: no blur/texture FPs, ~100x faster)
python3 tlr_autolabel.py <dir> --detector \
  /home/yoshiri/autoware_data/traffic_light_detector_testL/yolox-sPlus-opt-Co_MLOps-traffic_light-1920x1280_20260706_best.engine

# CoMLOps-Large-Detection-Model as detector (general 10-class model; only
# TRAFFIC_LIGHT detections are kept by default, see --det-classes)
python3 tlr_autolabel.py <dir> --detector \
  /home/yoshiri/autoware_data/traffic_light_classifier/CoMLOps-Large-Detection-Model-v1.0.1.onnx --det-score-thr 0.3
```

CoMLOps decode params live in `comlops_large_detector_ml.param.yaml`. No official
param file exists for that model; layout and preprocessing were reverse-engineered
(5 anchors x `[tx,ty,tw,th,obj,10cls]` per scale at strides 8/16/32, all sigmoid;
RGB + /255 letterbox — note this differs from YOLOX's raw-BGR) and the anchors
were fitted empirically against the T4 dataset 3D GT projections.

Detection-cleanup knobs (defaults are stricter than the ROS node because offline
autolabeling has no map_based_detector ROI prior to filter false positives):

| flag | default | effect |
|------|---------|--------|
| `--det-score-thr` | 0.35 | node-parity recall; L3 map matching filters FPs (was 0.5 — that missed mid-range signals) |
| `--det-nms-thr` | 0.35 | IoU-NMS; lower = merge overlaps harder (node uses 0.7). Plus a containment rule merges nested tight/whole-signal duplicates, keeping the higher-score box |
| `--min-box` | 8 | drop detections whose shorter side < N px |
| `--drop-unknown` | off | drop signals whose classifier found no lamp (recommended for state autolabeling) |
| `--cls-score-thr` / `--cls-nms-thr` | 0.2 / 0.2 | classifier thresholds (from car_traffic_light_classifier.param.yaml) |
| `--tiles` | off | add native-resolution tile passes on top of the full-frame pass (see below) |
| `--tile-overlap` | 128 | minimum overlap in px between neighbouring tiles |

### Tiled inference (`--tiles`)

The detector input (1920x1280) is below the native camera resolution
(2880x1860), so the full-frame letterbox pass downscales by 2/3 and starves
small distant signals of pixels. `--tiles` additionally runs detector-input-sized
crops covering the image at native resolution — for 2880x1860 that is exactly
4 tiles at offsets x∈{0,960}, y∈{0,580} with no resizing at all — and merges
everything with one global NMS in original pixel coords. Tile overlap (>=128px)
is far larger than a signal, so a signal clipped at one tile's edge is seen
whole by the neighbour and the containment NMS rule keeps the better box.
Measured effect: same true positives with clearly higher detector scores on
small/distant signals (e.g. 0.64->0.85, 0.53->0.78), no extra false positives
with the int8 engine. Cost: 5 detector passes per frame instead of 1 (cheap on
GPU via `.engine`).

### Per-image JSON schema (`tlr_autolabel/v1`) — internal IF, frozen 2026-07-18

```jsonc
{
  "schema_version": "tlr_autolabel/v1",
  "image": "data/CAM_FRONT/00000.jpg",     // PRIMARY: relative to the image root
                                            // (portable across machines/containers)
  "image_realpath": "/abs/path/...",        // convenience only; may be stale elsewhere
  "sample_data_token": "7d63cd...",         // T4 linkage via --t4-dataset, else null
  "channel": "CAM_FRONT",                   // from the path (uppercase component)
  "frame_index": 0,                         // numeric filename stem, else input order
  "width": 2880, "height": 1860,            // original image size, px
  "meta": {                                 // provenance: how these labels were made
    "run_id": "<--run-id or timestamp+rand>",
    "created_at": "2026-07-18T00:00:00+00:00",
    "detector": "<model file basename>",
    "detector_backend": "tensorrt-engine | ['CPUExecutionProvider'] | ...",
    "classifier": "<model file basename>",
    "tiles": true,
    "det_score_thr": 0.5, "det_nms_thr": 0.35,
    "cls_score_thr": 0.2, "cls_nms_thr": 0.2,
    "min_box": 8.0, "crop_pad": 0.0
  },
  "signals": [
    {
      "signal_id": "00000-00",              // deterministic: {frame stem}-{index};
                                            // same inputs -> same ids (not tracking)
      "detector_score": 0.93,
      "box_xyxy": [1683, 712, 1753, 761],   // ints, original px, x1/y1 exclusive
      "lamps": [                             // sorted by classifier confidence desc
        {"label": "red-circle", "color": "red", "shape": "circle",
         "arrow": null, "confidence": 1.0},
        {"label": "green-arrow-up", "color": "green", "shape": "arrow",
         "arrow": "up", "confidence": 1.0}
      ],
      "state": "green-arrow-up,red-circle"   // canonical rollup, see below
    }
  ]
}
```

The image root is `--image-root` if given, else the T4 dataset root with
`--t4-dataset`, else the input directory. `--t4-dataset <root>` resolves
`sample_data_token` from `annotation/sample_data.json` by realpath match.

Rules: `state` and `lamps[].label` are derived (see the state spec below) and
always re-derivable from the decomposed lamp fields; consumers should treat the
decomposed fields as the source and the strings as convenience. `signals` may
be empty (frame with no detection). `signal_id` is stable across re-runs of the
same inputs, not across frames (no tracking yet). Any schema change bumps
`schema_version`.

### Signal state spec

Each detected signal carries a canonical `state` string built from its
recognized lamps:

- lamp token grammar: `{color}-{shape}[-{direction}]`
  - color: `green | amber | red`
  - shape: `circle | arrow | u_turn | ped | number | cross`
  - direction (arrow lamps only, 8-way): `up | up_right | right | down_right |
    down | down_left | left | up_left`
- `state` = lamp tokens **sorted alphabetically**, comma-joined — the same
  physical state always serializes identically (classifier-confidence order
  does not); e.g. `green-arrow-up,red-circle`, `red-circle`, `red-ped`
- `unknown` when the classifier found no lamp in the box (facing away,
  occluded, too small). Filter these with `--drop-unknown` for state labels;
  keep them for detector-only training.

The `lamps` array keeps the full per-lamp detail (color/shape/arrow +
confidence), so `state` is always re-derivable. Files written before
2026-07-18 used a `signal` key with confidence-ordered `green-arrow(up)`
tokens; the exporter still reads them as a fallback.

### Exporting labels (COCO / CVAT)

The per-image JSON written by `--out-dir` is the source of truth (keeps xyxy
boxes, detector score, per-lamp color/shape/arrow + confidences — more than
either interchange format holds natively). `export_labels.py` converts a
directory of them, so the annotation-tool choice stays reversible:

```bash
# COCO: single labels.json; state + lamps kept in each annotation's "attributes"
python3 export_labels.py <labels_dir> --format coco --out labels_coco.json \
    --image-root <dataset_dir> [--category-mode single|state]

# CVAT for images 1.1 XML; state/confidence as CVAT box attributes
python3 export_labels.py <labels_dir> --format cvat --out annotations.xml \
    --image-root <dataset_dir>

# AWML adapter: derived T4 training dataset (symlinked view + generated
# object_ann.json/category.json with db_tlr state categories; see the AWML
# section above). Labels must carry sample_data_token (L1 run with --t4-dataset).
python3 export_awml.py <labels_dir> --t4-dataset <src_dataset> --out <derived_dir>
```

- `--image-root` makes image names relative (use the dataset root so paths match
  what COCO consumers / the CVAT task see); default is basenames only.
- `--category-mode state` (COCO) makes one category per distinct signal string
  instead of a single `traffic_light` category with attributes.
- The CVAT task must define label `traffic_light` with text attributes `state`
  and `confidence` for the XML to import cleanly. CVAT can also import the COCO
  file (it reads the `attributes` dict), so either path works.

### Map enrichment & RE fusion (L3)

Three tools take a T4 dataset from raw autolabels to a reviewed regulatory-
element timeline (ported from the dataset-side `tools/` on 2026-07-19; token
parsing unified on the canonical vocabulary via `state_tokens.py`, with legacy
paren-style fallback):

One command runs everything over one or more datasets (L1 per camera channel
with `--skip-existing` resume, then the three stages below):

```bash
python3 run_dataset.py <dataset_root> [...] --preset yolox-1920-int8 [--run-id ID]
```

Or stage by stage:

```bash
# 1) associate detections with lanelet2 traffic_light ways (Hungarian matching
#    on projected boxes) and resolve regulatory elements -> Tier B sidecar.
#    v1 labels are looked up by sample_data_token (file naming irrelevant);
#    both flat CHANNEL_frame.json dirs and per-channel subdirs are accepted.
python3 match_traffic_lights.py --dataset-root <dataset>

# 2) fuse across cameras and heads with map-bulb feasibility weighting ->
#    annotation/traffic_signal_re_timeseries.json (traffic_signal_re/v1)
#    + build/tl_match/re_verification_report.json (flags for review)
python3 aggregate_regulatory_signals.py --dataset-root <dataset>

# 3) interactive HTML timeline (one row per head group, flags marked)
python3 render_re_timeline.py --dataset-root <dataset>

# 4) optional richer state review: edit state by physical signal group/time.
#    The HTML also writes crop candidates under build/tl_match/re_review_assets/.
python3 render_re_review_timeline.py --dataset-root <dataset>
python3 make_re_review_template.py --dataset-root <dataset>
python3 apply_re_review.py --dataset-root <dataset> \
    --review annotation/traffic_signal_re_review.json \
    --output annotation/traffic_signal_2d_ann.reviewed.json
```

The lanelet2 `light_bulbs` color tags use `yellow`; canonical `amber` is
translated only at the map-comparison boundary (`state_tokens.bulb_color`).

Fusion repair policies (decided 2026-07-19):

- **Arrow directions snap to the map** when the detected direction is
  infeasible and the map's bulbs allow exactly one direction for that color
  (8-way classifiers miss by one sector; the map is authoritative for which
  arrows exist). Colors/shapes are never snapped — those mismatches may be
  stale-map bugs and only down-weight votes. Effect on c1af6a38: 30 snaps,
  cross-camera/head disagreements 36 -> 24 each.
- **Single-frame flips are repaired**, not just flagged: an observation whose
  neighbors agree on a different state takes the neighbors' state
  (`state_original` + `state_source: temporal_fix` keep the raw value).
- **Bounded gap filling** (`--fill-gaps`, default on; decided 2026-07-20,
  refining the earlier no-interpolation stance): when a regulatory element is
  matched before and after a short detection dropout (≤ `--max-gap-frames`
  consecutive missing frames, both sides the same state), the map traffic light
  is projected into each missing frame as an interpolated box (accurate
  map-anchored box + RE id, state copied from the bracket). Marked
  `source_type=interpolated`, `review_status=unchecked` so review/eval can keep
  or drop them. Only bridges same-state gaps a signal is unlikely to have
  changed across — not open-ended extrapolation. On c1af6a38: recovers small/distant signals detection missed. (`raw_state` is empty on
  these — they are not detections, so they never enter the prediction side of
  evaluation.)
- **Map-presence backfill** (`--map-fill`, default on; `--map-fill-max-distance`
  50m, front-facing only): a near signal the detector missed *entirely* still
  gets an `unknown` box at its projected map position — but only where a real
  detection of the *same* signal within `--map-fill-window` frames corroborates
  the projection. Without that guard the map alone drops boxes on occluded /
  mis-projected empty regions (verified: blind map-fill put boxes on empty sky).
  Boxes are clipped to the image and dropped if mostly off-frame; position
  carries the map projection's ~10-20px offset (no detection to snap to), so
  they are `unknown`/`source_type=map_presence` for review, not final GT.
- **Unmatched detections keep their info, not dropped silently**: the
  `unmatched_reason` and the nearest in-view map candidate
  (`map_candidate_id` / `regulatory_element_id_candidate`) are recorded on the
  Tier B box (and in `build/tl_match/match_report.json`), so a real signal that
  failed the geometric match still carries its likely RE for review. Reasons:
  `state_unknown_backside` / `no_map_candidate_in_view` / `candidate_taken` /
  `beyond_gate` / `geometry_mismatch`.

### Tier B schema: `traffic_signal_2d/v2` (per-detection sidecar)

Written by `match_traffic_lights.py` to `annotation/traffic_signal_2d_ann.json`.
This is the IF the L4 converters consume; the **attribute-level contract
(types, defaults, edit rules) lives in `docs/cvat_interop.md`** — the summary
below is for orientation. v1 read-fallback: `detector_signal` -> `raw_state`.

```jsonc
{
  "schema_version": "traffic_signal_2d/v1",
  "source": "map_projection_auto",        // or "manual" etc. for other producers
  "annotations": [
    {
      "token": "<md5(sample_data_token, det index, box)>",  // deterministic
      "sample_token": "...", "sample_data_token": "...",
      "channel": "CAM_FRONT",
      "filename": "data/CAM_FRONT/00000.jpg",
      "timestamp": 1783325716091628,       // µs, from sample_data
      "label": "traffic_light",
      "box2d": [x0, y0, x1, y1],           // float px, original image
      "occluded": false, "z_order": 0,
      "attributes": {
        "state": "green-arrow-up,red-circle",  // canonical vocab, normalized
        "signal_kind": "vehicle",              // vehicle | pedestrian | unknown
        "visibility": "unknown",               // review field
        "review_status": "unchecked",          // unchecked | accepted | fixed | rejected
        "map_traffic_light_id": "1234",        // lanelet2 way id, "" if unmatched
        "regulatory_element_id": "10302,10304",// comma-joined relation ids, "" if none
        "raw_state": "...",                    // detector state as emitted by L1
        "detector_score": "0.93",              // string; "" when absent
        "source_type": "auto"                  // auto | cvat (manual round-trip)
      }
    }
  ]
}
```

Sidecar companion: `build/tl_match/match_report.json` (`params`, `stats`,
per-frame `frames` diagnostics — projected candidates, IoU, distances).

### `traffic_signal_re/v1` (regulatory-element time series)

Written by `aggregate_regulatory_signals.py` to
`annotation/traffic_signal_re_timeseries.json`:

```jsonc
{
  "schema_version": "traffic_signal_re/v1",
  "source": "annotation/traffic_signal_2d_ann.json",
  "series": [
    {
      "regulatory_element_id": "10302",
      "member_ways": ["1234", "1235"],     // lanelet2 traffic_light ways (role=refers)
      "n_observations": 42,
      "segments": [                         // run-length view of the state sequence
        {"state": "red-circle", "t_start": ..., "t_end": ..., "n_frames": 3}
      ],
      "observations": [                     // one per sample where any head was seen
        {
          "sample_token": "...", "timestamp": ...,
          "state": "green-arrow-up,red-circle",   // canonical vocab, "unknown" allowed
          "elements": [{"color": "...", "shape": "...", "arrow": "...|null"}],
          "head_states": {"1234": "red-circle"},  // per-way fused state
          "n_heads": 2,
          "confidence": 0.87,               // winner weight / total vote weight
          "flags": ["cross_head_state_disagreement"]
        }
      ]
    }
  ]
}
```

### `traffic_signal_re_review/v1` (editable state review)

Written by `make_re_review_template.py` or exported from
`render_re_review_timeline.py`, then applied by `apply_re_review.py`. This file
is intentionally a sidecar: it records human decisions over physical signal
groups and time intervals without overwriting the CVAT/Tier B geometry edits.

Physical signal groups are keyed by the lanelet2 `traffic_light` member ways,
not by a single regulatory element, because multiple RE relations can share the
same heads:

```jsonc
{
  "schema_version": "traffic_signal_re_review/v1",
  "source_timeseries": "annotation/traffic_signal_re_timeseries.json",
  "groups": [
    {
      "signal_group_id": "ways:2180,2182",
      "member_ways": ["2180", "2182"],
      "regulatory_element_ids": ["10360", "10362"],
      "decisions": [
        {
          "start_sample_token": "...",
          "end_sample_token": "...",
          "start_timestamp": 1783325718047571,
          "end_timestamp": 1783325720947572,
          "state": "green-arrow-up,red-circle",
          "review_status": "fixed",
          "source": "manual_timeline_review",
          "note": ""
        }
      ]
    }
  ]
}
```

Application rules:

- `accepted` / `fixed` decisions overwrite Tier B `attributes.state`, derive
  `signal_kind`, and set `review_status`.
- `rejected` decisions set only `review_status=rejected`.
- `unchecked` decisions are retained in review JSON but are not applied.
- `box2d`, `visibility`, `map_traffic_light_id`, `raw_state`,
  `detector_score`, and `source_type` are never changed by this layer.

The normal human loop is: CVAT first for bbox/visibility/reject/map-id fixes,
then re-run aggregation, then RE timeline review for state intervals, then
`apply_re_review.py` to produce the reviewed Tier B sidecar consumed by L6.

### Map-less T4 `object_ann.json` GT evaluation

Some evaluator datasets are nuScenes/T4-derived 2D GT only: state is encoded as
`annotation/object_ann.json` category names (`red_right`, `crosswalk_green`,
...), with no usable lanelet2 traffic-light / regulatory-element association.
Use the L1-vs-T4 evaluator for those datasets:

```bash
python3 eval_l1_vs_t4_gt.py \
  --dataset-root <dataset_root> \
  --pred-dir <dataset_root>/tlr_autolabel/<CHANNEL> \
  --iou 0.3
```

It joins by `sample_data_token` and bbox IoU only, normalizes db_tlr category
names to the canonical state vocabulary, and reports detection precision/recall,
state accuracy, confusion, and per-category recall. It does not require or emit
`map_traffic_light_id`, `regulatory_element_id`, or RE metrics.

Flag vocabulary (review triage; also summarized in
`build/tl_match/re_verification_report.json`):

| flag | meaning |
|---|---|
| `cross_camera_state_disagreement` | cameras disagree on one head's state |
| `cross_head_state_disagreement` | heads of one regulatory element disagree |
| `partial_unknown` | some sources voted unknown |
| `single_frame_flip_fixed` | 1-frame glitch repaired from neighbors (raw in `state_original`) |
| `arrow_dir_snapped:<from>-><to>` | arrow direction corrected to the map's unique feasible direction |
| `color_not_in_map_bulbs:<c>` / `arrow_without_map_bulb:<d>` / `arrow_dir_mismatch:<d>(map:<dirs>)` | detected state impossible for the map's bulb layout (vote down-weighted, never snapped) |
| `ped_on_vehicle_bulbs` | pedestrian lamp matched to a vehicle head |
| `no_bulb_info_in_map` | map has no light_bulbs entry to validate against |

Fusion weighting: each vote is scaled by `0.25^n_bulb_flags` (map-infeasible
states count less) and, at RE level, by the head's own fused confidence.
`re_verification_report.json` orders `flagged_observations` by
`review_priority = n_flags + (1 - confidence)` for CVAT triage.

## Preprocessing (must match the Autoware nodes exactly)

Verified against the C++ nodes; these details are load-bearing for correctness:

| stage | color | resize | normalize |
|-------|-------|--------|-----------|
| detector (`tensorrt_yolox`) | **BGR** — node uses `cv_bridge::toCvCopy(BGR8)`, kernel keeps channel order | **letterbox** `scale=min(W/w,H/h)`, top-left, pad **114** | none (`norm_factor=1.0`, raw 0-255) |
| classifier (`CnnLampRecognizer`) | **RGB** — node uses `cv_bridge::toCvCopy(RGB8)` then `blobFromImages(swapRB=false)`, so RGB reaches the model | plain resize to 256×256 | `scale=1/255` |

Because `cv2.imread` returns BGR: the detector feeds BGR as-is; the classifier
uses `swapRB=True` to convert to RGB. Getting this wrong silently corrupts
**colors only** (a green arrow scored amber=0.999 under BGR vs green=1.0 under RGB)
while detection/boxes still look fine — so always eyeball colors when validating.

Detections that fall inside the letterbox 114-padding region are dropped
(otherwise the detector fires on the pad seam and clips to 1px ghosts at the
bottom/right image edge).

## Scripts

### `tlr_detector_onnx.py`
YOLOX traffic-light **detector** (`yolox-sPlus-opt-Co_MLOps-traffic_light-*`).
Decodes the (num_grids, 6) output — `4 box + 1 obj + 1 class`, `num_class=1`.
Grid/stride decode matches `tensorrt_yolox` detector node. Edit the paths at the
top; detector input size must match the model (960 or 1280).

### `tlr_lamp_recognizer_onnx.py`
YOLOX-based **classifier**, reproducing `autoware_traffic_light_classifier`
node with `classifier_type=2` (LampRecognizer):
`.../autoware_traffic_light_classifier/src/classifier/cnn_lamp_recognizer.cpp`

- model: `traffic_light_lamp_recognizer_comlops.onnx` — input `(N,3,256,256)`,
  output `(N,48,64,64)` = `(N, num_anchors*chans_per_anchor, gh, gw)`
- preprocess: `blobFromImage(scale=1/255, swapRB=False)` (BGR kept, no mean/std)
- decode: anchor-based (YOLOv4-style) — `bx = x + scale_x_y*tx - bbox_offset`,
  `bw = pw*(tw*2)^2`; per-anchor channels = `4 box + obj + 3 color + 6 type + cos + sin`
- outputs per lamp: color (green/amber/red), shape (circle/arrow/u_turn/ped/
  number/cross), and 8-way arrow direction from `atan2(sin, cos)`
- NMS (IoU) identical to `runNms()`

```bash
python3 tlr_lamp_recognizer_onnx.py <crop_or_dir> [--score-thr 0.2] [--nms-thr 0.2]
```

## Assets in this dir
- `traffic_light_lamp_recognizer_comlops.onnx` (from autoware_data / model zoo v4)
- `lamp_recognizer_ml.param.yaml` — model_params (anchors, indices, scale_x_y).
  Canonical source:
  `https://awf.ml.dev.web.auto/perception/models/traffic_light_classifier/v4/lamp_recognizer_ml.param.yaml`
- `lamp_labels.txt` — full-signal combo labels (reference only; the decoder emits
  per-lamp color+shape, not these combos)

## Provenance
The detector script was recovered from a prior Claude Code session
(`.claude-mine` session `3778405d`, 2026-07-10), where the inference was run as
inline `python3 - <<EOF` heredocs (never saved as a `/tmp` file). The classifier
script was newly written on 2026-07-17 from the Autoware node source.
