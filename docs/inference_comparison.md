# Inference comparison harness (any input × any model combination)

Status: **implemented 2026-08-07**. Feeds arbitrary images, a video, a rosbag or
a T4 dataset through N detector/classifier configurations and compares their raw
output without ground truth and without a map.

## Why this exists

Two things were in the way of "just run every combination and diff the output":

1. **Input was hard-wired to an image directory.** The frame identity written
   into Tier A (`image`, `frame_index`, `channel`, `sample_data_token`) was
   derived inside `cli/autolabel.py:main` from a sorted glob, so a video or a bag
   had no way in.
2. **Configuration was an argparse Namespace.** Presets were overlaid onto it
   via `ap._actions`, so a configuration could not be built without an
   `ArgumentParser` and no single process could hold two of them.

Comparison itself was also only available in map-referenced form:
`compare_runs.py` needs a T4 dataset + lanelet2 map to compute
`candidate_coverage`. That is the better metric when a map exists — it is a real
recall proxy — but it says nothing about a random mp4.

## Shape of the solution

Two tools, split along the repo's layer rule ("no layer after L1 re-runs
inference"):

| tool | layer | reads | writes |
|---|---|---|---|
| `scripts/run_compare.py` | L1 (needs GPU + models) | frame source + matrix YAML | one **ordinary Tier A label directory per configuration** + `run_manifest.json` |
| `scripts/compare_naive.py` | L6 (pure analysis) | those label directories | `compare_naive.{json,md}` + side-by-side grids |

Because the runner's output is plain Tier A, every existing tool still applies to
it: `match_traffic_lights.py`, `compare_runs.py` (when the input was a T4
dataset), `render_l1_video.py`, `eval_l1_vs_t4.py`. The harness adds no new
delivery format.

```
                    ┌─ ImageDirSource ─┐
frames.materialize ─┤  VideoSource     ├→ Frame ─→ Pipeline(cfg_1) ─→ labels/cfg_1/*.json ─┐
(video/bag only)    │  RosbagSource    │          Pipeline(cfg_2) ─→ labels/cfg_2/*.json ─┼→ compare_naive
                    └─ T4DatasetSource ┘          Pipeline(cfg_N) ─→ labels/cfg_N/*.json ─┘
```

## Frame sources

`tlr_autolabel/frames/` — a source yields `Frame`s and owns the frame identity.

| kind | `frame_id` | `sample_data_token` | notes |
|---|---|---|---|
| `images` | file stem | from `--t4-dataset` if given | byte-for-byte the historical L1 rules (glob order, `int(stem)`-or-position `frame_index`, first ALL-CAPS path component as channel) |
| `video` | source frame number, `%06d` | – | `stride` / `start` / `max_frames`; ids stay stable across strides so a stride-2 run still lines up with a stride-1 run |
| `rosbag` | `%06d` per topic (`<channel>/%06d` when multi-topic) | – | mcap/sqlite3, `sensor_msgs/Image` + `CompressedImage`, channel from the topic path |
| `t4` | file stem (`<channel>/<stem>` when multi-camera) | yes | camera `sample_data` rows in capture order |
| `materialized` | as extracted | preserved | a frame directory written by `frames.cache.materialize` |

`frame_id` is the join key: it names the Tier A file and is what the comparator
uses to line runs up. It may contain `/`; writers create the parent directory.

### Why video/bag frames are extracted first

`run_compare.py` materializes any source whose frames are not already files
(`has_files = False`) into `<out>/frames/` before running anything:

- **Identical pixels for every configuration.** Decoding a video or bag once per
  configuration is not guaranteed to yield identical arrays (seek behaviour,
  codec threading, dropped frames). A comparison that cannot separate a model
  difference from a decode difference is worthless. PNG by default (lossless);
  `--frame-format jpg` when size matters — the re-encode is then applied once,
  uniformly.
- **Downstream tools want real files.** The review video / CVAT export /
  timeline renderers open `image_realpath`; a synthetic `bag:/topic#123`
  reference is not openable. After extraction, `image_realpath` points at a real
  PNG, so a bag run can go through the normal review path.

Re-running the same extraction is free: a matching `origin` in `frames.json`
reuses it.

## Configuration

`InferenceConfig` (`tlr_autolabel/inference/config.py`) is the unit of
comparison — one frozen dataclass resolved as
**defaults ← preset ← explicit overrides** (`resolve_config`). The CLI path
(`config_from_args`) reproduces the historical rule exactly: any flag the user
typed beats the preset; anything left at its default is the preset's to set.

A matrix file is a frame source plus named configurations:

```yaml
# configs/compare/detector-matrix.yaml
schema_version: tlr_compare_matrix/v1
defaults: {det_score_thr: 0.35}          # applied to every combo
combos:
  - {name: S960,       preset: yolox-960-int8}
  - {name: S960_tiles, preset: yolox-960-int8, overrides: {tiles: true}}
  - {name: boxes_only, preset: yolox-960-int8, overrides: {classifier: none}}
```

Unknown keys in `defaults`/`overrides`/preset are hard errors — a typo must not
silently change nothing. `classifier: none` runs the detector alone (state is
always `unknown`), which is what you want when the question is recall.

Configurations run **one at a time, frames inner**: a `.engine` keeps a
`trt_run` helper process and a deserialized engine resident, so holding several
1920×1280 int8 engines at once is how the GPU runs out of memory.
`Pipeline.close()` (→ `TrtServer.close()`) releases each one before the next.

## Tier A additions

`tlr_autolabel/v1` is unchanged for existing runs. The additions are optional
keys that only appear when something actually carries them:

| key | when |
|---|---|
| `source` | non-file frame sources (`{kind, uri, topic, frame_number, stamp_ns, origin_ref, …}`) |
| `timestamp_us` | the frame has a timestamp (video position, bag header stamp, T4 `sample_data.timestamp`) |
| `timing_ms` | `--timing` (always on for comparison runs): `{detector, classifier, total, crops}` |
| `meta.detector_type`, `meta.classifier_type` | always — you cannot compare runs without knowing the family |
| `meta.detector_sha256`, `meta.classifier_sha256` | `--model-digest` (on by default in comparison runs) |

Verification of the refactor: 14 payloads across 6 CLI invocations were captured
before and after, and differ only by the two `meta` family keys — same boxes,
same states, same paths, same `frame_index`, same key order.

## What the comparison measures

`tlr_autolabel/compare/naive.py`, all GT-free and map-free:

**Per configuration** — frames, detections, detections/frame, unknown rate,
state mix, detector-score and box-size percentiles, `timing_ms` percentiles.

**Stability over time** (the cheapest quality signal a video/bag offers) —
detections linked across consecutive frames by IoU:
`frame_to_frame_match_rate` (does a detection survive to the next frame at all)
and `state_flip_rate` (how often the state changes across such a link). A
configuration that flickers is worse for downstream review even when its
per-frame counts look identical to a steadier one.
Reported as `n/a` when the run was subsampled (`--frame-stride`, or a sparse
image set): consecutive labels are then seconds apart, nothing links, and the
raw numbers would read as flicker. The report states the median frame gap
instead — measured from `frame_index`, so it works for any source.

**Pairwise against a reference** — box agreement (Jaccard of IoU-matched
detections), matched-IoU percentiles, state agreement (also excluding
both-unknown pairs), element-level overlap, top state disagreements, and the
**median score of the disagreeing boxes**: a low value means the difference is
marginal detections, a high value means the configurations genuinely see
different things.

**Consensus (optional, ≥3 configurations)** — majority vote as a pseudo
reference, giving each configuration a recall/precision/state-agreement number
against the pack. This is **correlated by construction**: configurations sharing
a detector share its misses. The report says so and lists which configurations
share a detector (by sha256 when recorded). Use it to spot the odd one out, not
as evidence of accuracy.

**Grids** — the N most-disagreeing frames rendered as one panel per
configuration, plus an optional side-by-side mp4. Usually the actual deliverable.
Boxes and labels are drawn *after* the panel resize, so they stay legible when a
2880 px frame is fitted into an 800 px panel. A traffic light is a fraction of a
percent of that frame, so `--grid-crop` crops each panel to the union of *all*
configurations' detections (a box only one of them found is still inside), and
`--grid-width` trades file size for detail. When the configurations agree
everywhere, the grids fall back to an evenly spaced sample — "nothing to show" is
the wrong answer to "show me the output".

### What it cannot say

Without GT, more detections is not better and a disagreement is not an error.
PLAN 2.8/2.9 is the cautionary tale: S960 > L1920 on a wide-angle camera
reversed on a telephoto one, and the extra detections were `state=unknown`
distant signals. Use this harness to locate differences; use `compare_runs.py`
(map-referenced) or `eval_l1_vs_t4.py` / `evaluate_signals.py` (GT) to decide
which difference is an improvement.

## Usage

`--frame-stride` / `--max-frames` apply to every input kind (per topic for a bag,
per camera for a T4 dataset), so a trial run is cheap before committing to a full
pass.

```bash
# a directory of images, two presets, compare right away
python3 scripts/run_compare.py ~/data/CAM_FRONT --out build/compare/cam_front \
    --combo S960=yolox-960-int8 --combo L1920=yolox-1920-int8 --compare

# a video, the shipped detector ablation, every 5th frame
python3 scripts/run_compare.py --matrix configs/compare/detector-matrix.yaml \
    --video drive.mp4 --frame-stride 5 --out build/compare/drive --compare

# a rosbag (needs a sourced ROS 2), one camera topic
source /opt/ros/humble/setup.bash
python3 scripts/run_compare.py --matrix configs/compare/detector-matrix.yaml \
    --bag ./rosbag2_2026_08_07 --bag-topics /sensing/camera/camera6/image_rect_color \
    --out build/compare/bag --compare

# a T4 dataset (keeps sample_data_token, so the result also feeds L3/L6)
python3 scripts/run_compare.py --t4-dataset ~/.webauto/.../0 --channels CAM_FRONT \
    --matrix configs/compare/threshold-sweep.yaml --out build/compare/ds --compare

# re-analyse an existing run at another IoU threshold, no GPU needed
python3 scripts/compare_naive.py --manifest build/compare/drive/run_manifest.json \
    --iou-thr 0.3 --reference S960 --grid-top 30 --grid-video build/compare/drive/grid.mp4

# compare label directories that were produced separately
python3 scripts/compare_naive.py old=./labels_old new=./labels_new --out build/compare/ab
```

On GPU, put `run_gpu.sh`'s environment around the runner the same way
`run_dataset.py` does for L1 (the engines path needs the venv's CUDA libs).

Verified end to end on 2026-08-07 with the real Autoware model-store stack
(`autoware-mlmodels-960-onnx`, CPU onnxruntime) over a T4 dataset: two
thresholds × 4 frames of `CAM_TRAFFIC_LIGHT_NEAR`, giving Tier A with
`sample_data_token` intact, `timing_ms` around 0.7-0.8 s/frame on CPU, a full
agreement report, and legible cropped grids. The `.engine` + `--bag` paths have
unit coverage but have not been run on real hardware/bags yet (PLAN item 12).

## Extending

- **New detector family**: one module under `tlr_autolabel/inference/models/`
  with `@register_detector`, then `detector_type:` in a preset —
  `docs/model_interface.md`.
- **New classifier family**: same, with `@register_classifier` and
  `classifier_type:`. The canonical lamp list
  (`{label, color, shape, arrow, confidence}`) is the contract; a family that
  predicts one label for the whole signal returns a single-element list. That is
  what keeps `state` comparable across families.
- **New frame source**: subclass `FrameSource`, implement
  `iter_frames(skip=None)` + `describe()`, register it in
  `frames.build_frame_source`. Set `has_files = False` if frames are not files,
  and the harness will materialize them.

## Decisions worth knowing

- **`tlr_autolabel/v1` was not bumped.** Every addition is an optional key and
  no existing key changed meaning, so consumers keep working; a version bump
  would have forced every reader to change for information they do not use.
- **`frame_id` conventions are frozen** (table above). The comparator joins on
  them, so changing them would silently break comparisons against older runs.
- **The bag reader is a reader, not a node.** It lives in
  `tlr_autolabel/frames/rosbag.py` with lazy `rosbag2_py`/`rclpy` imports and
  does not touch `ros2_pipeline/`, so "L5 is out of the dependency graph" still
  holds and the package stays importable without ROS. It decodes with
  numpy/cv2 rather than `cv_bridge`, which `ros2_pipeline/README.md` documents
  as fragile against NumPy 2.x.
- **Consensus is shipped but hedged.** It was tempting to leave it out; it earns
  its place for spotting an outlier configuration, and the correlation caveat is
  printed in the report rather than left to memory.
