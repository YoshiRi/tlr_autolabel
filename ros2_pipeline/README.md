# TLR autolabeling via the LIVE Autoware graph (option ②)

Autolabels a directory of images with the **actual production TLR recognition
nodes** (TensorRT int8), instead of the standalone ONNX reimplementation in the
parent directory. This is the most "production-faithful" path: identical
preprocessing, int8 numerics, and decode to what runs on the vehicle.

```
feeder ─publish(bgr8, stamp.sec=frame#)→ /tlr_autolabel/image
    │
    ├─► tensorrt_yolox_traffic_light_detector ─objects→ tlr_yolox_roi_adapter ─rois┐
    │                                                                               │
    ├─► car_traffic_light_classifier (type 2 = LampRecognizer) ◄──image + rois─────┤
    └─► pedestrian_traffic_light_classifier (type 1 = CNN)      ◄──image + rois─────┘
                                   │ traffic_signals (per traffic_light_id)
                                   ▼
                          collector ─join rois+signals by (stamp, id)→ <frame>.json
```

## Pieces
- `tlr_image_feeder.py` — publishes each image once, frame index encoded in
  `header.stamp.sec`; writes `frame_map.json` (sec → image path) up front. Waits
  for 3 subscribers (detector + 2 classifiers) before feeding.
- `tlr_label_collector.py` — subscribes rois + car/ped traffic_signals, joins by
  `traffic_light_id` within a stamp, writes one JSON per frame (named by source
  image via frame_map) in the **`tlr_autolabel/v1` schema** (identical to the
  offline pipeline, so the two are directly comparable). **Must** subscribe or
  the classifier (lazy) won't run. `detector_score` is `null` (not carried on
  `TrafficLightRoi`); a known, expected difference.
- `run_ros2_autolabel.sh` — orchestrates: build engines once → launch graph →
  collector → feeder → teardown.
- `parity_check.py` — L5 acceptance: compare ROS2 output vs offline output on the
  same frames (see below).

## L5 parity workflow (acceptance: ROS2 == offline int8)

Run the **offline** pipeline and the **live ROS2** pipeline on the SAME frames,
then compare. Per-frame JSON is named by image stem on both sides, so frames line
up automatically.

```bash
# 1) offline (int8 engine detector) — from the repo root
./run_gpu.sh <IMG_DIR> --out-dir /tmp/par/offline --image-root <DATASET_ROOT> \
    --detector <...>.engine

# 2) live ROS2 graph on the same images (real terminal; last arg = image_root)
ros2_pipeline/run_ros2_autolabel.sh <IMG_DIR> /tmp/par/ros2 cam_front 2 0 <DATASET_ROOT>

# 3) parity report
python3 ros2_pipeline/parity_check.py /tmp/par/offline /tmp/par/ros2 --iou 0.5 --mode both \
    --out /tmp/par/report.json
```

`parity_check.py`:
- **bbox** mode: greedy IoU matching (`--iou`), reports matched / miss
  (offline-only) / extra (ros2-only), mean IoU, recall/precision/f1 vs offline.
- **state** mode: on IoU-matched pairs only (state can't be right if the box is
  wrong), compares the canonical order-independent state via `state_tokens`
  (legacy `green-arrow(up)` and canonical `green-arrow-up` normalize equal;
  `yellow`→`amber`). Reports agreement + the mismatching pairs.
- `both` (default) prints both. L5 passes when bbox f1 and state agreement are
  ~1.0 on a representative frame set.

## Run (in a REAL terminal, not an automated sandbox)

```bash
# args: <image_dir> <out_dir> [camera_ns] [rate_hz] [limit(0=all)]
scripts/tlr_autolabel/ros2_pipeline/run_ros2_autolabel.sh \
    ~/.webauto/.../0/data/CAM_FRONT  ~/tlr_labels_ros2/CAM_FRONT  cam_front  2  0
# engines are cached after the first run; subsequent runs: SKIP_BUILD=1 ...
```

Config baked into the orchestrator: `car_classifier_type=2` (LampRecognizer, the
YOLOX classifier) with model `traffic_light_lamp_recognizer_comlops.onnx`;
`pedestrian_classifier_type=1` (CNN, prebuilt engine). Both car and ped pointing
at the same lamp onnx makes them RACE to build the same engine file on one GPU —
keep them on different models, or pre-build once.

## Status / caveats
- **Validated**: the graph loads all engines (detector 20 MiB, lamp 4 MiB, ped
  CNN 6 MiB), selects the GPU, and the feeder→collector chain produces correct
  per-frame JSON, e.g. `{"signal":"red-circle","box_xyxy":[2253,915,2262,933],
  "lamps":[{"color":"red","shape":"circle","confidence":0.84}]}`.
- **Environment**: the live multi-process ROS 2 graph (DDS + GPU) needs a normal
  interactive shell. It will not run to completion inside a restricted automated
  sandbox (processes get reaped after a few seconds). Run it from a real terminal.
- **NumPy**: ROS Humble's `cv_bridge` is built against NumPy 1.x. A NumPy 2.x in
  the user site (`~/.local`) shadows the system NumPy and **breaks** cv_bridge
  (`_ARRAY_API not found`; `cvtColor2` fails -> images may publish malformed) —
  it is not merely a warning. `run_ros2_autolabel.sh` sets `PYTHONNOUSERSITE=1`
  so the feeder/collector use the system NumPy 1.x (no package changes). If you
  invoke the feeder/collector directly, prefix them with `PYTHONNOUSERSITE=1`.
- Engine build is one-time and slow; the detector engine and
  `traffic_light_lamp_recognizer_comlops.engine` are already built under
  `/opt/autoware/mlmodels/`.

## vs the ONNX path (parent dir)
Both produce the same JSON schema. The ONNX path (`../tlr_autolabel.py`, GPU via
`../run_gpu.sh`) is portable and fast to iterate; this ROS path is the ground
truth (exact production nodes, int8). Use the ONNX `.engine` detector option to
get most of the int8 benefit without the full graph.
