# TLR evaluation design — evaluating the live ROS node against our GT

Goal: use the GT built by this repo (Tier B boxes+state, RE time series) to
evaluate the **actual Autoware ROS node pipeline** at three decoupled levels:

1. **ROI / detection** — did the detector find the signal boxes?
2. **classification** — given a signal ROI, is the state (color/shape/arrow) right?
3. **RE / matching+merge** — after map association + multi-camera/head fusion,
   is the per-regulatory-element signal state right over time?

## Architectural principle

**The node's output at each stage is converted to our canonical formats, then
scored against GT with the same engine that scores the offline pipeline.** The
evaluator is source-agnostic: "offline autolabel" and "ROS node" are just two
runs; GT is shared. So we reuse L3 (`match_traffic_lights.py`) and L6
(`evaluate_signals.py`) instead of writing node-specific metrics.

```
ROS node graph ──topics──► collector ──► node run in tlr_autolabel/v1 (+ RE)
                                              │
GT (reviewed Tier B + RE timeseries) ─────────┤
                                              ▼
                             two-source evaluator (join by IoU / RE id + time)
                             → detection P/R/IoU · state accuracy · RE accuracy
```

`ros2_pipeline/tlr_label_collector.py` already emits `tlr_autolabel/v1`, so a
node run flows straight into `match_traffic_lights.py` → Tier B, exactly like an
offline run. What's new is a **two-source evaluator** that joins a *prediction*
run (node) to a *GT* run (reviewed offline) — the current
`evaluate_signals.py` GT block lives inside one sidecar (review_status), which
is fine for "offline vs its own review" but not for "node vs GT".

## The three levels

| level | node output (topic) | GT source | join key | metrics |
|---|---|---|---|---|
| **1 detection** | detected ROIs (boxes) — `.../detection/rois` | reviewed Tier B boxes that are real (accepted/fixed; rejected excluded) | per frame, box IoU | precision, recall, IoU dist, by distance/facing/visibility; missed signals, false ROIs |
| **2 classification** | per-ROI state — `.../classification/{car,pedestrian}/traffic_signals` (keyed by traffic_light_id) | reviewed Tier B `state` (legible only: visibility full/partial) | frame + matched ROI (or GT ROI) | state accuracy given a correct ROI; color/shape/arrow confusion; by distance/visibility. **decoupled from detection** |
| **3 RE / merge** | final per-RE `TrafficSignalArray` after map assoc + multi-camera fusion + arbiter (keyed by regulatory_element_id) | RE timeseries GT (`traffic_signal_re/v1`, human-reviewed via the time-series tool) | regulatory_element_id + timestamp (nearest) | per-RE state accuracy over time, state-change latency, flip/stability vs GT segments |


## Bag-based flow (chosen 2026-07-22)

The user runs the `autoware_ml_model_launchers` detect+classify launcher over a
rosbag and hands over a **bag containing the node output**. We evaluate that bag
against our GT (levels 1 & 2). Level 3 (per-RE, final `/traffic_signals`) is
already covered by tier4's `driving_log_replayer_v2` traffic_light use case
(topic `/perception/traffic_light_recognition/traffic_signals`,
`tier4_perception_msgs/TrafficSignalArray`, GT from t4dataset, ±75ms temporal
match, per-label TP/FP/FN + confusion) — we align our temporal tolerance to it.

Flow:
```
node bag ──bag_to_labels.py──► tlr_autolabel/v1 (roi box + classified state,
   (TrafficLightRoiArray            time-aligned to GT keyframes, ±75ms)
    + TrafficLightArray)              │
                                      ▼  match_traffic_lights.py
                                 node Tier B
                                      │  eval_vs_gt.py --gt <reviewed GT>
                                      ▼
                    detection P/R/IoU by distance (level 1) +
                    classification state accuracy + confusion (level 2)
```

- `bag_to_labels.py` (built 2026-07-22): reads `*/detection/rois`
  (TrafficLightRoiArray) + `*/classification/traffic_signals`
  (TrafficLightArray), joins roi<->signal by `traffic_light_id` per stamp, maps
  `TrafficLightElement` enums to canonical tokens (validated:
  RED+CIRCLE / GREEN+UP_ARROW -> `green-arrow-up,red-circle`; ped -> `*-ped`;
  SOLID_OFF -> dropped), aligns each stamp to the nearest camera keyframe within
  `--time-tol-ms` (75). Needs a sourced ROS 2 humble + the autoware workspace.
- match by box IoU (detector-driven ROI ids are not map RE ids), so levels 1 & 2
  need no map/localization. Level 3 = driving_log_replayer_v2.

## Two harness modes

- **detector-driven** (current `ros2_pipeline/` harness): detector finds ROIs →
  classifier. Matches our offline pipeline; enough for levels 1 & 2. No map /
  localization needed. Also the existing `parity_check.py` (node vs offline).
- **map-driven / production** (needed for level 3): the full traffic_light graph
  — `map_based_traffic_light_detector` projects map ROIs → fine detector →
  classifier → `traffic_light_multi_camera_fusion` → `traffic_light_arbiter`,
  producing per-RE signals. Needs the lanelet2 map + localization/TF from the
  input bag. Bigger setup.

## Decoupling detector from classifier (level 2)

The Autoware classifier takes `rois` as input, so we can run it two ways:
- **end-to-end**: classify the detector's own ROIs → deployed-performance number
  (detection errors propagate).
- **classifier-isolated**: feed **GT ROIs** as `rois` → pure classifier accuracy,
  independent of detector recall. This is the clean way to separate "the
  detector missed it" from "the classifier got the color wrong".

## What exists vs what's new

Reuse:
- `ros2_pipeline/` feeder + collector (node output → `tlr_autolabel/v1`) and
  `parity_check.py` (node vs offline boxes/states).
- `match_traffic_lights.py` (node run → Tier B, map association) and
  `evaluate_signals.py` ledgers/profiles.

New:
- **two-source evaluator** `eval_vs_gt.py` (implemented 2026-07-22): prediction
  run × GT run → detection P/R/IoU by distance (level 1), state accuracy +
  element P/R + confusion (level 2); RE level is the remaining extension. Falls
  back to non-rejected-as-GT with a warning when GT is unreviewed (machinery
  test). Validated self-test S960(pred) vs L1920(gt).
- **GT-ROI feeder mode**: publish GT boxes as `rois` for classifier-isolated eval.
- **full-graph harness** (level 3): launch the production traffic_light pipeline
  on the bag with map + localization, collect the per-RE `TrafficSignalArray`.

## Time / frame alignment

GT is at keyframes (`sample_data_token`, with timestamps). The node is stamped;
the feeder encodes frame index in `header.stamp.sec` (detector-driven), or for
the bag-driven run we align node output to the nearest GT keyframe timestamp.

## Dependencies / open questions

- **GT must be human-reviewed first** — until the CVAT round + the time-series
  RE tool produce reviewed labels, eval runs against the *provisional* GT
  (self-consistency / node-vs-offline parity only, not accuracy).
- Level 3 needs the production launch + map + localization from the bag —
  confirm which launch and that the bag carries TF/pose.
- Pedestrian vs vehicle classifier split, and `unknown`/occluded handling in the
  denominators, follow the visibility spec (occluded excluded from state metrics).

## Phasing

1. **Phase 1 (now, provisional GT)**: detector-driven harness → node Tier B →
   `eval_vs_gt.py` for detection P/R + classification accuracy. Validate the
   machinery against provisional GT and existing parity_check.
2. **Phase 2**: classifier-isolated (GT-ROI feeder) → pure classification
   accuracy + confusion matrix.
3. **Phase 3**: full production graph → per-RE evaluation over time, once RE GT
   exists from the time-series tool.
