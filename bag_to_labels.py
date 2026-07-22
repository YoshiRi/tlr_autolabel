#!/usr/bin/env python3
"""Read a rosbag of the live TLR node output and emit tlr_autolabel/v1 labels,
time-aligned to a T4 dataset's camera keyframes, so the node can be scored
against our GT with the same L3/L6 tools (see docs/eval_design.md).

Consumes (per camera, from the autoware_ml_model_launchers detect+classify graph):
  <ns>/detection/rois            tier4_perception_msgs/TrafficLightRoiArray
  <ns>/classification/traffic_signals  tier4_perception_msgs/TrafficLightArray
joins roi<->signal by traffic_light_id within a stamp (box from the roi, state
from the signal's elements), maps elements to our canonical state tokens, aligns
each message stamp to the nearest camera sample_data keyframe (±--time-tol),
and writes one tlr_autolabel/v1 JSON per matched frame.

Then:  match_traffic_lights.py --autolabel-dir <out> ...  ->  node Tier B
       eval_vs_gt.py --pred <node Tier B> --gt <reviewed GT> ...

Needs a sourced ROS 2 (humble) + the autoware workspace (tier4_perception_msgs).
Run e.g.:  source /opt/ros/humble/setup.bash && source <ws>/install/setup.bash
           python3 bag_to_labels.py --bag <bag> --dataset-root <ds> --out-dir <out>
"""
import argparse
import glob
import json
import os
from bisect import bisect_left
from collections import defaultdict

# ---- tier4_perception_msgs/TrafficLightElement enum -> canonical tokens ----
# color 1=RED 2=AMBER 3=GREEN 4=WHITE ; shape 5=CIRCLE 6..13=arrows 14=CROSS ...
COLOR = {1: "red", 2: "amber", 3: "green", 4: "white"}
SHAPE = {5: "circle", 14: "cross", 15: "circle"}   # 15 SOLID_OFF treated as circle-off upstream
ARROW = {6: "left", 7: "right", 8: "up", 9: "up_left", 10: "up_right",
         11: "down", 12: "down_left", 13: "down_right"}
STATUS_OFF = 15  # SOLID_OFF: lamp not lit


def element_token(color, shape, status, ped=False):
    """One TrafficLightElement -> a canonical lamp token, or None if it carries
    no state (off / unknown)."""
    if status == STATUS_OFF:
        return None
    c = COLOR.get(color)
    if c is None:
        return None
    if ped:
        return f"{c}-ped"
    if shape in ARROW:
        return f"{c}-arrow-{ARROW[shape]}"
    if shape == 14:
        return f"{c}-cross"
    if shape in (5, 15) or shape is None:
        return f"{c}-circle"
    return f"{c}-circle"


def signal_state(elements, ped=False):
    """A TrafficLight's elements -> canonical state string (sorted, comma-joined)."""
    toks = [t for e in elements
            if (t := element_token(e["color"], e["shape"], e["status"], ped))]
    return ",".join(sorted(set(toks))) if toks else "unknown"


# ------------------------------------------------------------- T4 keyframes


def camera_keyframes(dataset_root):
    """[(timestamp_us, sample_data_token, channel, filename, w, h)] sorted by ts."""
    ann = os.path.join(dataset_root, "annotation")
    calib = {r["token"]: r for r in json.load(open(f"{ann}/calibrated_sensor.json"))}
    sensor = {r["token"]: r for r in json.load(open(f"{ann}/sensor.json"))}
    rows = []
    for sd in json.load(open(f"{ann}/sample_data.json")):
        cal = calib[sd["calibrated_sensor_token"]]
        sen = sensor[cal["sensor_token"]]
        if sen["modality"] != "camera":
            continue
        rows.append((sd["timestamp"], sd["token"], sen["channel"],
                     sd["filename"], sd.get("width"), sd.get("height")))
    rows.sort()
    return rows


def nearest_keyframe(ts_us, channel, kf_by_channel):
    rows = kf_by_channel.get(channel)
    if not rows:
        return None
    ts = [r[0] for r in rows]
    i = bisect_left(ts, ts_us)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(rows):
            d = abs(rows[j][0] - ts_us)
            if best is None or d < best[0]:
                best = (d, rows[j])
    return best  # (delta_us, row)


# ------------------------------------------------------------- bag reading


def read_bag(bag_path, topics):
    """Yield (topic, stamp_ns, deserialized_msg). Requires sourced ROS 2 + msgs."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_id = "mcap" if bag_path.endswith(".mcap") else "sqlite3"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
                rosbag2_py.ConverterOptions("", ""))
    typemap = {t.name: t.type for t in reader.get_all_topics_and_types()}
    want = set(topics)
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic in want:
            yield topic, t_ns, deserialize_message(data, get_message(typemap[topic]))


def channel_of_topic(topic):
    """.../<channel>/detection/rois or .../<channel>/classification/... -> channel."""
    parts = topic.strip("/").split("/")
    for kw in ("detection", "classification"):
        if kw in parts:
            i = parts.index(kw)
            if i > 0:
                return parts[i - 1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rois-topic", action="append", default=[],
                    help="TrafficLightRoiArray topic(s); default: any *detection/rois")
    ap.add_argument("--signals-topic", action="append", default=[],
                    help="TrafficLightArray topic(s); default: any *classification/traffic_signals")
    ap.add_argument("--time-tol-ms", type=float, default=75.0,
                    help="max |stamp - keyframe| to align (driving_log_replayer uses 75ms)")
    args = ap.parse_args()

    kf = camera_keyframes(args.dataset_root)
    kf_by_channel = defaultdict(list)
    for r in kf:
        kf_by_channel[r[2]].append(r)

    # discover topics if not given
    import rosbag2_py
    storage_id = "mcap" if args.bag.endswith(".mcap") else "sqlite3"
    probe = rosbag2_py.SequentialReader()
    probe.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id=storage_id),
               rosbag2_py.ConverterOptions("", ""))
    all_topics = {t.name: t.type for t in probe.get_all_topics_and_types()}
    del probe
    rois_topics = args.rois_topic or [n for n, ty in all_topics.items()
                                      if ty.endswith("TrafficLightRoiArray")]
    sig_topics = args.signals_topic or [n for n, ty in all_topics.items()
                                        if ty.endswith("TrafficLightArray")]
    print(f"rois topics: {rois_topics}\nsignal topics: {sig_topics}")

    # buffer rois and signals by (channel, stamp_ns)
    rois = {}     # (channel, stamp) -> {tl_id: (roi, type)}
    sigs = {}     # (channel, stamp) -> {tl_id: (elements, type)}
    for topic, t_ns, msg in read_bag(args.bag, rois_topics + sig_topics):
        ch = channel_of_topic(topic)
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if topic in rois_topics:
            rois.setdefault((ch, stamp), {})
            for r in msg.rois:
                rois[(ch, stamp)][r.traffic_light_id] = (r.roi, r.traffic_light_type)
        else:
            sigs.setdefault((ch, stamp), {})
            for s in msg.signals:
                sigs[(ch, stamp)][s.traffic_light_id] = (s.elements, s.traffic_light_type)

    os.makedirs(args.out_dir, exist_ok=True)
    tol_us = args.time_tol_ms * 1000
    n_written = n_unaligned = 0
    for (ch, stamp), roi_map in sorted(rois.items()):
        near = nearest_keyframe(stamp // 1000, ch or "", kf_by_channel)
        if near is None or near[0] > tol_us:
            n_unaligned += 1
            continue
        _, (_ts, sd_token, channel, filename, w, h) = near
        sig_map = sigs.get((ch, stamp), {})
        signals = []
        for tl_id, (roi, tl_type) in roi_map.items():
            ped = (tl_type == 1)
            state = "unknown"
            if tl_id in sig_map:
                elems = [{"color": e.color, "shape": e.shape, "status": e.status}
                         for e in sig_map[tl_id][0]]
                state = signal_state(elems, ped)
            signals.append({
                "detector_score": None,   # not carried on TrafficLightRoi
                "box_xyxy": [int(roi.x_offset), int(roi.y_offset),
                             int(roi.x_offset + roi.width), int(roi.y_offset + roi.height)],
                "lamps": [], "state": state,
            })
        name = os.path.splitext(os.path.basename(filename))[0]
        out = {"schema_version": "tlr_autolabel/v1",
               "image": filename, "image_realpath": os.path.join(args.dataset_root, filename),
               "sample_data_token": sd_token, "channel": channel,
               "frame_index": int(name) if name.isdigit() else 0,
               "width": w, "height": h,
               "meta": {"source": "ros_node_bag", "bag": os.path.basename(args.bag),
                        "align_delta_us": near[0]},
               "signals": signals}
        json.dump(out, open(os.path.join(args.out_dir, f"{channel}_{name}.json"), "w"),
                  indent=2, ensure_ascii=False)
        n_written += 1
    print(f"wrote {n_written} frames to {args.out_dir} ({n_unaligned} bag frames unaligned)")


if __name__ == "__main__":
    main()
