#!/usr/bin/env python3
"""Collect the live TLR graph's output into per-frame JSON autolabels.

Emits the SAME `tlr_autolabel/v1` schema as the offline pipeline
(../tlr_autolabel.py) so the two can be compared frame-by-frame for L5 parity
(see parity_check.py). Subscribes the ROI array (boxes) and the car/pedestrian
traffic-signal arrays (states), buffers per frame (keyed by header.stamp.sec,
which the feeder set to the frame index), joins ROI<->signal by traffic_light_id,
and writes one JSON per frame named after the source image (via frame_map.json).

Subscribing to the classifier outputs is REQUIRED: the classifier node runs
lazily only when its output topic has a subscriber.

Note: the ROS path has no per-ROI detector confidence (the detector score is not
carried on TrafficLightRoi), so `detector_score` is null here — a known, expected
difference from the offline output; parity on boxes uses IoU, not score.
"""
import argparse
import json
import os
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tier4_perception_msgs.msg import TrafficLightArray, TrafficLightRoiArray
from tier4_perception_msgs.msg import TrafficLightElement as TE

SCHEMA_VERSION = "tlr_autolabel/v1"

COLOR = {TE.RED: "red", TE.AMBER: "amber", TE.GREEN: "green", TE.WHITE: "white",
         TE.UNKNOWN: "unknown"}
# ROS folds the arrow direction into `shape`; the v1 schema keeps shape="arrow"
# plus a separate `arrow` direction, so split it here to match the offline output.
_ARROW = {TE.LEFT_ARROW: "left", TE.RIGHT_ARROW: "right", TE.UP_ARROW: "up",
          TE.UP_LEFT_ARROW: "up_left", TE.UP_RIGHT_ARROW: "up_right",
          TE.DOWN_ARROW: "down", TE.DOWN_LEFT_ARROW: "down_left",
          TE.DOWN_RIGHT_ARROW: "down_right"}
_SHAPE = {TE.CIRCLE: "circle", TE.CROSS: "cross", TE.UNKNOWN: "unknown"}
TYPE = {0: "car", 1: "pedestrian"}


def elem_to_lamp(e):
    """One TrafficLightElement -> a v1 lamp dict {label,color,shape,arrow,confidence}."""
    color = COLOR.get(e.color, "unknown")
    if e.shape in _ARROW:
        shape, arrow = "arrow", _ARROW[e.shape]
        label = f"{color}-arrow({arrow})"
    else:
        shape, arrow = _SHAPE.get(e.shape, "unknown"), None
        label = f"{color}-{shape}"
    return {"label": label, "color": color, "shape": shape, "arrow": arrow,
            "confidence": round(float(e.confidence), 4)}


class Collector(Node):
    def __init__(self, args):
        super().__init__("tlr_label_collector")
        self.args = args
        self.image_root = os.path.expanduser(args.image_root) if args.image_root else None
        self.frame_map = {}
        if os.path.exists(args.frame_map):
            self.frame_map = json.load(open(args.frame_map))
        os.makedirs(args.out_dir, exist_ok=True)
        self.dims_cache = {}
        self.meta = {
            "run_id": args.run_id,
            "source": "ros2_pipeline",
            "detector": args.detector,
            "classifier": args.classifier,
            "note": "live Autoware graph (TensorRT int8); detector_score unavailable on ROS path",
        }
        self.lock = threading.Lock()
        self.buf = {}           # sec -> slot
        self.written = set()
        self.n_written = 0
        qos = QoSProfile(depth=10)
        self.create_subscription(TrafficLightRoiArray, args.rois_topic, self.on_rois, qos)
        self.create_subscription(TrafficLightArray, args.car_topic, self.on_sig, qos)
        self.create_subscription(TrafficLightArray, args.ped_topic, self.on_sig, qos)
        self.create_timer(1.0, self.flush_stale)
        self.get_logger().info(f"collecting -> {args.out_dir} (schema {SCHEMA_VERSION})")

    def _slot(self, sec):
        return self.buf.setdefault(
            sec, {"rois": {}, "signals": {}, "types": {}, "t": time.time(), "got_rois": False})

    def on_rois(self, msg: TrafficLightRoiArray):
        sec = msg.header.stamp.sec
        with self.lock:
            s = self._slot(sec); s["t"] = time.time()
            for r in msg.rois:
                s["rois"][r.traffic_light_id] = [
                    int(r.roi.x_offset), int(r.roi.y_offset),
                    int(r.roi.x_offset + r.roi.width), int(r.roi.y_offset + r.roi.height)]
                s["types"][r.traffic_light_id] = TYPE.get(r.traffic_light_type, "unknown")
            s["got_rois"] = True
            self._maybe_write(sec)

    def on_sig(self, msg: TrafficLightArray):
        sec = msg.header.stamp.sec
        with self.lock:
            s = self._slot(sec); s["t"] = time.time()
            for sig in msg.signals:
                s["signals"][sig.traffic_light_id] = [elem_to_lamp(e) for e in sig.elements]
            self._maybe_write(sec)

    def _maybe_write(self, sec):
        s = self.buf.get(sec)
        if not s or not s["got_rois"] or sec in self.written:
            return
        if s["rois"] and any(i not in s["signals"] for i in s["rois"]):
            return  # still waiting for some signals
        self._write(sec)

    def flush_stale(self):
        now = time.time()
        with self.lock:
            for sec in [k for k, v in self.buf.items()
                        if k not in self.written and now - v["t"] > self.args.timeout]:
                self._write(sec, stale=True)

    def _dims(self, path):
        if path in self.dims_cache:
            return self.dims_cache[path]
        wh = (None, None)
        if path and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                wh = (img.shape[1], img.shape[0])
        self.dims_cache[path] = wh
        return wh

    def _resolve_path(self, sec):
        path = self.frame_map.get(str(sec))
        if path is None and os.path.exists(self.args.frame_map):
            try:
                self.frame_map = json.load(open(self.args.frame_map))
                path = self.frame_map.get(str(sec))
            except Exception:
                pass
        return path

    def _write(self, sec, stale=False):
        s = self.buf[sec]
        self.written.add(sec)
        path = self._resolve_path(sec)
        name = os.path.splitext(os.path.basename(path))[0] if path else f"frame_{sec:06d}"
        if path and self.image_root:
            rel = os.path.relpath(path, self.image_root)
        else:
            rel = os.path.basename(path) if path else None
        channel = None
        if path:
            channel = next((c for c in os.path.dirname(path).split(os.sep)
                            if c and c.upper() == c and not c.startswith(".")), None)
        frame_index = int(name) if name.isdigit() else sec
        w, h = self._dims(path)

        signals = []
        for i, (tid, box) in enumerate(sorted(s["rois"].items())):
            lamps = s["signals"].get(tid, [])
            named = [l for l in lamps if not (l["color"] == "unknown" and l["shape"] == "unknown")]
            signals.append({
                "signal_id": f"{name}-{i:02d}",
                "traffic_light_id": tid,
                "type": s["types"].get(tid, "unknown"),
                "detector_score": None,   # not available on the ROS path
                "box_xyxy": box,
                "lamps": lamps,
                "signal": ",".join(sorted(l["label"] for l in named)) if named else "unknown",
            })
        out = {
            "schema_version": SCHEMA_VERSION,
            "image": rel,
            "image_realpath": path,
            "sample_data_token": None,
            "channel": channel,
            "frame_index": frame_index,
            "width": w, "height": h,
            "meta": self.meta,
            "signals": signals,
        }
        if stale and not s["got_rois"]:
            out["note"] = "no roi message received (flushed on timeout)"
        with open(os.path.join(self.args.out_dir, name + ".json"), "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        self.n_written += 1
        if self.n_written % 50 == 0:
            self.get_logger().info(f"wrote {self.n_written} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frame-map", default="/tmp/tlr_frame_map.json")
    ap.add_argument("--rois-topic", required=True)
    ap.add_argument("--car-topic", required=True)
    ap.add_argument("--ped-topic", required=True)
    ap.add_argument("--image-root", default=None,
                    help="make `image` relative to this root (match the offline run's --image-root)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--detector", default="live-ros2-graph")
    ap.add_argument("--classifier", default="traffic_light_lamp_recognizer_comlops")
    ap.add_argument("--timeout", type=float, default=3.0,
                    help="seconds after last update to flush an incomplete frame")
    args = ap.parse_args()
    rclpy.init()
    node = Collector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"total frames written: {node.n_written}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
