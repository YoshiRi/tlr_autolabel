#!/usr/bin/env python3
"""Collect the live TLR graph's output into per-frame JSON autolabels.

Subscribes the ROI array (boxes) and the car/pedestrian traffic-signal arrays
(states), buffers them per frame (keyed by header.stamp.sec, which the feeder set
to the frame index), joins ROI<->signal by traffic_light_id, and writes one JSON
per frame named after the source image (via the feeder's frame_map.json).

Subscribing to the classifier outputs is REQUIRED: the classifier node runs lazily
only when its output topic has a subscriber.

Usage (after sourcing the workspace):
  python3 tlr_label_collector.py --out-dir <dir> --frame-map /tmp/tlr_frame_map.json \
    --rois-topic  /perception/traffic_light_recognition/camera6/detection/rois \
    --car-topic   /perception/traffic_light_recognition/camera6/classification/car/traffic_signals \
    --ped-topic   /perception/traffic_light_recognition/camera6/classification/pedestrian/traffic_signals
"""
import argparse
import json
import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tier4_perception_msgs.msg import TrafficLightArray, TrafficLightRoiArray
from tier4_perception_msgs.msg import TrafficLightElement as TE

COLOR = {TE.RED: "red", TE.AMBER: "amber", TE.GREEN: "green", TE.WHITE: "white",
         TE.UNKNOWN: "unknown"}
SHAPE = {TE.CIRCLE: "circle", TE.LEFT_ARROW: "left", TE.RIGHT_ARROW: "right",
         TE.UP_ARROW: "up", TE.UP_LEFT_ARROW: "up_left", TE.UP_RIGHT_ARROW: "up_right",
         TE.DOWN_ARROW: "down", TE.DOWN_LEFT_ARROW: "down_left",
         TE.DOWN_RIGHT_ARROW: "down_right", TE.CROSS: "cross", TE.UNKNOWN: "unknown"}
TYPE = {0: "car", 1: "pedestrian"}


def elem_to_dict(e):
    return {"color": COLOR.get(e.color, str(e.color)),
            "shape": SHAPE.get(e.shape, str(e.shape)),
            "confidence": round(float(e.confidence), 4)}


class Collector(Node):
    def __init__(self, args):
        super().__init__("tlr_label_collector")
        self.args = args
        self.frame_map = {}
        if os.path.exists(args.frame_map):
            self.frame_map = json.load(open(args.frame_map))
        os.makedirs(args.out_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.buf = {}           # sec -> {"rois":{id:roi}, "signals":{id:[elem...]}, "types":{id:type}, "t":last_update}
        self.written = set()
        self.n_written = 0
        qos = QoSProfile(depth=10)
        self.create_subscription(TrafficLightRoiArray, args.rois_topic, self.on_rois, qos)
        self.create_subscription(TrafficLightArray, args.car_topic, self.on_sig, qos)
        self.create_subscription(TrafficLightArray, args.ped_topic, self.on_sig, qos)
        self.create_timer(1.0, self.flush_stale)
        self.get_logger().info(f"collecting -> {args.out_dir}")

    def _slot(self, sec):
        return self.buf.setdefault(sec, {"rois": {}, "signals": {}, "types": {}, "t": time.time()})

    def on_rois(self, msg: TrafficLightRoiArray):
        sec = msg.header.stamp.sec
        with self.lock:
            s = self._slot(sec); s["t"] = time.time()
            for r in msg.rois:
                s["rois"][r.traffic_light_id] = [int(r.roi.x_offset), int(r.roi.y_offset),
                                                 int(r.roi.x_offset + r.roi.width),
                                                 int(r.roi.y_offset + r.roi.height)]
                s["types"][r.traffic_light_id] = TYPE.get(r.traffic_light_type, str(r.traffic_light_type))
            s["got_rois"] = True
            self._maybe_write(sec)

    def on_sig(self, msg: TrafficLightArray):
        sec = msg.header.stamp.sec
        with self.lock:
            s = self._slot(sec); s["t"] = time.time()
            for sig in msg.signals:
                s["signals"][sig.traffic_light_id] = [elem_to_dict(e) for e in sig.elements]
            self._maybe_write(sec)

    def _maybe_write(self, sec):
        # write once ROIs are in and every ROI id has a matching signal (or no ROIs at all)
        s = self.buf.get(sec)
        if not s or not s.get("got_rois") or sec in self.written:
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

    def _write(self, sec, stale=False):
        s = self.buf[sec]
        self.written.add(sec)
        path = self.frame_map.get(str(sec))
        if path is None and os.path.exists(self.args.frame_map):
            # feeder may have (re)written the map after we started; reload once
            try:
                self.frame_map = json.load(open(self.args.frame_map))
                path = self.frame_map.get(str(sec))
            except Exception:
                pass
        name = os.path.splitext(os.path.basename(path))[0] if path else f"frame_{sec:06d}"
        signals = []
        for tid, box in s["rois"].items():
            lamps = s["signals"].get(tid, [])
            named = [l for l in lamps if l["color"] != "unknown" or l["shape"] != "unknown"]
            signals.append({
                "traffic_light_id": tid,
                "type": s["types"].get(tid, "unknown"),
                "box_xyxy": box,
                "lamps": lamps,
                "signal": ",".join(f"{l['color']}-{l['shape']}" for l in named) if named else "unknown",
            })
        out = {"image": path, "frame_sec": sec, "num_signals": len(signals), "signals": signals}
        if stale and not s.get("got_rois"):
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
