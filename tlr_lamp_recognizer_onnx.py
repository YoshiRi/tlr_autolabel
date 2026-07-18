#!/usr/bin/env python3
"""[DEBUG standalone + shared decode module] TLR YOLOX-based classifier (CnnLampRecognizer).

tlr_autolabel.py imports the decode/preprocess from here (this file is the
canonical LampRecognizer decode); the __main__ CLI is a single-crop debug tool.

Faithfully reproduces the decode of Autoware's `autoware_traffic_light_classifier`
node, classifier_type=2 (LampRecognizer):
  src/universe/.../autoware_traffic_light_classifier/src/classifier/cnn_lamp_recognizer.cpp

Model  : traffic_light_lamp_recognizer_comlops.onnx
  input : (N, 3, 256, 256), preprocessing = blobFromImages(scale=1/255, swapRB=false)
  output: (N, 48, 64, 64) == (N, num_anchors*chans_per_anchor, grid_h, grid_w)

The node operates on already-cropped traffic-light ROIs (one crop per signal).
This script runs the same inference on a single crop (or a directory of crops)
for autolabeling / debugging, printing per-lamp color + shape + arrow direction.

Usage:
  python3 tlr_lamp_recognizer_onnx.py <image_or_dir> \
      [--model PATH.onnx] [--param lamp_recognizer_ml.param.yaml] \
      [--score-thr 0.2] [--nms-thr 0.2]
"""
import argparse
import glob
import math
import os

import cv2
import numpy as np
import onnxruntime as ort
import yaml

# --- enum maps (from cnn_lamp_recognizer.hpp) ---
COLORS = {0: "green", 1: "amber", 2: "red", 3: "unknown"}          # color_start argmax
SHAPES = {0: "circle", 1: "arrow", 2: "u_turn", 3: "ped",         # type_start argmax
          4: "number", 5: "cross", 6: "unknown"}
ARROWS = ["up", "up_right", "right", "down_right",
          "down", "down_left", "left", "up_left"]


def angle_to_arrow(angle_rad: float) -> str:
    """8-way bucketing identical to angleToArrowDirection() in the node."""
    pi = math.pi
    if -pi / 8 <= angle_rad < pi / 8:
        return "up"
    if pi / 8 <= angle_rad < 3 * pi / 8:
        return "up_right"
    if 3 * pi / 8 <= angle_rad < 5 * pi / 8:
        return "right"
    if 5 * pi / 8 <= angle_rad < 7 * pi / 8:
        return "down_right"
    if angle_rad >= 7 * pi / 8 or angle_rad < -7 * pi / 8:
        return "down"
    if angle_rad < -5 * pi / 8:
        return "down_left"
    if angle_rad < -3 * pi / 8:
        return "left"
    return "up_left"


def iou(a, b) -> float:
    def overlap_1d(a0, a1, b0, b1):
        if a0 > b0:
            a0, b0 = b0, a0
            a1, b1 = b1, a1
        return 0.0 if a1 < b0 else min(a1, b1) - b0
    ox = overlap_1d(a["x1"], a["x2"], b["x1"], b["x2"])
    oy = overlap_1d(a["y1"], a["y2"], b["y1"], b["y2"])
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    inter = ox * oy
    u = area_a + area_b - inter
    return 0.0 if u == 0 else inter / u


def nms(dets, iou_thr):
    """stable sort by prob desc, greedy suppress (matches runNms())."""
    dets = sorted(dets, key=lambda d: -d["prob"])
    keep = []
    for d in dets:
        if all(iou(d, k) <= iou_thr for k in keep):
            keep.append(d)
    return keep


def load_model_params(param_path):
    with open(param_path) as f:
        y = yaml.safe_load(f)
    mp = y["/**"]["ros__parameters"]["model_params"]
    mp["bbox_offset"] = 0.5 * (mp["scale_x_y"] - 1.0)
    assert len(mp["anchors"]) == 2 * mp["num_anchors"], "anchors must be 2*num_anchors"
    return mp


def preprocess(img, input_w, input_h):
    # The node feeds RGB: it converts the ROS image with cv_bridge toCvCopy(RGB8)
    # then calls blobFromImages(scale=1/255, swapRB=False) — i.e. RGB reaches the
    # model. cv2.imread gives BGR, so we set swapRB=True to feed RGB (verified:
    # a green arrow scores green=1.0 with RGB vs amber=0.999 with BGR).
    blob = cv2.dnn.blobFromImage(
        img, 1.0 / 255.0, (input_w, input_h), (0, 0, 0), swapRB=True, crop=False
    )
    return blob.astype(np.float32)  # (1, 3, H, W)


def decode(out, mp, input_w, input_h, score_thr):
    """out: (out_c, grid_h, grid_w). Reproduces decodeTlrOutput()."""
    out_c, gh, gw = out.shape
    cpa = mp["chans_per_anchor"]
    dets = []
    for a in range(mp["num_anchors"]):
        base = a * cpa  # channel offset for this anchor
        pw = mp["anchors"][a * 2]
        ph = mp["anchors"][a * 2 + 1]
        obj = out[base + mp["obj_index"]]                       # (gh, gw)
        mask = obj >= score_thr
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            objectness = float(obj[y, x])
            tx = out[base + mp["x_index"], y, x]
            ty = out[base + mp["y_index"], y, x]
            tw = out[base + mp["w_index"], y, x]
            th = out[base + mp["h_index"], y, x]
            bx = x + mp["scale_x_y"] * tx - mp["bbox_offset"]
            by = y + mp["scale_x_y"] * ty - mp["bbox_offset"]
            bw = pw * (tw * 2.0) ** 2
            bh = ph * (th * 2.0) ** 2

            types = out[base + mp["type_start"]: base + mp["type_start"] + mp["num_types"], y, x]
            type_idx = int(np.argmax(types))
            max_type_prob = float(types[type_idx])
            colors = out[base + mp["color_start"]: base + mp["color_start"] + mp["num_colors"], y, x]
            color_idx = int(np.argmax(colors))

            score = objectness * max_type_prob
            if score < score_thr:
                continue

            cos_v = float(out[base + mp["cos_index"], y, x])
            sin_v = float(out[base + mp["sin_index"], y, x])

            cx, cy = bx / gw, by / gh
            hw = (bw * 0.5) / input_w
            hh = (bh * 0.5) / input_h
            det = {
                "x1": min(1.0, max(0.0, cx - hw)),
                "y1": min(1.0, max(0.0, cy - hh)),
                "x2": min(1.0, max(0.0, cx + hw)),
                "y2": min(1.0, max(0.0, cy + hh)),
                "prob": score,
                "color": color_idx,
                "shape": type_idx,
                "sin": sin_v,
                "cos": cos_v,
            }
            dets.append(det)
    return dets


def arrow_of(det):
    if SHAPES.get(det["shape"]) != "arrow":
        return None
    norm_sq = det["sin"] ** 2 + det["cos"] ** 2
    if norm_sq < 1e-12:
        return "unknown"
    return angle_to_arrow(math.atan2(det["sin"], det["cos"]))


def run_one(sess, in_name, out_name, img_path, mp, input_w, input_h, score_thr, nms_thr):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[skip] cannot read {img_path}")
        return
    blob = preprocess(img, input_w, input_h)
    out = sess.run([out_name], {in_name: blob})[0][0]  # (out_c, gh, gw)
    dets = decode(out, mp, input_w, input_h, score_thr)
    kept = nms(dets, nms_thr)

    print(f"\n=== {os.path.basename(img_path)}  ({img.shape[1]}x{img.shape[0]}) : "
          f"{len(kept)} lamp(s) ===")
    for d in sorted(kept, key=lambda d: -d["prob"]):
        label = f"{COLORS.get(d['color'])}-{SHAPES.get(d['shape'])}"
        arrow = arrow_of(d)
        if arrow:
            label += f"({arrow})"
        print(f"  {label:24s} score={d['prob']:.3f} "
              f"box=({d['x1']:.3f},{d['y1']:.3f})-({d['x2']:.3f},{d['y2']:.3f})")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="image file or directory of crops")
    ap.add_argument("--model", default=os.path.join(here, "traffic_light_lamp_recognizer_comlops.onnx"))
    ap.add_argument("--param", default=os.path.join(here, "lamp_recognizer_ml.param.yaml"))
    ap.add_argument("--score-thr", type=float, default=0.2)
    ap.add_argument("--nms-thr", type=float, default=0.2)
    args = ap.parse_args()

    mp = load_model_params(args.param)
    opts = ort.SessionOptions()
    opts.log_severity_level = 4  # silence initializer-in-input warnings
    sess = ort.InferenceSession(args.model, sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_info = sess.get_outputs()[0]
    out_name = out_info.name
    # input dims (N,3,H,W)
    _, _, input_h, input_w = [d if isinstance(d, int) else 256 for d in sess.get_inputs()[0].shape]

    print(f"model={os.path.basename(args.model)} input={input_w}x{input_h} "
          f"output={out_info.shape} anchors={mp['num_anchors']} "
          f"score_thr={args.score_thr} nms_thr={args.nms_thr}")

    if os.path.isdir(args.image):
        paths = sorted(sum([glob.glob(os.path.join(args.image, e))
                            for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
    else:
        paths = [args.image]
    for p in paths:
        run_one(sess, in_name, out_name, p, mp, input_w, input_h, args.score_thr, args.nms_thr)


if __name__ == "__main__":
    main()
