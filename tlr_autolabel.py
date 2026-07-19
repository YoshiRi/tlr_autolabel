#!/usr/bin/env python3
"""Combined TLR autolabeling: detector -> per-ROI YOLOX classifier (LampRecognizer).

full image ─> detector (find traffic_light boxes, NMS)
           ─> for each box: crop ─> lamp recognizer (color + shape + arrow)
           ─> annotated image (--viz) + JSON labels (--out-dir)

Two detector families are supported; the type is auto-detected from the ONNX
output count:
  1 output  -> YOLOX traffic_light detector (tensorrt_yolox style: BGR, no /255,
               grid/stride decode, num_class=1)
  3 outputs -> CoMLOps-Large-Detection-Model (darknet YOLO style: RGB, /255,
               5 anchors x (4 box + obj + 10 cls) per scale, strides 8/16/32,
               bw = pw*(2*sig(tw))^2; anchors from comlops_large_detector_ml.param.yaml,
               empirically fitted -- see that yaml's header).
               Kept classes default to TRAFFIC_LIGHT only (--det-classes).

Reuses the faithful classifier decode from tlr_lamp_recognizer_onnx.py.

Usage:
  # 1) verification pass on a few images, write annotated pngs to eyeball
  python3 tlr_autolabel.py <image_or_dir> --viz --out-dir ./out

  # 2) once inference looks right, run over the dataset for labels
  python3 tlr_autolabel.py <dir> --out-dir ./labels

  # detector selection: named preset (configs/detectors/*.yaml) or explicit path
  python3 tlr_autolabel.py <dir> --preset yolox-1920-int8
  python3 tlr_autolabel.py <dir> --detector .../CoMLOps-Large-Detection-Model-v1.0.1.onnx
"""
import argparse
import glob
import json
import math
import os
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import onnxruntime as ort

from tlr_lamp_recognizer_onnx import (
    COLORS, SHAPES, arrow_of, decode as classify_decode,
    load_model_params, nms as cls_nms, preprocess as cls_preprocess,
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------- detector -----------------------------
def make_session(model_path):
    opts = ort.SessionOptions()
    opts.log_severity_level = 4
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in ort.get_available_providers() \
        else ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def det_preprocess(img, w, h, rgb=False, norm=1.0):
    """Letterbox exactly like autoware_tensorrt_yolox:
    scale=min(W/w, H/h), resize (bilinear) keeping aspect ratio, paste top-left,
    pad bottom/right with 114.
    YOLOX: BGR (no swap), norm=1.0 (raw 0-255).
    CoMLOps darknet: rgb=True, norm=1/255 (verified: obj stays 0.000 on raw input
    while class channels still respond -- /255 is load-bearing)."""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    if (nw, nh) == (iw, ih):
        resized = img  # native-resolution tiles: skip the no-op resample
    else:
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if rgb:
        resized = resized[:, :, ::-1]
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    # ascontiguousarray after the transpose is load-bearing for speed: a
    # non-contiguous blob makes every downstream serialization (tofile to the
    # trt_run helper) do an element-wise gather (~400 ms vs ~26 ms for 9.4 MB).
    blob = np.ascontiguousarray(
        (canvas.astype(np.float32) * norm).transpose(2, 0, 1)[None, ...])  # NCHW
    return blob, scale


_grid_cache = {}


def det_grids(w, h):
    """(num_grids, 3) columns [gx, gy, stride] matching the yolox output order."""
    key = (w, h)
    if key not in _grid_cache:
        parts = []
        for s in (8, 16, 32):
            gy, gx = np.mgrid[0:h // s, 0:w // s]
            parts.append(np.stack(
                [gx.ravel(), gy.ravel(), np.full(gx.size, s)], axis=1))
        _grid_cache[key] = np.concatenate(parts).astype(np.float32)
    return _grid_cache[key]


def det_decode(out, w, h, score_thr):
    """out: (num_grids, 4 box + 1 obj + num_class) yolox head output.
    num_class is taken from the tensor shape (the TL detectors ship with 1
    class; a multi-class variant scores as obj * best-class, like the node).
    Returns boxes in the WxH network space. Vectorized: the per-cell python
    loop cost ~60 ms/pass, which dominates once inference itself is ~30 ms."""
    if out.ndim != 2 or out.shape[1] < 6:
        raise ValueError(
            f"unexpected yolox output shape {out.shape}; expected "
            "(num_grids, 4+1+num_class). If this is a new model family, "
            "extend Detector/det_decode instead of forcing it through here.")
    grids = det_grids(w, h)
    if grids.shape[0] != out.shape[0]:
        raise ValueError(
            f"grid count {grids.shape[0]} (strides 8/16/32 at {w}x{h}) does not "
            f"match output rows {out.shape[0]} — input size or stride set of "
            "this model differs from the tensorrt_yolox convention.")
    prob = out[:, 4] * out[:, 5:].max(axis=1)
    keep = prob > score_thr
    if not keep.any():
        return []
    o, g, p = out[keep], grids[keep], prob[keep]
    cx = (o[:, 0] + g[:, 0]) * g[:, 2]
    cy = (o[:, 1] + g[:, 1]) * g[:, 2]
    bw = np.exp(o[:, 2]) * g[:, 2]
    bh = np.exp(o[:, 3]) * g[:, 2]
    return [{"prob": float(p[i]),
             "x1": float(cx[i] - bw[i] / 2), "y1": float(cy[i] - bh[i] / 2),
             "x2": float(cx[i] + bw[i] / 2), "y2": float(cy[i] + bh[i] / 2)}
            for i in range(len(p))]


def comlops_load_params(param_path):
    with open(param_path) as f:
        import yaml
        y = yaml.safe_load(f)
    return y["/**"]["ros__parameters"]["model_params"]


def comlops_decode(outs, mp, score_thr, keep_class_ids):
    """outs: list of 3 arrays (num_anchors*chans, gh, gw), strides 8/16/32.
    Per-anchor channels: [tx, ty, tw, th, obj, cls0..cls9], all sigmoid in [0,1].
    Decode: cx=(gx+tx)*stride, bw=pw*(2*tw)^2 with per-(scale,anchor) pixel anchors.
    score = obj * cls; only keep_class_ids survive."""
    na = mp["num_anchors"]
    cpa = mp["chans_per_anchor"]
    ncls = mp["num_classes"]
    dets = []
    for si, (out, stride) in enumerate(zip(outs, mp["strides"])):
        anchors = mp["anchors"][si]  # flat [pw0, ph0, pw1, ph1, ...]
        r = out.reshape(na, cpa, out.shape[1], out.shape[2])
        obj = r[:, 4]                                    # (na, gh, gw)
        ays, gys, gxs = np.nonzero(obj >= score_thr)
        for a, gy, gx in zip(ays.tolist(), gys.tolist(), gxs.tolist()):
            cls_v = r[a, 5:5 + ncls, gy, gx]
            cls_id = int(np.argmax(cls_v))
            prob = float(obj[a, gy, gx] * cls_v[cls_id])
            if cls_id not in keep_class_ids or prob <= score_thr:
                continue
            cx = (gx + float(r[a, 0, gy, gx])) * stride
            cy = (gy + float(r[a, 1, gy, gx])) * stride
            bw = anchors[a * 2] * (2.0 * float(r[a, 2, gy, gx])) ** 2
            bh = anchors[a * 2 + 1] * (2.0 * float(r[a, 3, gy, gx])) ** 2
            dets.append({"prob": prob, "class_id": cls_id,
                         "x1": cx - bw / 2, "y1": cy - bh / 2,
                         "x2": cx + bw / 2, "y2": cy + bh / 2})
    return dets


class TrtServer:
    """Runs a TensorRT .engine through the trt_run helper (serve mode keeps the
    engine deserialized across images). Single input / single output engines
    only, i.e. the YOLOX detectors. Compiled from trt_run.cpp on first use."""

    def __init__(self, engine_path):
        import subprocess
        import tempfile
        binary = os.path.join(HERE, "trt_run")
        if not os.path.exists(binary):
            cuda = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            subprocess.check_call(
                ["g++", "-O2", os.path.join(HERE, "trt_run.cpp"), "-o", binary,
                 f"-I{cuda}/include", f"-L{cuda}/lib64",
                 "-lnvinfer", "-lcudart"])
        self.proc = subprocess.Popen([binary, engine_path, "serve"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     text=True)
        self.in_shape = self.out_shape = None
        for line in self.proc.stdout:
            t = line.split()
            if t and t[0] == "INPUT":
                self.in_shape = [int(x) for x in t[1:]]
            elif t and t[0] == "OUTPUT":
                self.out_shape = [int(x) for x in t[1:]]
            elif t and t[0] == "READY":
                break
        assert self.in_shape and self.out_shape, "trt_run did not report tensor shapes"
        self.tmp = tempfile.mkdtemp(prefix="trt_run_")

    def run(self, blob):
        ip = os.path.join(self.tmp, "in.f32")
        op = os.path.join(self.tmp, "out.f32")
        blob.astype(np.float32).tofile(ip)
        self.proc.stdin.write(f"{ip} {op}\n")
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline().strip()
        if resp != "DONE":
            raise RuntimeError(f"trt_run inference failed: {resp!r}")
        return np.fromfile(op, dtype=np.float32).reshape(self.out_shape)


class Detector:
    """Wraps either detector family behind detect(img, score_thr) -> (dets, scale).
    dets are in network (letterboxed) pixel space.

    Accepts .onnx (onnxruntime, CPU) or .engine (TensorRT via trt_run, GPU).
    Prefer the int8 .engine for the YOLOX detectors when available: the fp32
    ONNX fires high-confidence false positives on motion-blur/texture that the
    deployed int8 engine does not (verified on frame 00250: 4 FPs at fp32 vs 0
    at int8, with identical true positives on 00000)."""

    def __init__(self, model_path, comlops_param_path):
        if model_path.endswith(".engine"):
            self.sess = None
            self.trt = TrtServer(model_path)
            _, _, self.h, self.w = self.in_shape = self.trt.in_shape
            self.kind = "yolox"  # engine path implemented for yolox only
            return
        self.sess = make_session(model_path)
        self.in_name = self.sess.get_inputs()[0].name
        in_shape = self.sess.get_inputs()[0].shape
        if len(in_shape) != 4 or not all(isinstance(d, int) and d > 0 for d in in_shape[2:]):
            raise SystemExit(
                f"detector input shape {in_shape} is not a static NCHW image — "
                "dynamic-dim ONNX exports are not supported; re-export with a "
                "fixed input size or extend Detector.")
        _, _, self.h, self.w = in_shape
        n_out = len(self.sess.get_outputs())
        if n_out == 1:
            self.kind = "yolox"
        elif n_out == 3:
            self.kind = "comlops"
        else:
            raise SystemExit(
                f"unrecognized detector family: {n_out} outputs "
                f"({[o.name for o in self.sess.get_outputs()]}). Known: 1 output "
                "= yolox head, 3 outputs = CoMLOps darknet. New model families "
                "need a decode added to Detector.")
        if self.kind == "comlops":
            self.mp = comlops_load_params(comlops_param_path)
            self.keep_ids = set()
            self.labels = self.mp["labels"]

    def set_keep_classes(self, names):
        self.keep_ids = {self.labels.index(n) for n in names}

    def detect(self, img, score_thr):
        if self.kind == "yolox":
            blob, scale = det_preprocess(img, self.w, self.h)
            if self.sess is None:
                out = self.trt.run(blob)[0]
            else:
                out = self.sess.run(None, {self.in_name: blob})[0][0]
            return det_decode(out, self.w, self.h, score_thr), scale
        blob, scale = det_preprocess(img, self.w, self.h, rgb=True, norm=1.0 / 255.0)
        outs = [o[0] for o in self.sess.run(None, {self.in_name: blob})]
        return comlops_decode(outs, self.mp, score_thr, self.keep_ids), scale


def det_nms(dets, iou_thr, contain_thr=0.7):
    """Greedy NMS by score. Suppress a lower-score box if it overlaps a kept box
    by IoU>iou_thr, OR one box is mostly contained in the other
    (inter/min(area) > contain_thr) -- the latter merges nested duplicates (a
    tight lamp box and a whole-signal box around it, either nesting direction)
    that plain IoU-NMS leaves behind; the higher-score box always survives."""
    dets = sorted(dets, key=lambda d: -d["prob"])
    keep = []
    for d in dets:
        area_d = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
        drop = False
        for k in keep:
            ix = max(0.0, min(d["x2"], k["x2"]) - max(d["x1"], k["x1"]))
            iy = max(0.0, min(d["y2"], k["y2"]) - max(d["y1"], k["y1"]))
            inter = ix * iy
            area_k = (k["x2"] - k["x1"]) * (k["y2"] - k["y1"])
            ua = area_d + area_k - inter
            smaller = min(area_d, area_k)
            if (ua > 0 and inter / ua > iou_thr) or \
               (smaller > 0 and inter / smaller > contain_thr):
                drop = True
                break
        if not drop:
            keep.append(d)
    return keep


# ----------------------------- pipeline -----------------------------
def lamp_label(d):
    """Canonical lamp token: {color}-{shape}[-{direction}] (e.g. green-arrow-up,
    red-circle, red-ped). Direction only for arrow lamps."""
    label = f"{COLORS.get(d['color'])}-{SHAPES.get(d['shape'])}"
    arrow = arrow_of(d)
    if arrow:
        label += f"-{arrow}"
    return label


def signal_state(lamps):
    """Canonical signal state: lamp tokens sorted alphabetically, comma-joined;
    'unknown' when no lamp was recognized. Sorting makes the same physical
    state always serialize identically (confidence order does not)."""
    return ",".join(sorted(l["label"] for l in lamps)) if lamps else "unknown"


def detect_in_orig(detector, img, score_thr):
    """Run the detector on one image/crop and return dets in its pixel coords.
    Drops detections firing inside the 114 letterbox padding region (the
    detector can fire on the pad seam and clip to the border as 1px ghosts)."""
    ih, iw = img.shape[:2]
    boxes, scale = detector.detect(img, score_thr)
    valid_w, valid_h = iw * scale, ih * scale
    margin = 2.0
    out = []
    for b in boxes:
        if (b["x1"] + b["x2"]) * 0.5 > valid_w - margin or \
           (b["y1"] + b["y2"]) * 0.5 > valid_h - margin:
            continue
        b = dict(b)
        for k in ("x1", "y1", "x2", "y2"):
            b[k] = b[k] / scale
        out.append(b)
    return out


def tile_origins(size, net, min_overlap=128):
    """Top-left offsets of `net`-sized tiles covering `size` with at least
    `min_overlap` px overlap between neighbours. [0] when it already fits."""
    if size <= net:
        return [0]
    n = math.ceil((size - net) / (net - min_overlap)) + 1
    return [round(i * (size - net) / (n - 1)) for i in range(n)]


def detect_full_and_tiles(detector, img, args):
    """Full-frame letterboxed pass, plus (with --tiles) native-resolution tile
    passes, merged by one global NMS in original pixel coords. Tiles overlap by
    >= min_overlap px so a signal cut at one tile's edge is seen whole by the
    neighbour; the containment rule in det_nms then merges the clipped duplicate."""
    dets = detect_in_orig(detector, img, args.det_score_thr)
    if args.tiles:
        ih, iw = img.shape[:2]
        for oy in tile_origins(ih, detector.h, args.tile_overlap):
            for ox in tile_origins(iw, detector.w, args.tile_overlap):
                tile = img[oy:oy + detector.h, ox:ox + detector.w]
                for b in detect_in_orig(detector, tile, args.det_score_thr):
                    b["x1"] += ox; b["x2"] += ox
                    b["y1"] += oy; b["y2"] += oy
                    dets.append(b)
    return det_nms(dets, args.det_nms_thr)


def process_image(img, detector,
                  cls_sess, cls_in, cls_out, cls_mp, cls_w, cls_h, args):
    ih, iw = img.shape[:2]
    boxes = detect_full_and_tiles(detector, img, args)

    results = []
    for b in boxes:
        X0 = int(max(0, min(iw - 1, b["x1"])))
        Y0 = int(max(0, min(ih - 1, b["y1"])))
        X1 = int(max(0, min(iw, b["x2"])))
        Y1 = int(max(0, min(ih, b["y2"])))
        # optional padding around the ROI before classifying
        pad = args.crop_pad
        px = int((X1 - X0) * pad)
        py = int((Y1 - Y0) * pad)
        # drop tiny detections (noise) by shorter side in original pixels
        if min(X1 - X0, Y1 - Y0) < args.min_box:
            continue
        cx0, cy0 = max(0, X0 - px), max(0, Y0 - py)
        cx1, cy1 = min(iw, X1 + px), min(ih, Y1 + py)
        crop = img[cy0:cy1, cx0:cx1]
        lamps = []
        if crop.size > 0:
            cblob = cls_preprocess(crop, cls_w, cls_h)
            cout = cls_sess.run([cls_out], {cls_in: cblob})[0][0]
            cdets = cls_nms(
                classify_decode(cout, cls_mp, cls_w, cls_h, args.cls_score_thr),
                args.cls_nms_thr)
            for d in sorted(cdets, key=lambda d: -d["prob"]):
                lamps.append({
                    "label": lamp_label(d),
                    "color": COLORS.get(d["color"]),
                    "shape": SHAPES.get(d["shape"]),
                    "arrow": arrow_of(d),
                    "confidence": round(d["prob"], 4),
                })
        if args.drop_unknown and not lamps:
            continue
        results.append({
            "detector_score": round(b["prob"], 4),
            "box_xyxy": [X0, Y0, X1, Y1],
            "lamps": lamps,
            "state": signal_state(lamps),
        })
    return results


def draw(img, results):
    vis = img.copy()
    color_bgr = {"green": (0, 200, 0), "amber": (0, 200, 255),
                 "red": (0, 0, 255), None: (200, 200, 200), "unknown": (200, 200, 200)}
    for r in results:
        x0, y0, x1, y1 = r["box_xyxy"]
        main = r["lamps"][0]["color"] if r["lamps"] else None
        c = color_bgr.get(main, (200, 200, 200))
        cv2.rectangle(vis, (x0, y0), (x1, y1), c, 2)
        text = f"{r['state']} {r['detector_score']:.2f}"
        ty = y0 - 6 if y0 - 6 > 10 else y1 + 16
        cv2.putText(vis, text, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return vis


PRESET_DIR = os.path.join(HERE, "configs", "detectors")

# Search order for the ML model root, matching Autoware launch's `data_path`
# default at the end so presets resolve the same on a deployed machine.
MODEL_ROOT_CANDIDATES = ["~/autoware_data", "/opt/autoware/mlmodels"]


def model_root():
    """Resolve the model root: $TLR_MODEL_ROOT if set, else the first existing
    candidate. Presets reference models as ${TLR_MODEL_ROOT}/..., so this keeps
    them free of machine-specific absolute paths."""
    env = os.environ.get("TLR_MODEL_ROOT")
    if env:
        return os.path.expanduser(env)
    for cand in MODEL_ROOT_CANDIDATES:
        p = os.path.expanduser(cand)
        if os.path.isdir(p):
            return p
    return os.path.expanduser(MODEL_ROOT_CANDIDATES[0])


def expand_path(val):
    """Expand ~, $VARS, and ${TLR_MODEL_ROOT} (even when the env var is unset,
    using the resolved model root) in a preset/CLI path string."""
    return os.path.expanduser(os.path.expandvars(
        val.replace("${TLR_MODEL_ROOT}", model_root())
           .replace("$TLR_MODEL_ROOT", model_root())))


def list_presets():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(PRESET_DIR)
                  if f.endswith(".yaml")) if os.path.isdir(PRESET_DIR) else []


def apply_preset(ap, args):
    """Overlay preset values onto args, but only where the user did not pass an
    explicit CLI flag (explicit CLI always beats the preset)."""
    import yaml
    path = os.path.join(PRESET_DIR, args.preset + ".yaml")
    if not os.path.exists(path):
        raise SystemExit(f"unknown preset {args.preset!r}; available: {', '.join(list_presets())}")
    with open(path) as f:
        preset = yaml.safe_load(f)
    defaults = {a.dest: a.default for a in ap._actions}
    for key, val in preset.items():
        dest = key.replace("-", "_")
        if dest not in defaults:
            raise SystemExit(f"preset {args.preset}: unknown key {key!r}")
        if isinstance(val, str):
            val = expand_path(val)
        if getattr(args, dest) == defaults[dest]:
            setattr(args, dest, val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="image file or directory")
    ap.add_argument("--preset", default=None,
                    help="named detector preset from configs/detectors/ "
                         f"(available: {', '.join(list_presets())})")
    ap.add_argument("--detector", default=None,
                    help="detector model path (.onnx or .engine); overrides the preset's")
    ap.add_argument("--classifier", default=os.path.join(HERE, "traffic_light_lamp_recognizer_comlops.onnx"))
    ap.add_argument("--classifier-param", default=os.path.join(HERE, "lamp_recognizer_ml.param.yaml"))
    ap.add_argument("--comlops-param", default=os.path.join(HERE, "comlops_large_detector_ml.param.yaml"),
                    help="decode params (anchors etc.) for the CoMLOps darknet detector")
    ap.add_argument("--det-classes", default="TRAFFIC_LIGHT",
                    help="comma-separated class names kept from the CoMLOps detector")
    # Detector defaults are stricter than the ROS node (score 0.35 / nms 0.7),
    # because offline autolabeling has no map_based_detector ROI prior to filter
    # false positives: raise score to drop low-confidence hits, lower nms to merge
    # duplicate boxes on the same signal more aggressively.
    ap.add_argument("--det-score-thr", type=float, default=0.5)
    ap.add_argument("--det-nms-thr", type=float, default=0.35)
    ap.add_argument("--cls-score-thr", type=float, default=0.2)
    ap.add_argument("--cls-nms-thr", type=float, default=0.2)
    ap.add_argument("--min-box", type=float, default=8.0,
                    help="drop detections whose shorter side (px, original image) is below this")
    ap.add_argument("--drop-unknown", action="store_true",
                    help="drop signals whose classifier found no lamp (signal=='unknown')")
    ap.add_argument("--crop-pad", type=float, default=0.0, help="ROI padding ratio before classifying")
    ap.add_argument("--tiles", action="store_true",
                    help="add native-resolution tile passes (detector-input-sized crops "
                         "covering the image) on top of the full-frame pass; recovers "
                         "small distant signals lost to the letterbox downscale")
    ap.add_argument("--no-tiles", action="store_true",
                    help="force tile passes off (overrides a preset's tiles: true)")
    ap.add_argument("--tile-overlap", type=int, default=128,
                    help="minimum overlap in px between neighbouring tiles")
    ap.add_argument("--out-dir", default=None, help="write <name>.json (and <name>.viz.png with --viz)")
    ap.add_argument("--viz", action="store_true", help="save annotated images")
    ap.add_argument("--image-root", default=None,
                    help="write image paths relative to this dir (default: the input dir; "
                         "with --t4-dataset, the dataset root)")
    ap.add_argument("--t4-dataset", default=None,
                    help="T4 dataset root: fills sample_data_token/channel from "
                         "annotation/sample_data.json and uses it as image root")
    ap.add_argument("--run-id", default=None,
                    help="identifier stored in meta.run_id (default: timestamp + random)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip images whose <name>.json already exists in --out-dir "
                         "(resume an interrupted batch)")
    args = ap.parse_args()

    if args.preset:
        apply_preset(ap, args)
    if args.no_tiles:
        args.tiles = False
    if not args.detector:
        raise SystemExit("choose a detector: --preset <name> "
                         f"(available: {', '.join(list_presets())}) or --detector <model path>")
    # a --detector given directly may also use ~ / $VARS / ${TLR_MODEL_ROOT}
    args.detector = expand_path(args.detector)
    if not os.path.exists(args.detector):
        raise SystemExit(
            f"detector model not found: {args.detector}"
            + (f" (from preset {args.preset})" if args.preset else "")
            + f"\nmodel root = {model_root()} "
            "(set $TLR_MODEL_ROOT to point at your ML model directory).")

    detector = Detector(args.detector, args.comlops_param)
    if detector.kind == "comlops":
        detector.set_keep_classes([s.strip() for s in args.det_classes.split(",") if s.strip()])

    cls_sess = make_session(args.classifier)
    cls_in = cls_sess.get_inputs()[0].name
    cls_out = cls_sess.get_outputs()[0].name
    _, _, cls_h, cls_w = [d if isinstance(d, int) else 256 for d in cls_sess.get_inputs()[0].shape]
    cls_mp = load_model_params(args.classifier_param)

    backend = ("tensorrt-engine" if detector.sess is None
               else str(detector.sess.get_providers()))
    print(f"detector={os.path.basename(args.detector)} [{detector.kind}] "
          f"in={detector.w}x{detector.h} backend={backend}")
    print(f"classifier={os.path.basename(args.classifier)} in={cls_w}x{cls_h}")

    if os.path.isdir(args.image):
        paths = sorted(sum([glob.glob(os.path.join(args.image, e))
                            for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
    else:
        paths = [args.image]

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # image paths in the JSON are relative to image_root (portable across
    # machines/containers); the realpath is kept alongside as a convenience.
    if args.t4_dataset and not args.image_root:
        image_root = os.path.realpath(args.t4_dataset)
    else:
        image_root = os.path.realpath(
            args.image_root or (args.image if os.path.isdir(args.image)
                                else os.path.dirname(args.image) or "."))
    t4map = {}
    if args.t4_dataset:
        with open(os.path.join(args.t4_dataset, "annotation", "sample_data.json")) as f:
            for sd in json.load(f):
                key = os.path.realpath(os.path.join(args.t4_dataset, sd["filename"]))
                t4map[key] = sd

    # provenance block written into every per-image JSON (schema tlr_autolabel/v1)
    run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                             + "-" + uuid.uuid4().hex[:8])
    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preset": args.preset,
        "detector": os.path.basename(args.detector),
        "detector_backend": backend,
        "model_root": model_root(),
        "classifier": os.path.basename(args.classifier),
        "tiles": bool(args.tiles),
        "det_score_thr": args.det_score_thr,
        "det_nms_thr": args.det_nms_thr,
        "cls_score_thr": args.cls_score_thr,
        "cls_nms_thr": args.cls_nms_thr,
        "min_box": args.min_box,
        "crop_pad": args.crop_pad,
    }

    for seq, p in enumerate(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        if args.skip_existing and args.out_dir and \
                os.path.exists(os.path.join(args.out_dir, name + ".json")):
            continue
        img = cv2.imread(p)
        if img is None:
            print(f"[skip] {p}")
            continue
        results = process_image(img, detector,
                                cls_sess, cls_in, cls_out, cls_mp, cls_w, cls_h, args)
        name = os.path.splitext(os.path.basename(p))[0]
        print(f"\n=== {os.path.basename(p)} ({img.shape[1]}x{img.shape[0]}): "
              f"{len(results)} signal(s) ===")
        for r in results:
            lamp_str = ", ".join("{}({:.2f})".format(l["label"], l["confidence"]) for l in r["lamps"])
            print(f"  [{r['state']}] det={r['detector_score']:.2f} box={r['box_xyxy']} lamps=[{lamp_str}]")
        if args.out_dir:
            rp = os.path.realpath(p)
            rel = os.path.relpath(rp, image_root)
            sd = t4map.get(rp)
            # channel: from any CAM_*/etc. path component; frame_index: numeric
            # stem when the file is numbered (T4 style), else sequence order
            channel = next((c for c in rel.split(os.sep)[:-1]
                            if c.upper() == c and not c.startswith(".")), None)
            frame_index = int(name) if name.isdigit() else seq
            for i, r in enumerate(results):
                results[i] = {"signal_id": f"{name}-{i:02d}", **r}
            with open(os.path.join(args.out_dir, name + ".json"), "w") as f:
                json.dump({"schema_version": "tlr_autolabel/v1",
                           "image": rel,
                           "image_realpath": rp,
                           "sample_data_token": sd["token"] if sd else None,
                           "channel": channel,
                           "frame_index": frame_index,
                           "width": img.shape[1], "height": img.shape[0],
                           "meta": meta,
                           "signals": results}, f, indent=2, ensure_ascii=False)
            if args.viz:
                cv2.imwrite(os.path.join(args.out_dir, name + ".viz.png"), draw(img, results))


if __name__ == "__main__":
    main()
