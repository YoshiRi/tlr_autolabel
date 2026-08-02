"""Traffic-light detector backends: ONNX YOLOX, CoMLOps darknet, and the
TensorRT engine wrapper, plus tiling/NMS orchestration around them
(REFACTOR_PLAN.md phase 6). Extracted from tlr_autolabel.py.
"""
import math
import warnings

import cv2
import numpy as np
import onnxruntime as ort

from tlr_autolabel.inference.trt import TrtServer


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


class Detector:
    """Runtime wrapper: owns the onnxruntime session or TensorRT engine, resolves
    the input size, and exposes detect(img, score_thr) -> (dets, scale) with dets
    in network (letterboxed) pixel space. The family-specific preprocess/decode
    live in a DetectorModel from tlr_autolabel.inference.models; this wrapper only
    picks one (explicit `model_type`, else inferred from the output count) and
    feeds it batch-squeezed outputs.

    Accepts .onnx (onnxruntime, CPU) or .engine (TensorRT via trt_run, GPU).
    Prefer the int8 .engine for the YOLOX detectors when available: the fp32
    ONNX fires high-confidence false positives on motion-blur/texture that the
    deployed int8 engine does not (verified on frame 00250: 4 FPs at fp32 vs 0
    at int8, with identical true positives on 00000)."""

    def __init__(self, model_path, comlops_param_path, model_type=None):
        from tlr_autolabel.inference import models

        if model_path.endswith(".engine"):
            self.sess = None
            self.trt = TrtServer(model_path)
            _, _, self.h, self.w = self.in_shape = self.trt.in_shape
            # single-output engine; family cannot be read from the engine, so
            # trust the preset's model_type and default to yolox (the only
            # engine-supported family) otherwise.
            n_out = 1
        else:
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

        if model_type:
            self.kind = model_type
        else:
            self.kind = models.infer_detector_type(n_out)
            warnings.warn(
                f"detector_type not set; inferred {self.kind!r} from {n_out} "
                "model output(s). Set detector_type in the preset or pass "
                "--detector-type to make model selection explicit.",
                stacklevel=2)
        self.model = models.build_detector_model(
            self.kind, model_path, {"comlops_param_path": comlops_param_path})
        if self.sess is not None and self.model.num_outputs != n_out:
            raise SystemExit(
                f"detector_type={self.kind} expects {self.model.num_outputs} "
                f"output(s) but the model has {n_out} "
                f"({[o.name for o in self.sess.get_outputs()]}).")
        if self.sess is None and not self.model.supports_engine:
            raise SystemExit(
                f"detector_type={self.kind} does not support the .engine path; "
                "use its .onnx export.")

    def set_keep_classes(self, names):
        self.model.set_keep_classes(names)

    def _infer(self, blob):
        """Run inference, returning outputs as a list with the batch dim squeezed
        (matching what the decode functions expect)."""
        if self.sess is None:
            return self.trt.run(blob)  # single-output engine
        return [o[0] for o in self.sess.run(None, {self.in_name: blob})]

    def detect(self, img, score_thr):
        blob, scale = self.model.preprocess(img, self.w, self.h)
        outputs = self._infer(blob)
        return self.model.decode(outputs, self.w, self.h, score_thr), scale


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


def detect_full_and_tiles(detector, img, args, score_thr=None):
    """Full-frame letterboxed pass, plus (with --tiles) native-resolution tile
    passes, merged by one global NMS in original pixel coords. Tiles overlap by
    >= min_overlap px so a signal cut at one tile's edge is seen whole by the
    neighbour; the containment rule in det_nms then merges the clipped duplicate."""
    score_thr = args.det_score_thr if score_thr is None else score_thr
    dets = detect_in_orig(detector, img, score_thr)
    if args.tiles:
        ih, iw = img.shape[:2]
        for oy in tile_origins(ih, detector.h, args.tile_overlap):
            for ox in tile_origins(iw, detector.w, args.tile_overlap):
                tile = img[oy:oy + detector.h, ox:ox + detector.w]
                for b in detect_in_orig(detector, tile, score_thr):
                    b["x1"] += ox; b["x2"] += ox
                    b["y1"] += oy; b["y2"] += oy
                    dets.append(b)
    return det_nms(dets, args.det_nms_thr)


def clipped_detection_box(b, iw, ih, min_box):
    X0 = int(max(0, min(iw - 1, b["x1"])))
    Y0 = int(max(0, min(ih - 1, b["y1"])))
    X1 = int(max(0, min(iw, b["x2"])))
    Y1 = int(max(0, min(ih, b["y2"])))
    if min(X1 - X0, Y1 - Y0) < min_box:
        return None
    return X0, Y0, X1, Y1
