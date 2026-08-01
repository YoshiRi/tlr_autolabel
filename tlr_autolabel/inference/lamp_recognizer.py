"""Lamp-recognizer classifier backends: ONNX and TensorRT engine wrapper
(REFACTOR_PLAN.md phase 6). Extracted from tlr_autolabel.py.

The per-crop decode itself (canonical LampRecognizer output format) stays in
tlr_autolabel.inference.lamp_recognizer_onnx, the single shared decode
implementation used both here and by its own single-crop debug CLI.
"""
import numpy as np

from tlr_autolabel.inference.detector import make_session
from tlr_autolabel.inference.trt import TrtServer
from tlr_autolabel.inference.lamp_recognizer_onnx import (
    COLORS, SHAPES, arrow_of, decode as classify_decode,
    load_model_params, nms as cls_nms, preprocess as cls_preprocess,
)


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


def decode_lamps(cout, cls_mp, cls_w, cls_h, args):
    lamps = []
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
    return lamps


def normalize_lamps(lamps):
    """Keep Tier A classification output independent from classifier internals."""
    allowed = ("label", "color", "shape", "arrow", "confidence")
    return [{k: lamp.get(k) for k in allowed} for lamp in lamps]


def classify_box(img, box_xyxy, run_classifier, cls_mp, cls_w, cls_h, args):
    ih, iw = img.shape[:2]
    X0, Y0, X1, Y1 = box_xyxy
    pad = args.crop_pad
    px = int((X1 - X0) * pad)
    py = int((Y1 - Y0) * pad)
    cx0, cy0 = max(0, X0 - px), max(0, Y0 - py)
    cx1, cy1 = min(iw, X1 + px), min(ih, Y1 + py)
    crop = img[cy0:cy1, cx0:cx1]
    lamps = []
    if crop.size > 0:
        cblob = cls_preprocess(crop, cls_w, cls_h)
        lamps = decode_lamps(run_classifier(cblob), cls_mp, cls_w, cls_h, args)
    return lamps


class LampClassifier:
    """Lamp classifier adapter: classify(image, bbox) -> list of lamp dicts."""

    def __init__(self, model_path, param_path, args):
        self.model_path = model_path
        self.model_params = load_model_params(param_path)
        self.args = args
        self.trt = None
        self.sess = None

        if model_path.endswith(".engine"):
            self.trt = TrtServer(model_path)
            if len(self.trt.in_shape) != 4 or len(self.trt.out_shape) != 4:
                raise SystemExit(
                    f"classifier engine must be NCHW -> NCHW, got "
                    f"{self.trt.in_shape} -> {self.trt.out_shape}")
            self.input_shape = tuple(self.trt.in_shape)
            self.batch = self.input_shape[0]
            _, _, self.height, self.width = self.input_shape
            self.backend = "tensorrt-engine"
            return

        self.sess = make_session(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        _, _, self.height, self.width = [
            d if isinstance(d, int) else 256 for d in self.sess.get_inputs()[0].shape
        ]
        self.backend = str(self.sess.get_providers())

    def run(self, cblob):
        if self.trt is None:
            return self.sess.run([self.output_name], {self.input_name: cblob})[0][0]
        if cblob.shape != self.input_shape:
            if cblob.shape[1:] != self.input_shape[1:]:
                raise RuntimeError(
                    f"classifier blob shape {cblob.shape} does not match engine "
                    f"input shape {self.input_shape}")
            padded = np.zeros(self.input_shape, dtype=np.float32)
            padded[0] = cblob[0]
            cblob = padded
        return self.trt.run(cblob)[0]

    def classify(self, img, bbox):
        return classify_box(
            img,
            bbox,
            self.run,
            self.model_params,
            self.width,
            self.height,
            self.args,
        )
