"""Lamp-classifier runtime: ONNX / TensorRT engine session + crop handling
(REFACTOR_PLAN.md phase 6). Extracted from tlr_autolabel.py.

The runtime owns the session/engine, the input size, batching/padding and the
crop extraction; the family-specific preprocess/decode live in a
`ClassifierModel` from `tlr_autolabel.inference.models` (selected by
`classifier_type`, default `lamp_recognizer`), mirroring how `Detector` and
`DetectorModel` are split. `lamp_label` is re-exported here because it was part
of this module's surface before the split.
"""
import numpy as np

from tlr_autolabel.inference.detector import make_session
from tlr_autolabel.inference.models.lamp_recognizer import lamp_label  # noqa: F401
from tlr_autolabel.inference.trt import TrtServer


def signal_state(lamps):
    """Canonical signal state: lamp tokens sorted alphabetically, comma-joined;
    'unknown' when no lamp was recognized. Sorting makes the same physical
    state always serialize identically (confidence order does not)."""
    return ",".join(sorted(l["label"] for l in lamps)) if lamps else "unknown"


def normalize_lamps(lamps):
    """Keep Tier A classification output independent from classifier internals."""
    allowed = ("label", "color", "shape", "arrow", "confidence")
    return [{k: lamp.get(k) for k in allowed} for lamp in lamps]


def classify_box(img, box_xyxy, model, run_classifier, cls_w, cls_h, args):
    """Crop `box_xyxy` (padded by `crop_pad`) and run the classifier over it."""
    ih, iw = img.shape[:2]
    X0, Y0, X1, Y1 = box_xyxy
    pad = args.crop_pad
    px = int((X1 - X0) * pad)
    py = int((Y1 - Y0) * pad)
    cx0, cy0 = max(0, X0 - px), max(0, Y0 - py)
    cx1, cy1 = min(iw, X1 + px), min(ih, Y1 + py)
    crop = img[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return []
    cblob = model.preprocess(crop, cls_w, cls_h)
    return model.decode(run_classifier(cblob), cls_w, cls_h,
                        args.cls_score_thr, args.cls_nms_thr)


class LampClassifier:
    """Classifier adapter: classify(image, bbox) -> list of canonical lamp dicts."""

    def __init__(self, model_path, param_path, args, model_type=None):
        from tlr_autolabel.inference import models

        self.model_path = model_path
        self.args = args
        self.trt = None
        self.sess = None
        self.kind = model_type or models.DEFAULT_CLASSIFIER_TYPE
        self.model = models.build_classifier_model(
            self.kind, model_path, {"classifier_param_path": param_path})

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

    @property
    def model_params(self):
        """Back-compat accessor for the family's decode params."""
        return getattr(self.model, "model_params", None)

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
        return classify_box(img, bbox, self.model, self.run,
                            self.width, self.height, self.args)
