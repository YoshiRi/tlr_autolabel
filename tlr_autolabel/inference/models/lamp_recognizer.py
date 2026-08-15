"""LampRecognizer classifier model (docs/model_interface.md, classifier side).

Wraps the existing LampRecognizer preprocess/decode verbatim; behavior is
unchanged. RGB input, /255 normalization, one anchor-grid output decoded into
per-lamp color/shape/arrow.

This is the classifier half of the plug-in seam that phase A/B built for
detectors. It exists now because comparing "any detector x any classifier"
needs classifier families to be selectable by name, the same way
`detector_type` selects a detector family.
"""
from __future__ import annotations

import numpy as np

from tlr_autolabel.inference.lamp_recognizer_onnx import (
    COLORS, SHAPES, arrow_of, decode as classify_decode, load_model_params,
    nms as cls_nms, preprocess as cls_preprocess,
)
from tlr_autolabel.inference.models import register_classifier


def lamp_label(d):
    """Canonical lamp token: {color}-{shape}[-{direction}] (e.g. green-arrow-up,
    red-circle, red-ped). Direction only for arrow lamps."""
    label = f"{COLORS.get(d['color'])}-{SHAPES.get(d['shape'])}"
    arrow = arrow_of(d)
    if arrow:
        label += f"-{arrow}"
    return label


@register_classifier("lamp_recognizer")
class LampRecognizerModel:
    name = "lamp_recognizer"
    input_size = None  # resolved from the ONNX/engine input shape by the runtime

    def __init__(self, model_path=None, params=None):
        params = params or {}
        param_path = params.get("classifier_param_path")
        if not param_path:
            raise SystemExit(
                "classifier_type=lamp_recognizer needs its decode params; "
                "pass --classifier-param (configs/model_params/lamp_recognizer_ml.param.yaml)")
        self.model_params = load_model_params(param_path)

    def preprocess(self, crop: np.ndarray, w: int, h: int) -> np.ndarray:
        return cls_preprocess(crop, w, h)

    def decode(self, output: np.ndarray, w: int, h: int,
               score_thr: float, nms_thr: float) -> list[dict]:
        dets = cls_nms(classify_decode(output, self.model_params, w, h, score_thr),
                       nms_thr)
        return [{
            "label": lamp_label(d),
            "color": COLORS.get(d["color"]),
            "shape": SHAPES.get(d["shape"]),
            "arrow": arrow_of(d),
            "confidence": round(d["prob"], 4),
        } for d in sorted(dets, key=lambda d: -d["prob"])]
