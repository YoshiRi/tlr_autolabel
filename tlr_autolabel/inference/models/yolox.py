"""YOLOX traffic-light detector model (docs/model_interface.md, phase A).

Wraps the existing yolox preprocess/decode verbatim; behavior is unchanged.
BGR input, no normalization (raw 0-255), single output head.
"""
from __future__ import annotations

import numpy as np

from tlr_autolabel.inference.detector import det_decode, det_preprocess
from tlr_autolabel.inference.models import register_detector


@register_detector("yolox")
class YoloxDetectorModel:
    name = "yolox"
    num_outputs = 1
    supports_engine = True

    def __init__(self, model_path=None, params=None):
        pass

    def preprocess(self, img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float]:
        return det_preprocess(img, w, h)  # BGR, norm=1.0

    def decode(self, outputs: list[np.ndarray], w: int, h: int,
               score_thr: float) -> list[dict]:
        return det_decode(outputs[0], w, h, score_thr)

    def set_keep_classes(self, names: list[str]) -> None:
        pass  # single-class head; nothing to filter
