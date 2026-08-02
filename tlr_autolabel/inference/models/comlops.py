"""CoMLOps-Large darknet detector model (docs/model_interface.md, phase A).

Wraps the existing comlops preprocess/decode verbatim; behavior is unchanged.
RGB input, /255 normalization, three output scales (strides 8/16/32), decode
params (anchors etc.) from the CoMLOps param yaml. Multi-class; only the
kept class ids survive.
"""
from __future__ import annotations

import numpy as np

from tlr_autolabel.inference.detector import (
    comlops_decode, comlops_load_params, det_preprocess,
)
from tlr_autolabel.inference.models import register_detector


@register_detector("comlops")
class ComlopsDetectorModel:
    name = "comlops"
    num_outputs = 3
    supports_engine = False

    def __init__(self, model_path=None, params=None):
        params = params or {}
        param_path = params.get("comlops_param_path")
        self.mp = comlops_load_params(param_path)
        self.labels = self.mp["labels"]
        self.keep_ids: set[int] = set()

    def preprocess(self, img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float]:
        return det_preprocess(img, w, h, rgb=True, norm=1.0 / 255.0)

    def decode(self, outputs: list[np.ndarray], w: int, h: int,
               score_thr: float) -> list[dict]:
        return comlops_decode(outputs, self.mp, score_thr, self.keep_ids)

    def set_keep_classes(self, names: list[str]) -> None:
        self.keep_ids = {self.labels.index(n) for n in names}
