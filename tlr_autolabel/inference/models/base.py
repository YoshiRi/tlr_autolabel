"""Detector model protocol (docs/model_interface.md, phase A).

A *model* owns one family's preprocess + decode; it does not own the
onnxruntime session or TensorRT engine (that stays in the `Detector` runtime
wrapper). Family-specific constants (rgb/norm, anchors, expected output count)
live here, with the model, instead of scattered across the wrapper.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DetectorModel(Protocol):
    name: str            # registry key, e.g. "yolox"
    num_outputs: int     # expected ONNX output count (validated against the model)
    supports_engine: bool  # runnable through the single-output TrtServer path

    def preprocess(self, img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float]:
        """Return (NCHW float32 blob, letterbox scale). Owns rgb/norm/pad."""
        ...

    def decode(self, outputs: list[np.ndarray], w: int, h: int,
               score_thr: float) -> list[dict]:
        """`outputs` are the raw session/engine outputs with the batch dim
        already squeezed. Return dets in network pixel space as dicts with
        keys {prob, x1, y1, x2, y2} (+ optional class_id)."""
        ...

    def set_keep_classes(self, names: list[str]) -> None:
        """No-op for single-class families; filters class_id for multi-class."""
        ...
