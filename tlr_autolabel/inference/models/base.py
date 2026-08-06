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


@runtime_checkable
class ClassifierModel(Protocol):
    """One classifier family's preprocess + decode over a traffic-light crop.

    The runtime (`LampClassifier`) owns the session/engine, the crop extraction
    and the padding; the model owns how a crop becomes a blob and how the raw
    output becomes lamps.

    `decode` returns the canonical Tier A lamp list — dicts with
    `{label, color, shape, arrow, confidence}`. A family that predicts one label
    for the whole signal (rather than per-lamp boxes) returns a single-element
    list; that keeps `state` comparable across families, which is the whole
    point of the canonical vocabulary.
    """

    name: str
    input_size: tuple[int, int] | None  # (w, h) when the family fixes it, else None

    def preprocess(self, crop: np.ndarray, w: int, h: int) -> np.ndarray:
        """Return the NCHW float32 blob for one crop."""
        ...

    def decode(self, output: np.ndarray, w: int, h: int,
               score_thr: float, nms_thr: float) -> list[dict]:
        """Raw session/engine output (batch dim squeezed) -> canonical lamps."""
        ...
