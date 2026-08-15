"""Detector model registry (docs/model_interface.md, phase A).

A new detector family is one module under this package that defines a
`DetectorModel` and calls `@register_detector("<name>")`. The `Detector`
runtime wrapper resolves the family through `build_detector_model`, so adding a
family needs no edit to the wrapper.

Selection is explicit via a preset `detector_type`; when absent, the wrapper
falls back to `infer_detector_type` (the historical ONNX-output-count guess).
"""
from __future__ import annotations

from typing import Callable

from tlr_autolabel.inference.models.base import ClassifierModel, DetectorModel

_DETECTORS: dict[str, Callable[..., DetectorModel]] = {}
_CLASSIFIERS: dict[str, Callable[..., ClassifierModel]] = {}
_BUILTINS_LOADED = False

DEFAULT_CLASSIFIER_TYPE = "lamp_recognizer"


def register_detector(name: str):
    """Decorator: register a DetectorModel factory under `name`."""
    def deco(factory: Callable[..., DetectorModel]):
        _DETECTORS[name] = factory
        return factory
    return deco


def register_classifier(name: str):
    """Decorator: register a ClassifierModel factory under `name`."""
    def deco(factory: Callable[..., ClassifierModel]):
        _CLASSIFIERS[name] = factory
        return factory
    return deco


def load_builtins() -> None:
    """Import the shipped model modules so their @register_detector runs.

    Deferred (not imported at package load) to avoid a cycle: the builtin
    modules import the shared preprocess/decode functions from
    `inference.detector`, which in turn calls this loader. By the time a
    `Detector` is constructed, `inference.detector` is fully defined."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    from tlr_autolabel.inference.models import (  # noqa: F401
        comlops, lamp_recognizer, yolox,
    )


def list_detectors() -> list[str]:
    load_builtins()
    return sorted(_DETECTORS)


def list_classifiers() -> list[str]:
    load_builtins()
    return sorted(_CLASSIFIERS)


def infer_detector_type(n_out: int) -> str:
    """Historical fallback: family from the ONNX output count (1=yolox,
    3=comlops). Used only when a preset does not set detector_type."""
    guess = {1: "yolox", 3: "comlops"}.get(n_out)
    if guess is None:
        raise SystemExit(
            f"unrecognized detector family: {n_out} outputs. Known: 1 output = "
            "yolox head, 3 outputs = CoMLOps darknet. Set detector_type in the "
            "preset or add a new model under tlr_autolabel/inference/models/.")
    return guess


def build_detector_model(model_type: str, model_path: str, params: dict) -> DetectorModel:
    load_builtins()
    if model_type not in _DETECTORS:
        raise SystemExit(
            f"unknown detector_type {model_type!r}; available: "
            f"{', '.join(sorted(_DETECTORS))}")
    return _DETECTORS[model_type](model_path=model_path, params=params)


def build_classifier_model(model_type: str | None, model_path: str,
                           params: dict) -> ClassifierModel:
    """Resolve a classifier family. Unlike the detector side there is no
    output-shape guess: absent `classifier_type` means the one historical
    family (`lamp_recognizer`), which is what every existing preset implies."""
    load_builtins()
    model_type = model_type or DEFAULT_CLASSIFIER_TYPE
    if model_type not in _CLASSIFIERS:
        raise SystemExit(
            f"unknown classifier_type {model_type!r}; available: "
            f"{', '.join(sorted(_CLASSIFIERS))}")
    return _CLASSIFIERS[model_type](model_path=model_path, params=params)
