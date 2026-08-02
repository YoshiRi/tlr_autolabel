# Model interface design (detector / classifier plug-in)

Status: **phase A implemented** (2026-08-02); phases B/C pending. Prerequisite
for onboarding other open models (PLAN.md item 11). No behavior change — this is
a seam, not a rewrite.

Phase A shipped: `tlr_autolabel/inference/models/` (registry + `DetectorModel`
protocol + `yolox`/`comlops` model modules), with `Detector` routed through the
registry and the output-count guess kept as the fallback. The `.decode` /
`.preprocess` bodies call the existing functions verbatim, so detector output is
bit-identical (unit-pinned in `tests/test_detector_models.py`; verified on the
real GPU onnx and `.engine` paths). Phases B (explicit `detector_type` in
presets + `--detector-type`) and C (onboard one external model) remain. The
classifier seam is deferred until a second classifier family appears (only
LampRecognizer exists today).

## Problem

Adding a detector or classifier family today means editing the runtime wrapper
classes, because the family is chosen by **implicit signals**:

- `Detector.__init__` (`inference/detector.py`) guesses the family from the ONNX
  **output count**: `1 → yolox`, `3 → comlops`, else `SystemExit`. The `.engine`
  path is hard-coded to `yolox`.
- `detect()` branches on `self.kind` and inlines the per-family preprocess args
  (`rgb`, `norm`) and decode call.
- `LampClassifier` supports exactly one decode (LampRecognizer).

Consequences:

- A second family that happens to share an output count collides silently.
- An `.engine` can never be anything but yolox, even when the preset knows
  better.
- Every new model edits a growing `if/elif`, and the family-specific knowledge
  (preprocess constants, decode, expected output count) is scattered across the
  wrapper instead of living with the model.

## Goals / non-goals

**Goals**
- A new model family = **one new module + one registry entry**, no edits to the
  runtime wrappers.
- Family selection is **explicit** (`detector_type` / `classifier_type` in the
  preset), with the current output-count guess kept only as a back-compat
  fallback.
- Bit-identical output for the existing yolox / comlops / LampRecognizer paths.

**Non-goals**
- No change to the Tier A `tlr_autolabel/v1` output schema.
- No change to tiling / NMS / crop orchestration (those stay in the wrappers —
  they are family-agnostic).
- Not touching `cli/match.py` or the data IFs (separate, deferred work).

## Design

Split each wrapper into two responsibilities:

1. **Runtime** (stays in `Detector` / `LampClassifier`): owns the onnxruntime
   session or `TrtServer`, input-shape resolution, batching/padding, tiling, NMS,
   crop extraction. Family-agnostic.
2. **Model** (new, pluggable): owns preprocess + decode for one family. Pure
   numpy, no session ownership.

A tiny registry maps a `model_type` string to a **factory** that builds the
model object from `(model_path, params)`.

### Preset schema addition

```yaml
# configs/detectors/yolox-1920-int8.yaml
detector: ${TLR_MODEL_ROOT}/.../*.engine
detector_type: yolox        # NEW — explicit family
tiles: true
```

`detector_type` is optional for back-compat: absent → infer from output count
(current behavior) and warn. Presets are overlaid onto argparse, so this needs a
matching `--detector-type` flag (default `None`).

### Protocols

```python
# tlr_autolabel/inference/models/base.py
from typing import Protocol
import numpy as np

class DetectorModel(Protocol):
    name: str                 # registry key, e.g. "yolox"
    num_outputs: int          # expected ONNX output count (validation)
    supports_engine: bool     # can run through the single-output TrtServer path

    def preprocess(self, img: np.ndarray, w: int, h: int) -> tuple[np.ndarray, float]:
        """Return (NCHW blob, letterbox scale). Owns rgb/norm/pad choices."""

    def decode(self, outputs: list[np.ndarray], w: int, h: int,
               score_thr: float) -> list[dict]:
        """outputs = raw session outputs, batch dim already squeezed.
        Return dets in network pixel space: {prob, x1, y1, x2, y2[, class_id]}."""

    def set_keep_classes(self, names: list[str]) -> None:
        """No-op for single-class families; filters class_id for multi-class."""


class ClassifierModel(Protocol):
    name: str

    def preprocess(self, crop: np.ndarray, w: int, h: int) -> np.ndarray: ...

    def decode(self, output: np.ndarray, w: int, h: int,
               score_thr: float, nms_thr: float) -> list[dict]:
        """Return lamp dicts {label, color, shape, arrow, confidence}."""
```

These are exactly the boundaries the current code already has — the yolox/comlops
decode functions (`det_decode`, `comlops_decode`) and the LampRecognizer decode
(`lamp_recognizer_onnx.decode`/`nms`) become the bodies of `.decode`, and
`det_preprocess` / `cls_preprocess` become `.preprocess`. **The existing
functions are reused verbatim**, just wrapped.

### Registry

```python
# tlr_autolabel/inference/models/__init__.py
_DETECTORS: dict[str, Callable[..., DetectorModel]] = {}

def register_detector(name):
    def deco(factory):
        _DETECTORS[name] = factory
        return factory
    return deco

def build_detector_model(model_type, model_path, params) -> DetectorModel:
    if model_type not in _DETECTORS:
        raise SystemExit(
            f"unknown detector_type {model_type!r}; "
            f"available: {', '.join(sorted(_DETECTORS))}")
    return _DETECTORS[model_type](model_path=model_path, params=params)

def infer_detector_type(n_out: int) -> str:
    return {1: "yolox", 3: "comlops"}.get(n_out) or _fail(n_out)
```

### Wrapper after the change (sketch)

```python
class Detector:
    def __init__(self, model_path, params, model_type=None):
        self._open_runtime(model_path)          # session or TrtServer, self.w/h, n_out
        if model_type is None:
            model_type = infer_detector_type(self.n_out)   # back-compat + warn
        self.model = build_detector_model(model_type, model_path, params)
        if self.model.num_outputs != self.n_out:
            raise SystemExit(
                f"preset says detector_type={model_type} "
                f"({self.model.num_outputs} outputs) but model has {self.n_out}")

    def detect(self, img, score_thr):
        blob, scale = self.model.preprocess(img, self.w, self.h)
        outputs = self._run(blob)               # engine or session, list of arrays
        return self.model.decode(outputs, self.w, self.h, score_thr), scale
```

`set_keep_classes` forwards to `self.model.set_keep_classes`. Tiling, NMS,
letterbox-pad rejection, crop classification: unchanged.

## File layout

```
tlr_autolabel/inference/
  models/
    __init__.py          # registry + infer_*_type
    base.py              # Protocols
    yolox.py             # @register_detector("yolox")  -> wraps det_decode
    comlops.py           # @register_detector("comlops") -> wraps comlops_decode
    lamp_recognizer.py   # @register_classifier("lamp_recognizer")
  detector.py            # Detector runtime (session/engine/tiling) — uses registry
  lamp_recognizer.py     # LampClassifier runtime — uses registry
  lamp_recognizer_onnx.py# unchanged shared decode + debug CLI
  trt.py                 # unchanged
```

## Onboarding a new model (worked example)

To add, say, an RT-DETR-style detector:

1. `tlr_autolabel/inference/models/rtdetr.py`:
   ```python
   @register_detector("rtdetr")
   class RtDetrDetector:
       name = "rtdetr"; num_outputs = 2; supports_engine = False
       def __init__(self, model_path, params): ...
       def preprocess(self, img, w, h): ...   # own norm/rgb
       def decode(self, outputs, w, h, score_thr): ...  # -> {prob,x1,y1,x2,y2}
       def set_keep_classes(self, names): ...
   ```
2. Import it in `models/__init__.py` (or auto-discover the package).
3. Add a preset with `detector_type: rtdetr`.

No wrapper edits. The classifier side is symmetric.

## Migration (phased, small)

- **A. Introduce the seam, no behavior change.** Add `models/` (registry +
  protocols), wrap the existing yolox/comlops/LampRecognizer decode/preprocess,
  route the wrappers through the registry, keep output-count inference as the
  default. Gate: the CLI smoke test output stays **bit-identical** (add a golden
  compare if not already covered).
- **B. Make type explicit.** Add `--detector-type` / `--classifier-type`, set
  `detector_type` in the shipped presets, downgrade the inference path to a
  warned fallback. Let the `.engine` presets declare their real type.
- **C. Onboard one external model** end to end to prove the seam is thin, then
  document the recipe in README.

## Testing

- Reuse `tests/test_inference_detector.py` /
  `tests/test_inference_lamp_recognizer.py` — they already unit-test the decode
  functions; point them at the wrapped model objects.
- Add a registry test: every shipped preset's `detector_type` resolves, and its
  `num_outputs` matches a tiny synthetic ONNX (or is asserted against the
  documented family).
- Keep a golden fixture for one real frame per family if a small model is
  available, asserting bit-identical dets before/after the migration.

## Open questions

- Auto-discovery vs. explicit imports in `models/__init__.py` (explicit is
  simpler and lint-friendly; auto-discovery avoids a forgotten import). Lean
  explicit.
- Where the comlops param file resolves from once `params` is a generic bag —
  keep the current `--comlops-param` flag, pass it through as
  `params={"param_path": ...}`.
- Whether `classifier_type` is worth adding in phase B given there is only one
  classifier family today (probably defer until a second one appears, but the
  registry costs nothing to add).
