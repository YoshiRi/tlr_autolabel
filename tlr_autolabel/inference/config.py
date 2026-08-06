"""L1 inference configuration as data (docs/inference_comparison.md).

Everything the detector + classifier stack needs to run one pass, in one frozen
dataclass, resolved from `defaults <- preset <- explicit overrides`. Extracted
from `cli/autolabel.py`, which used to keep the same values on an argparse
Namespace and overlay presets onto it via `ap._actions` — that made a
configuration unbuildable without an ArgumentParser, so no in-process tool could
hold several configurations at once (which is exactly what the comparison
harness does).

The field names are deliberately identical to the old argparse dests
(`det_score_thr`, `cls_nms_thr`, `crop_pad`, ...), so the functions that used to
receive `args` (`detect_full_and_tiles`, `classify_box`, `LampClassifier`) take
an `InferenceConfig` unchanged.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
PRESET_DIR = os.path.join(REPO_ROOT, "configs", "detectors")

DEFAULT_CLASSIFIER = os.path.join(
    REPO_ROOT, "models", "traffic_light_lamp_recognizer_comlops.onnx")
DEFAULT_CLASSIFIER_PARAM = os.path.join(
    REPO_ROOT, "configs", "model_params", "lamp_recognizer_ml.param.yaml")
DEFAULT_COMLOPS_PARAM = os.path.join(
    REPO_ROOT, "configs", "model_params", "comlops_large_detector_ml.param.yaml")

# Search order for the ML model root, matching Autoware launch's `data_path`
# default at the end so presets resolve the same on a deployed machine.
MODEL_ROOT_CANDIDATES = ["~/autoware_data", "/opt/autoware/mlmodels"]

# `classifier:` values that mean "run the detector only" (box-level comparison).
NO_CLASSIFIER = ("none", "null", "off", "")


def autoware_mlmodels_root():
    """Resolve the standard Autoware model-store mount without changing the
    older TLR_MODEL_ROOT search order used by existing presets."""
    env = os.environ.get("AUTOWARE_MLMODELS") or os.environ.get("AUTOWARE_MLMODELS_ROOT")
    return os.path.expanduser(env) if env else "/opt/autoware/mlmodels"


def model_root():
    """Resolve the model root: $TLR_MODEL_ROOT if set, else the first existing
    candidate. Presets reference models as ${TLR_MODEL_ROOT}/..., so this keeps
    them free of machine-specific absolute paths."""
    env = os.environ.get("TLR_MODEL_ROOT")
    if env:
        return os.path.expanduser(env)
    for cand in MODEL_ROOT_CANDIDATES:
        p = os.path.expanduser(cand)
        if os.path.isdir(p):
            return p
    return os.path.expanduser(MODEL_ROOT_CANDIDATES[0])


def expand_path(val):
    """Expand ~, $VARS, ${TLR_MODEL_ROOT}, and ${AUTOWARE_MLMODELS} in a
    preset/CLI path string."""
    return os.path.expanduser(os.path.expandvars(
        val.replace("${TLR_MODEL_ROOT}", model_root())
           .replace("$TLR_MODEL_ROOT", model_root())
           .replace("${AUTOWARE_MLMODELS}", autoware_mlmodels_root())
           .replace("$AUTOWARE_MLMODELS", autoware_mlmodels_root())))


def list_presets():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(PRESET_DIR)
                  if f.endswith(".yaml")) if os.path.isdir(PRESET_DIR) else []


def load_preset(name):
    """Read one `configs/detectors/<name>.yaml` into a dict (no expansion)."""
    import yaml
    path = os.path.join(PRESET_DIR, name + ".yaml")
    if not os.path.exists(path):
        raise SystemExit(
            f"unknown preset {name!r}; available: {', '.join(list_presets())}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class InferenceConfig:
    """One resolved detector+classifier configuration.

    A configuration is what the comparison harness treats as a unit: two runs
    differing in any field here are two configurations, and `meta` in the Tier A
    output records the fields that change the numbers.
    """

    # --- detector ---
    detector: str | None = None
    detector_type: str | None = None
    comlops_param: str = DEFAULT_COMLOPS_PARAM
    det_classes: str = "TRAFFIC_LIGHT"
    det_score_thr: float = 0.35
    det_low_score_thr: float | None = None
    det_nms_thr: float = 0.35
    tiles: bool = False
    tile_overlap: int = 128
    min_box: float = 8.0
    # --- classifier (None = detector-only run) ---
    classifier: str | None = DEFAULT_CLASSIFIER
    classifier_type: str | None = None
    classifier_param: str = DEFAULT_CLASSIFIER_PARAM
    cls_score_thr: float = 0.2
    cls_nms_thr: float = 0.2
    crop_pad: float = 0.0
    classify_low_detections: bool = False
    # --- post ---
    drop_unknown: bool = False
    # --- provenance / instrumentation (off by default: they add Tier A keys) ---
    preset: str | None = None
    record_timing: bool = False
    record_model_digest: bool = False

    @property
    def classifier_enabled(self) -> bool:
        return bool(self.classifier)


CONFIG_FIELDS = tuple(f.name for f in fields(InferenceConfig))
_PATH_FIELDS = ("detector", "classifier", "classifier_param", "comlops_param")


def _normalize(key, val):
    if key == "classifier" and isinstance(val, str) and val.strip().lower() in NO_CLASSIFIER:
        return None
    if key == "classifier" and val is None:
        return None
    if isinstance(val, str) and key in _PATH_FIELDS:
        return expand_path(val)
    return val


def resolve_config(preset=None, overrides=None) -> InferenceConfig:
    """Build a config from `defaults <- preset <- overrides`.

    `overrides` are the values a caller set explicitly; they always beat the
    preset (the rule the CLI has always had). Unknown preset keys are a hard
    error, as before — a typo in a preset must not silently do nothing.
    """
    values = {}
    if preset:
        for key, val in load_preset(preset).items():
            dest = key.replace("-", "_")
            if dest not in CONFIG_FIELDS:
                raise SystemExit(
                    f"preset {preset}: unknown key {key!r}; valid keys: "
                    f"{', '.join(CONFIG_FIELDS)}")
            values[dest] = _normalize(dest, val)
    for key, val in (overrides or {}).items():
        if key not in CONFIG_FIELDS:
            raise SystemExit(f"unknown configuration key {key!r}")
        values[key] = _normalize(key, val)
    # `preset` is recorded on the config, so it must not also arrive as a value
    values.pop("preset", None)
    cfg = InferenceConfig(preset=preset, **values)
    # paths given directly (CLI/matrix) may also use ~ / $VARS / ${TLR_MODEL_ROOT}
    return replace(cfg, **{f: (expand_path(getattr(cfg, f))
                               if isinstance(getattr(cfg, f), str) else getattr(cfg, f))
                           for f in _PATH_FIELDS})


def config_from_args(parser, args, extra_overrides=None) -> InferenceConfig:
    """Config from an argparse Namespace: any value the user typed explicitly
    (i.e. differs from the parser default) becomes an override, the rest is left
    to the preset. This reproduces the old `apply_preset` overlay exactly."""
    defaults = {a.dest: a.default for a in parser._actions}
    overrides = {}
    for name in CONFIG_FIELDS:
        if name in ("preset", "record_timing", "record_model_digest"):
            continue
        if not hasattr(args, name):
            continue
        val = getattr(args, name)
        if name in defaults and val == defaults[name]:
            continue
        overrides[name] = val
    overrides.update(extra_overrides or {})
    return resolve_config(getattr(args, "preset", None), overrides)


def validate_config(cfg: InferenceConfig) -> None:
    """Fail loud, before any model is loaded, with the same messages the CLI
    has always produced."""
    from_preset = f" (from preset {cfg.preset})" if cfg.preset else ""
    if not cfg.detector:
        raise SystemExit("choose a detector: --preset <name> "
                         f"(available: {', '.join(list_presets())}) or --detector <model path>")
    if cfg.det_low_score_thr is not None and cfg.det_low_score_thr > cfg.det_score_thr:
        raise SystemExit("--det-low-score-thr must be <= --det-score-thr")
    if not os.path.exists(cfg.detector):
        raise SystemExit(
            f"detector model not found: {cfg.detector}{from_preset}"
            f"\nmodel root = {model_root()} "
            "(set $TLR_MODEL_ROOT to point at your ML model directory).")
    if cfg.classifier_enabled:
        if not os.path.exists(cfg.classifier):
            raise SystemExit(
                f"classifier model not found: {cfg.classifier}{from_preset}"
                f"\nmodel root = {model_root()} "
                "(set $TLR_MODEL_ROOT to point at your ML model directory).")
        if not os.path.exists(cfg.classifier_param):
            raise SystemExit(
                f"classifier param not found: {cfg.classifier_param}{from_preset}")
    elif cfg.drop_unknown:
        raise SystemExit(
            "--drop-unknown with no classifier would drop every detection "
            "(a detector-only run has no lamp state); drop one of the two.")


def file_digest(path, _cache={}):
    """sha256 of a model file, memoized per process. Two runs that claim the
    same model name are only comparable if this matches (PLAN.md item 9)."""
    if not path or not os.path.exists(path):
        return None
    key = (os.path.realpath(path), os.path.getmtime(path), os.path.getsize(path))
    if key not in _cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _cache[key] = h.hexdigest()
    return _cache[key]
