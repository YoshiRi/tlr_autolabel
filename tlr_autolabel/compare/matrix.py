"""Comparison matrix: one frame source x N detector/classifier configurations.

This is L1 work (it runs inference), and it deliberately produces nothing but
ordinary Tier A label directories — one per configuration. That keeps every
existing downstream tool applicable to a comparison run (`match_traffic_lights`,
`compare_runs`, `render_l1_video`, `eval_l1_vs_t4`) and keeps the analysis in
L6 where it cannot re-run inference.

Configurations run one at a time, frames inner: a `.engine` keeps a `trt_run`
helper (and a deserialized engine) resident, so holding several 1920x1280 int8
engines at once is how you run a GPU out of memory. Frames are materialized
first when they do not already exist as files, so every configuration sees
identical pixels rather than an independent video/bag decode.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from time import perf_counter

from tlr_autolabel.frames import build_frame_source
from tlr_autolabel.frames.cache import materialize
from tlr_autolabel.inference.config import (
    CONFIG_FIELDS, InferenceConfig, resolve_config,
)
from tlr_autolabel.inference.pipeline import Pipeline, new_run_id

MATRIX_SCHEMA = "tlr_compare_matrix/v1"
RUN_SCHEMA = "tlr_compare_run/v1"
COMBO_KEYS = ("name", "preset", "overrides")


@dataclass(frozen=True)
class Combo:
    """One named configuration in the matrix."""

    name: str
    config: InferenceConfig


@dataclass
class Matrix:
    frames: dict
    combos: list
    frames_dir: str | None = None

    def describe(self) -> dict:
        return {"frames": self.frames,
                "combos": [{"name": c.name, "config": asdict(c.config)}
                           for c in self.combos]}


def _combo_from_dict(entry, defaults, index):
    if not isinstance(entry, dict):
        raise SystemExit(f"combos[{index}]: expected a mapping, got {type(entry).__name__}")
    entry = {k.replace("-", "_"): v for k, v in entry.items()}
    unknown = [k for k in entry if k not in COMBO_KEYS and k not in CONFIG_FIELDS]
    if unknown:
        raise SystemExit(
            f"combos[{index}]: unknown key(s) {', '.join(sorted(unknown))}; "
            f"allowed: {', '.join(COMBO_KEYS)} or any configuration field "
            f"({', '.join(CONFIG_FIELDS)})")
    overrides = dict(defaults)
    overrides.update({k: v for k, v in entry.items()
                      if k in CONFIG_FIELDS and k not in COMBO_KEYS})
    inline = entry.get("overrides") or {}
    if not isinstance(inline, dict):
        raise SystemExit(f"combos[{index}]: 'overrides' must be a mapping")
    overrides.update({k.replace("-", "_"): v for k, v in inline.items()})
    preset = entry.get("preset")
    name = entry.get("name") or preset
    if not name:
        raise SystemExit(f"combos[{index}]: needs a 'name' (or a 'preset' to name it after)")
    if "/" in name or os.sep in name:
        raise SystemExit(f"combos[{index}]: name {name!r} must not contain a path separator")
    return Combo(name=name, config=resolve_config(preset, overrides))


def load_matrix(path) -> Matrix:
    """Read a comparison matrix YAML.

    ```yaml
    frames: {kind: video, uri: drive.mp4, stride: 5}
    defaults: {det_score_thr: 0.35}        # applied to every combo
    combos:
      - {name: S960,       preset: yolox-960-int8}
      - {name: S960_tiles, preset: yolox-960-int8, overrides: {tiles: true}}
      - {name: boxes_only, preset: yolox-960-int8, classifier: none}
    ```
    """
    import yaml

    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise SystemExit(f"{path}: expected a mapping at the top level")
    version = doc.get("schema_version", MATRIX_SCHEMA)
    if version != MATRIX_SCHEMA:
        raise SystemExit(f"{path}: unsupported schema_version {version!r} "
                         f"(expected {MATRIX_SCHEMA})")
    unknown = set(doc) - {"schema_version", "frames", "frames_dir", "defaults", "combos"}
    if unknown:
        raise SystemExit(f"{path}: unknown key(s) {', '.join(sorted(unknown))}")
    combos_doc = doc.get("combos") or []
    if not combos_doc:
        raise SystemExit(f"{path}: 'combos' is empty — nothing to compare")
    defaults = {k.replace("-", "_"): v for k, v in (doc.get("defaults") or {}).items()}
    combos = [_combo_from_dict(e, defaults, i) for i, e in enumerate(combos_doc)]
    names = [c.name for c in combos]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SystemExit(f"{path}: duplicate combo name(s): {', '.join(dupes)}")
    return Matrix(frames=doc.get("frames") or {}, combos=combos,
                  frames_dir=doc.get("frames_dir"))


def prepare_frames(frames_spec, out_dir, frame_format="png", force=False,
                   verbose=True):
    """Build the frame source, materializing it when frames are not files yet.

    Returns (source, frames_dir_or_None)."""
    source = build_frame_source(frames_spec)
    if source.has_files:
        return source, None
    frames_dir = os.path.join(out_dir, "frames")
    source = materialize(source, frames_dir, fmt=frame_format, force=force,
                         verbose=verbose)
    return source, frames_dir


def run_matrix(matrix: Matrix, out_dir, *, frame_format="png", skip_existing=False,
               model_digest=True, verbose=True, run_id=None) -> dict:
    """Run every configuration over the same frames. Returns the run manifest."""
    out_dir = os.path.realpath(out_dir)
    labels_root = os.path.join(out_dir, "labels")
    os.makedirs(labels_root, exist_ok=True)
    frames_dir = matrix.frames_dir or out_dir
    source, materialized = prepare_frames(
        matrix.frames, frames_dir, frame_format=frame_format, verbose=verbose)
    run_id = run_id or new_run_id()

    combos_out = []
    for combo in matrix.combos:
        labels_dir = os.path.join(labels_root, combo.name)
        os.makedirs(labels_dir, exist_ok=True)
        cfg = replace(combo.config, record_timing=True,
                      record_model_digest=model_digest)
        if verbose:
            print(f"\n=== combo {combo.name} ===", flush=True)
        started = perf_counter()
        pipeline = Pipeline.build(cfg, run_id=f"{run_id}-{combo.name}")
        if verbose:
            print(pipeline.describe(), flush=True)

        def already_done(frame_id, _dir=labels_dir):
            return skip_existing and os.path.exists(os.path.join(_dir, frame_id + ".json"))

        n_frames = n_signals = 0
        try:
            for frame in source.iter_frames(skip=already_done):
                payload = pipeline.run(frame)
                path = os.path.join(labels_dir, frame.frame_id + ".json")
                os.makedirs(os.path.dirname(path) or labels_dir, exist_ok=True)
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                n_frames += 1
                n_signals += len(payload["signals"])
                if verbose and n_frames % 50 == 0:
                    print(f"  {n_frames} frames, {n_signals} signals", flush=True)
        finally:
            pipeline.close()
        wall_s = perf_counter() - started
        if verbose:
            print(f"  {combo.name}: {n_frames} frames, {n_signals} signals, "
                  f"{wall_s:.1f}s", flush=True)
        combos_out.append({
            "name": combo.name,
            "labels_dir": os.path.relpath(labels_dir, out_dir),
            "config": asdict(cfg),
            "meta": pipeline.meta(),
            "frames": n_frames,
            "signals": n_signals,
            "wall_s": round(wall_s, 3),
        })

    manifest = {
        "schema_version": RUN_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "frames": source.describe(),
        "frames_dir": (os.path.relpath(materialized, out_dir)
                       if materialized else None),
        "combos": combos_out,
    }
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"\nwrote {manifest_path}")
    return manifest
