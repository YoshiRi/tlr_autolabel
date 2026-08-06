"""Materialize a frame source to image files + a `frames.json` index.

Two reasons a comparison run wants this, both load-bearing:

1. **Identical pixels for every configuration.** Decoding a video or a bag once
   per configuration is not guaranteed to produce identical arrays (seeking,
   codec threading, dropped frames), and a comparison that cannot tell a model
   difference from a decode difference is worthless. Extract once, then every
   configuration reads the same files.
2. **Downstream tools want real images.** The review video / CVAT export /
   timeline renderers resolve `image_realpath`; a synthetic `bag:/topic#123`
   reference is not something they can open.
"""
from __future__ import annotations

import json
import os

import cv2

from tlr_autolabel.frames.images import (
    FRAMES_INDEX, FRAMES_INDEX_SCHEMA, MaterializedSource,
)

EXT_PARAMS = {
    "png": [cv2.IMWRITE_PNG_COMPRESSION, 3],
}


def materialize(source, out_dir, fmt="png", jpeg_quality=95, force=False,
                verbose=True) -> MaterializedSource:
    """Write every frame of `source` under `out_dir` and return a source over
    the result. Reuses an existing extraction whose `origin` matches, so a
    second comparison run over the same bag does not re-decode it."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in ("png", "jpg", "jpeg"):
        raise SystemExit(f"unsupported frame format {fmt!r} (png | jpg)")
    origin = source.describe()
    index_path = os.path.join(out_dir, FRAMES_INDEX)
    if not force and os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f)
        if existing.get("origin") == origin and existing.get("format") == fmt and \
                all(os.path.exists(os.path.join(out_dir, r["file"]))
                    for r in existing.get("frames", [])):
            if verbose:
                print(f"reusing {len(existing['frames'])} extracted frames in {out_dir}")
            return MaterializedSource(path=out_dir)

    os.makedirs(out_dir, exist_ok=True)
    params = EXT_PARAMS.get(fmt, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    rows = []
    for frame in source.iter_frames():
        rel_file = f"{frame.frame_id}.{fmt}"
        path = os.path.join(out_dir, rel_file)
        os.makedirs(os.path.dirname(path) or out_dir, exist_ok=True)
        if not cv2.imwrite(path, frame.image, params):
            raise SystemExit(f"failed to write frame {path}")
        src = dict(frame.source or {})
        src.setdefault("kind", source.kind)
        src["origin_ref"] = frame.rel_path
        rows.append({
            "frame_id": frame.frame_id,
            "frame_index": frame.frame_index,
            "file": rel_file,
            "rel_path": rel_file,
            "channel": frame.channel,
            "sample_data_token": frame.sample_data_token,
            "timestamp_us": frame.timestamp_us,
            "source": src,
        })
    with open(index_path, "w") as f:
        json.dump({"schema_version": FRAMES_INDEX_SCHEMA, "format": fmt,
                   "origin": origin, "frames": rows}, f, indent=2)
    if verbose:
        print(f"extracted {len(rows)} frames -> {out_dir}")
    return MaterializedSource(path=out_dir)
