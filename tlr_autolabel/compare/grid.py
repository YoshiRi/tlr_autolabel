"""Side-by-side rendering of several configurations on the same frame.

The numbers say *how much* two configurations differ; a panel per configuration
on the frames where they differ most says *what* the difference is — a missed
distant signal, a duplicated box, a state read differently. That eyeball check
is the deliverable a naive comparison is usually asked for, so it ships with the
metrics rather than as a separate manual step.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile

import cv2
import numpy as np

STATE_COLORS = {
    "red": (0, 0, 255),
    "amber": (0, 200, 255),
    "yellow": (0, 200, 255),
    "green": (0, 200, 0),
    "white": (230, 230, 230),
    "unknown": (190, 190, 190),
}
HEADER_H = 26


def color_for_state(state: str):
    first = (state or "unknown").split(",")[0]
    return STATE_COLORS.get(first.split("-", 1)[0], STATE_COLORS["unknown"])


def resolve_image(record, image_root=None):
    """Frame image for a Tier A record: its realpath, else `image` under
    `image_root` (a materialized frame directory)."""
    rp = record.get("image_realpath")
    if rp and os.path.exists(rp):
        return rp
    rel = record.get("image")
    if rel and image_root:
        cand = os.path.join(image_root, rel)
        if os.path.exists(cand):
            return cand
    return None


def draw_panel(img, signals, title, scale=1.0, offset=(0, 0), box_thickness=2):
    """Draw boxes + labels on an already-resized panel.

    Boxes arrive in original image pixels; `scale`/`offset` map them onto the
    panel. Drawing after the resize (not before) is load-bearing: a 2880 px frame
    shrunk into an 800 px panel would otherwise reduce a 0.45-scale label and a
    2 px box to unreadable smudges."""
    vis = img.copy()
    ox, oy = offset
    for s in signals:
        x0, y0, x1, y1 = [int(round((v - o) * scale))
                          for v, o in zip(s["box"], (ox, oy, ox, oy))]
        color = color_for_state(s.get("state"))
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, box_thickness)
        score = s.get("score")
        label = s.get("state", "unknown")
        if score is not None:
            label = f"{label} {score:.2f}"
        ty = y0 - 5 if y0 - 5 > 12 else min(vis.shape[0] - 3, y1 + 15)
        cv2.putText(vis, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
    header = np.full((HEADER_H, vis.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(header, title, (6, HEADER_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([header, vis])


def detection_crop(present, shape, pad_ratio=1.5, min_size=320):
    """Region covering every configuration's detections, padded.

    Traffic lights are a fraction of a percent of a 2880x1860 frame; a
    full-frame panel shows the difference between two configurations as a few
    pixels. The crop is the union over *all* configurations, so a box only one
    of them found is inside it."""
    boxes = [s["box"] for _n, rec in present for s in rec["signals"]]
    if not boxes:
        return None
    h, w = shape[:2]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w = max(min_size / 2, (x1 - x0) * (1 + pad_ratio) / 2)
    half_h = max(min_size / 2, (y1 - y0) * (1 + pad_ratio) / 2)
    # keep the panel aspect close to the frame's so the mosaic stays regular
    half_h = max(half_h, half_w * h / w / 2)
    half_w = max(half_w, half_h * w / h / 2)
    cx0 = int(max(0, min(w - 1, cx - half_w)))
    cy0 = int(max(0, min(h - 1, cy - half_h)))
    cx1 = int(max(cx0 + 1, min(w, cx + half_w)))
    cy1 = int(max(cy0 + 1, min(h, cy + half_h)))
    return cx0, cy0, cx1, cy1


def render_frame_grid(runs, frame_key, image_root=None, width=1600, columns=None,
                      crop=False):
    """One image with a panel per run. None when the frame image is unavailable."""
    records = [(run.name, run.frames.get(frame_key)) for run in runs]
    present = [(n, r) for n, r in records if r]
    if not present:
        return None
    path = next((resolve_image(r, image_root) for _n, r in present
                 if resolve_image(r, image_root)), None)
    if path is None:
        return None
    img = cv2.imread(path)
    if img is None:
        return None

    offset = (0, 0)
    region = detection_crop(present, img.shape) if crop else None
    if region:
        cx0, cy0, cx1, cy1 = region
        img = img[cy0:cy1, cx0:cx1]
        offset = (cx0, cy0)

    columns = columns or (2 if len(present) > 2 else len(present))
    rows = math.ceil(len(present) / columns)
    panel_w = max(1, width // columns)
    scale = panel_w / img.shape[1]
    resized = cv2.resize(img, (panel_w, max(1, int(round(img.shape[0] * scale)))))
    panels = []
    for name, rec in present:
        signals = rec["signals"]
        title = f"{name} | {frame_key} | {len(signals)} det"
        panels.append(draw_panel(resized, signals, title, scale=scale, offset=offset))
    blank = np.zeros_like(panels[0])
    while len(panels) < rows * columns:
        panels.append(blank)
    return np.vstack([np.hstack(panels[r * columns:(r + 1) * columns])
                      for r in range(rows)])


def render_grids(runs, frame_keys, out_dir, image_root=None, width=1600,
                 columns=None, crop=False) -> list:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for key in frame_keys:
        grid = render_frame_grid(runs, key, image_root=image_root, width=width,
                                 columns=columns, crop=crop)
        if grid is None:
            continue
        path = os.path.join(out_dir, key.replace(os.sep, "_") + ".grid.png")
        cv2.imwrite(path, grid)
        written.append(path)
    return written


def render_grid_video(runs, frame_keys, out_path, image_root=None, width=1600,
                      columns=None, crop=False, fps=10) -> str | None:
    """Encode the grids as an mp4 (needs ffmpeg). Returns None if nothing rendered.

    Panels must all come out the same size for the encoder, so cropping is not
    applied here even when asked for."""
    with tempfile.TemporaryDirectory(prefix="tlr_grid_") as tmp:
        n = 0
        for key in frame_keys:
            grid = render_frame_grid(runs, key, image_root=image_root, width=width,
                                     columns=columns, crop=False)
            if grid is None:
                continue
            cv2.imwrite(os.path.join(tmp, f"{n:06d}.png"), grid)
            n += 1
        if not n:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
               "-i", os.path.join(tmp, "%06d.png"), "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_path]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print("ffmpeg not found — wrote no video (the PNG grids are still there)")
            return None
    return out_path
