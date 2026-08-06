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


def draw_panel(img, signals, title, box_thickness=2):
    vis = img.copy()
    for s in signals:
        x0, y0, x1, y1 = [int(round(v)) for v in s["box"]]
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
    cv2.putText(header, title, (6, HEADER_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([header, vis])


def render_frame_grid(runs, frame_key, image_root=None, width=1600, columns=None):
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

    columns = columns or (2 if len(present) > 2 else len(present))
    rows = math.ceil(len(present) / columns)
    panel_w = max(1, width // columns)
    scale = panel_w / img.shape[1]
    panels = []
    for name, rec in present:
        signals = rec["signals"]
        title = f"{name}  |  {frame_key}  |  {len(signals)} det"
        panel = draw_panel(img, signals, title)
        panels.append(cv2.resize(panel, (panel_w, int(round(panel.shape[0] * scale)))))
    blank = np.zeros_like(panels[0])
    while len(panels) < rows * columns:
        panels.append(blank)
    return np.vstack([np.hstack(panels[r * columns:(r + 1) * columns])
                      for r in range(rows)])


def render_grids(runs, frame_keys, out_dir, image_root=None, width=1600,
                 columns=None) -> list:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for key in frame_keys:
        grid = render_frame_grid(runs, key, image_root=image_root, width=width,
                                 columns=columns)
        if grid is None:
            continue
        path = os.path.join(out_dir, key.replace(os.sep, "_") + ".grid.png")
        cv2.imwrite(path, grid)
        written.append(path)
    return written


def render_grid_video(runs, frame_keys, out_path, image_root=None, width=1600,
                      columns=None, fps=10) -> str | None:
    """Encode the grids as an mp4 (needs ffmpeg). Returns None if nothing rendered."""
    with tempfile.TemporaryDirectory(prefix="tlr_grid_") as tmp:
        n = 0
        for key in frame_keys:
            grid = render_frame_grid(runs, key, image_root=image_root, width=width,
                                     columns=columns)
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
