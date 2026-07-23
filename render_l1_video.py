#!/usr/bin/env python3
"""Render tlr_autolabel/v1 detections as an MP4 review video."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2


STATE_COLORS = {
    "red": (0, 0, 255),
    "amber": (0, 200, 255),
    "yellow": (0, 200, 255),
    "green": (0, 200, 0),
    "unknown": (210, 210, 210),
}


def color_for_state(state: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in state.split(",") if p.strip()]
    if not parts:
        return STATE_COLORS["unknown"]
    first = parts[0]
    prefix = first.split("-", 1)[0]
    return STATE_COLORS.get(prefix, STATE_COLORS["unknown"])


def text_with_bg(img, text: str, xy: tuple[int, int], color: tuple[int, int, int], scale: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(round(scale * 1.3)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = xy
    y0 = max(0, y - th - base - 4)
    x1 = min(img.shape[1] - 1, x + tw + 6)
    y1 = min(img.shape[0] - 1, y + base + 2)
    cv2.rectangle(img, (x, y0), (x1, y1), (20, 20, 20), -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, color, thickness, cv2.LINE_AA)


def iter_label_files(labels_dir: Path):
    files = sorted(labels_dir.glob("*.json"))
    for path in files:
        yield path


def resolve_image_path(label: dict, dataset_root: Path | None) -> Path:
    realpath = label.get("image_realpath")
    if realpath:
        return Path(realpath)
    image = label.get("image")
    if not image:
        raise ValueError("label has neither image_realpath nor image")
    if dataset_root is None:
        return Path(image)
    return dataset_root / image


def draw_frame(img, label: dict, width: int, min_score: float) -> tuple:
    h0, w0 = img.shape[:2]
    width = width if width % 2 == 0 else width - 1
    scale = width / float(w0)
    height = int(round(h0 * scale))
    height = height if height % 2 == 0 else height - 1
    vis = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    line_w = max(2, int(round(scale * 4)))
    font_scale = max(0.42, min(0.7, scale * 1.35))

    signals = []
    for signal in label.get("signals", []):
        if float(signal.get("detector_score", 0.0)) < min_score:
            continue
        signals.append(signal)

    state_counts = Counter()
    for signal in signals:
        state = str(signal.get("state") or "unknown")
        state_counts[state] += 1
        box = signal.get("box_xyxy") or signal.get("bbox")
        if not box or len(box) != 4:
            continue
        x0, y0, x1, y1 = [int(round(float(v) * scale)) for v in box]
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))
        color = color_for_state(state)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, line_w)
        score = float(signal.get("detector_score", 0.0))
        label_text = f"{state} {score:.2f}"
        ty = y0 - 6 if y0 > 24 else min(height - 4, y1 + 18)
        text_with_bg(vis, label_text, (x0, ty), color, font_scale)

    frame_index = label.get("frame_index")
    channel = label.get("channel") or ""
    name = Path(label.get("image") or "").name
    summary = " ".join(f"{k}={v}" for k, v in state_counts.most_common(4))
    header = f"{channel} frame={frame_index} {name} detections={len(signals)} {summary}".strip()
    cv2.rectangle(vis, (0, 0), (width, 28), (25, 25, 25), -1)
    cv2.putText(vis, header[:150], (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis, len(signals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", required=True, type=Path,
                        help="Directory containing tlr_autolabel/v1 JSON files.")
    parser.add_argument("--dataset-root", type=Path,
                        help="Dataset root used to resolve relative image paths.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--crf", type=int, default=26)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    label_files = list(iter_label_files(args.labels_dir))
    if not label_files:
        raise SystemExit(f"no JSON labels found under {args.labels_dir}")

    total_detections = 0
    with tempfile.TemporaryDirectory(prefix="tlr_l1_video_") as tmp:
        tmp_path = Path(tmp)
        frame_count = 0
        for label_path in label_files:
            with label_path.open() as f:
                label = json.load(f)
            img_path = resolve_image_path(label, args.dataset_root)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[skip] unreadable image: {img_path}")
                continue
            vis, n = draw_frame(img, label, args.width, args.min_score)
            total_detections += n
            cv2.imwrite(str(tmp_path / f"{frame_count:06d}.jpg"), vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            frame_count += 1

        if frame_count == 0:
            raise SystemExit("no frames rendered")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", str(tmp_path / "%06d.jpg"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(args.crf),
            str(args.output),
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"wrote {args.output} frames={frame_count} detections={total_detections} size={size_mb:.1f}MB")


if __name__ == "__main__":
    main()
