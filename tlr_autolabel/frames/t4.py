"""T4 dataset frame source.

Reads the camera `sample_data` rows straight out of the annotation tables, so a
comparison run over a T4 dataset carries `sample_data_token` and `channel` for
free — which is what lets its Tier A output flow into the existing L3 (map
matching) and L6 (eval) tools unchanged.

Multi-channel runs use `<channel>/<stem>` as the frame id so that per-camera
labels land in the per-camera layout `run_dataset.py` already produces.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field

import cv2

from tlr_autolabel.frames import Frame, FrameSource
from tlr_autolabel.frames.images import frame_index_of


def camera_sample_data(dataset_root: str, channels=None) -> list[dict]:
    """Camera `sample_data` rows sorted by (channel, timestamp), each with its
    resolved `channel`."""
    ann = os.path.join(dataset_root, "annotation")
    with open(os.path.join(ann, "calibrated_sensor.json")) as f:
        calib = {r["token"]: r for r in json.load(f)}
    with open(os.path.join(ann, "sensor.json")) as f:
        sensor = {r["token"]: r for r in json.load(f)}
    with open(os.path.join(ann, "sample_data.json")) as f:
        sample_data = json.load(f)
    want = set(channels) if channels else None
    rows = []
    for sd in sample_data:
        cal = calib.get(sd.get("calibrated_sensor_token"))
        sen = sensor.get(cal["sensor_token"]) if cal else None
        if not sen or sen.get("modality") != "camera":
            continue
        channel = sen.get("channel")
        if want and channel not in want:
            continue
        rows.append({**sd, "channel": channel})
    rows.sort(key=lambda r: (r["channel"] or "", r.get("timestamp") or 0,
                             r.get("filename") or ""))
    return rows


@dataclass
class T4DatasetSource(FrameSource):
    root: str = ""
    channels: list[str] | None = None
    stride: int = 1
    max_frames: int | None = None            # per channel
    kind: str = field(default="t4", init=False)

    def __post_init__(self):
        if not self.root:
            raise SystemExit("t4 source: 'root' is required")
        if self.stride < 1:
            raise SystemExit(f"t4 source: stride must be >= 1, got {self.stride}")
        if isinstance(self.channels, str):
            self.channels = [c.strip() for c in self.channels.split(",") if c.strip()]
        self.root = os.path.realpath(os.path.expanduser(self.root))
        if not os.path.isdir(os.path.join(self.root, "annotation")):
            raise SystemExit(f"not a T4 dataset (no annotation/): {self.root}")
        self.rows = camera_sample_data(self.root, self.channels)
        self.multi_channel = len({r["channel"] for r in self.rows}) > 1

    def __len__(self):
        return len(self.rows)

    def iter_frames(self, skip=None):
        # subsampling is per channel, so `--max-frames 5` on a 6-camera dataset
        # gives 5 frames of each camera rather than 5 frames of the first one
        seen = defaultdict(int)
        emitted = defaultdict(int)
        for seq, sd in enumerate(self.rows):
            channel = sd["channel"]
            index = seen[channel]
            seen[channel] += 1
            if index % self.stride:
                continue
            if self.max_frames is not None and emitted[channel] >= self.max_frames:
                continue
            rel = sd["filename"]
            stem = os.path.splitext(os.path.basename(rel))[0]
            frame_id = f"{channel}/{stem}" if self.multi_channel else stem
            if skip is not None and skip(frame_id):
                emitted[channel] += 1
                continue
            path = os.path.join(self.root, rel)
            img = cv2.imread(path)
            if img is None:
                print(f"[skip] {path}")
                continue
            emitted[channel] += 1
            yield Frame(
                frame_id=frame_id,
                frame_index=frame_index_of(stem, seq),
                image=img,
                rel_path=rel,
                realpath=os.path.realpath(path),
                channel=sd["channel"],
                sample_data_token=sd.get("token"),
                timestamp_us=sd.get("timestamp"),
            )

    def describe(self) -> dict:
        return {"kind": self.kind, "uri": self.root,
                "channels": sorted({r["channel"] for r in self.rows}),
                "frames": len(self.rows), "stride": self.stride,
                "max_frames": self.max_frames}
