"""Video frame source (mp4 / avi / anything OpenCV can open).

`frame_id` is the *source* frame number zero-padded to 6 digits, so ids are
stable no matter what `stride` a run used: comparing a stride-2 run against a
stride-1 run lines up on the frames they share instead of silently comparing
frame 10 with frame 20.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2

from tlr_autolabel.frames import Frame, FrameSource


@dataclass
class VideoSource(FrameSource):
    uri: str = ""
    stride: int = 1
    start: int = 0
    max_frames: int | None = None
    channel: str | None = None
    kind: str = field(default="video", init=False)
    has_files = False

    def __post_init__(self):
        if not self.uri:
            raise SystemExit("video source: 'uri' is required")
        self.uri = os.path.expanduser(self.uri)
        if not os.path.exists(self.uri):
            raise SystemExit(f"video not found: {self.uri}")
        if self.stride < 1:
            raise SystemExit(f"video source: stride must be >= 1, got {self.stride}")
        self.name = os.path.basename(self.uri)

    def _open(self):
        cap = cv2.VideoCapture(self.uri)
        if not cap.isOpened():
            raise SystemExit(f"cannot open video: {self.uri} "
                             "(OpenCV has no decoder for it?)")
        return cap

    def iter_frames(self, skip=None):
        cap = self._open()
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        try:
            number = -1
            emitted = 0
            while True:
                if self.max_frames is not None and emitted >= self.max_frames:
                    break
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if not cap.grab():
                    break
                number += 1
                if number < self.start or (number - self.start) % self.stride:
                    continue
                frame_id = f"{number:06d}"
                if skip is not None and skip(frame_id):
                    emitted += 1
                    continue
                ok, img = cap.retrieve()
                if not ok or img is None:
                    print(f"[skip] {self.name}#{frame_id} (decode failed)")
                    continue
                if pos_ms and pos_ms > 0:
                    ts_us = int(round(pos_ms * 1000.0))
                elif fps > 0:
                    ts_us = int(round(number / fps * 1e6))
                else:
                    ts_us = None
                emitted += 1
                yield Frame(
                    frame_id=frame_id,
                    frame_index=number,
                    image=img,
                    rel_path=f"{self.name}#{frame_id}",
                    realpath=None,
                    channel=self.channel,
                    timestamp_us=ts_us,
                    source={"kind": "video", "uri": self.uri,
                            "frame_number": number, "stride": self.stride,
                            "fps": round(fps, 4) if fps else None},
                )
        finally:
            cap.release()

    def describe(self) -> dict:
        cap = self._open()
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        finally:
            cap.release()
        return {"kind": self.kind, "uri": self.uri, "stride": self.stride,
                "start": self.start, "max_frames": self.max_frames,
                "video_frames": total, "fps": round(fps, 4) if fps else None}
