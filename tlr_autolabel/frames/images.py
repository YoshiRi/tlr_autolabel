"""Image-directory and materialized-frame sources.

`ImageDirSource` reproduces, byte for byte, the frame identity the L1 CLI has
always written: the same sorted glob order, the same `int(stem)`-or-sequence
`frame_index`, the same "first ALL-CAPS path component" channel guess, and the
same optional T4 `sample_data_token` lookup.

`MaterializedSource` reads a frame directory produced by
`tlr_autolabel.frames.cache.materialize` — the shape a video/rosbag run takes
once its frames have been written out, so every configuration in a comparison
sees identical pixels and so downstream review tools (which want real image
files) keep working.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import cv2

from tlr_autolabel.frames import IMAGE_GLOBS, Frame, FrameSource

FRAMES_INDEX = "frames.json"
FRAMES_INDEX_SCHEMA = "tlr_frames/v1"


def channel_of_relpath(rel: str) -> str | None:
    """First ALL-CAPS directory component of the relative path, e.g.
    `CAM_FRONT/00000.png` -> `CAM_FRONT`. The historical L1 rule."""
    return next((c for c in rel.split(os.sep)[:-1]
                 if c.upper() == c and not c.startswith(".")), None)


def frame_index_of(stem: str, seq: int) -> int:
    """Numeric stem when the file is numbered (T4 style), else its position."""
    return int(stem) if stem.isdigit() else seq


def load_t4_sample_data(dataset_root: str) -> dict:
    """realpath -> sample_data row, for filling `sample_data_token`/`channel`."""
    path = os.path.join(dataset_root, "annotation", "sample_data.json")
    with open(path) as f:
        return {os.path.realpath(os.path.join(dataset_root, sd["filename"])): sd
                for sd in json.load(f)}


@dataclass
class ImageDirSource(FrameSource):
    """A directory of images (or a single image file)."""

    path: str = ""
    image_root: str | None = None
    t4_dataset: str | None = None
    kind: str = field(default="images", init=False)

    def __post_init__(self):
        if not self.path:
            raise SystemExit("images source: 'path' is required")
        self.path = os.path.expanduser(self.path)
        is_dir = os.path.isdir(self.path)
        if is_dir:
            self.paths = sorted(sum([glob.glob(os.path.join(self.path, e))
                                     for e in IMAGE_GLOBS], []))
        else:
            self.paths = [self.path]
        # image paths in the JSON are relative to image_root (portable across
        # machines/containers); the realpath is kept alongside as a convenience.
        if self.t4_dataset and not self.image_root:
            root = os.path.realpath(self.t4_dataset)
        else:
            root = os.path.realpath(
                self.image_root or (self.path if is_dir
                                    else os.path.dirname(self.path) or "."))
        self.root = root
        self.t4map = load_t4_sample_data(self.t4_dataset) if self.t4_dataset else {}

    def __len__(self):
        return len(self.paths)

    def iter_frames(self, skip=None):
        for seq, p in enumerate(self.paths):
            stem = os.path.splitext(os.path.basename(p))[0]
            if skip is not None and skip(stem):
                continue
            img = cv2.imread(p)
            if img is None:
                print(f"[skip] {p}")
                continue
            rp = os.path.realpath(p)
            rel = os.path.relpath(rp, self.root)
            sd = self.t4map.get(rp)
            yield Frame(
                frame_id=stem,
                frame_index=frame_index_of(stem, seq),
                image=img,
                rel_path=rel,
                realpath=rp,
                channel=channel_of_relpath(rel),
                sample_data_token=sd["token"] if sd else None,
            )

    def describe(self) -> dict:
        return {"kind": self.kind, "uri": self.path, "image_root": self.root,
                "frames": len(self.paths)}


@dataclass
class MaterializedSource(FrameSource):
    """Frames written out by `frames.cache.materialize` (`frames.json` index)."""

    path: str = ""
    kind: str = field(default="materialized", init=False)

    def __post_init__(self):
        if not self.path:
            raise SystemExit("materialized source: 'path' is required")
        self.root = os.path.realpath(os.path.expanduser(self.path))
        index_path = os.path.join(self.root, FRAMES_INDEX)
        if not os.path.exists(index_path):
            raise SystemExit(
                f"{index_path} not found — not a materialized frame directory "
                "(use kind: images for a plain image directory)")
        with open(index_path) as f:
            index = json.load(f)
        if index.get("schema_version") != FRAMES_INDEX_SCHEMA:
            raise SystemExit(f"{index_path}: expected schema_version "
                             f"{FRAMES_INDEX_SCHEMA}, got {index.get('schema_version')!r}")
        self.index = index
        self.rows = index["frames"]

    def __len__(self):
        return len(self.rows)

    def iter_frames(self, skip=None):
        for row in self.rows:
            if skip is not None and skip(row["frame_id"]):
                continue
            file_path = os.path.join(self.root, row["file"])
            img = cv2.imread(file_path)
            if img is None:
                print(f"[skip] {file_path}")
                continue
            yield Frame(
                frame_id=row["frame_id"],
                frame_index=row["frame_index"],
                image=img,
                rel_path=row.get("rel_path") or row["file"],
                realpath=os.path.realpath(file_path),
                channel=row.get("channel"),
                sample_data_token=row.get("sample_data_token"),
                timestamp_us=row.get("timestamp_us"),
                source=row.get("source"),
            )

    def describe(self) -> dict:
        return {"kind": self.kind, "uri": self.root, "frames": len(self.rows),
                "origin": self.index.get("origin")}
