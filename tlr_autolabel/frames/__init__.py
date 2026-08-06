"""Frame sources: what L1 inference reads, decoupled from where it came from.

L1 used to be hard-wired to "a directory of images" (a sorted glob inside
`cli/autolabel.py:main`), which made the frame identity written into Tier A
(`image`, `frame_index`, `channel`, `sample_data_token`) a property of the file
layout. A `FrameSource` makes that identity the source's job, so a video, a
rosbag, or a T4 dataset can feed the exact same pipeline.

Sources yield `Frame`s. `frame_id` is the join key: it names the Tier A JSON
file and is what the comparison layer uses to line up two runs frame by frame.
It may contain `/` (T4 / multi-topic sources use `<channel>/<stem>`); writers
create the parent directory.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


@dataclass
class Frame:
    """One image plus the identity that ends up in the Tier A payload."""

    frame_id: str                     # Tier A basename + comparison join key
    frame_index: int                  # payload `frame_index`
    image: np.ndarray | None          # BGR, as cv2.imread gives it
    rel_path: str                     # payload `image` (portable reference)
    realpath: str | None = None       # payload `image_realpath` (None: not a file)
    channel: str | None = None
    sample_data_token: str | None = None
    timestamp_us: int | None = None
    source: dict | None = None        # payload `source` block (non-file sources)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


class FrameSource:
    """Base class. Subclasses implement `iter_frames(skip=None)`.

    `skip(frame_id) -> bool` is consulted *before* the image is decoded, so
    `--skip-existing` stays as cheap as it was when the loop owned the glob.

    `has_files` says whether frames already exist as image files on disk. When
    they do not (video, rosbag), a comparison run materializes them first — see
    `tlr_autolabel.frames.cache`.
    """

    kind = ""
    has_files = True

    def iter_frames(self, skip=None):
        raise NotImplementedError

    def __iter__(self):
        return self.iter_frames()

    def describe(self) -> dict:
        return {"kind": self.kind}


def build_frame_source(spec: dict) -> FrameSource:
    """Build a source from a plain dict (the `frames:` block of a comparison
    matrix, or CLI flags mapped onto the same keys)."""
    spec = dict(spec or {})
    kind = spec.pop("kind", None)
    if not kind:
        raise SystemExit("frame source: 'kind' is required "
                         "(images | video | rosbag | t4 | materialized)")
    builders = {
        "images": _images,
        "video": _video,
        "rosbag": _rosbag,
        "t4": _t4,
        "materialized": _materialized,
    }
    if kind not in builders:
        raise SystemExit(f"unknown frame source kind {kind!r}; "
                         f"available: {', '.join(sorted(builders))}")
    return builders[kind](spec)


def _images(spec):
    from tlr_autolabel.frames.images import ImageDirSource
    return ImageDirSource(**spec)


def _video(spec):
    from tlr_autolabel.frames.video import VideoSource
    return VideoSource(**spec)


def _rosbag(spec):
    from tlr_autolabel.frames.rosbag import RosbagSource
    return RosbagSource(**spec)


def _t4(spec):
    from tlr_autolabel.frames.t4 import T4DatasetSource
    return T4DatasetSource(**spec)


def _materialized(spec):
    from tlr_autolabel.frames.images import MaterializedSource
    return MaterializedSource(**spec)
