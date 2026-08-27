"""Rosbag2 camera-image frame source (mcap or sqlite3).

This is a *reader*, not a node: it does not import anything from
`ros2_pipeline/` and nothing in L1-L4 imports it unless a run actually asks for
`kind: rosbag`, so the "L5 is out of the dependency graph" rule still holds.
`rosbag2_py` / `rclpy` are imported lazily for the same reason — the package
must stay importable on a machine without a sourced ROS 2.

Images are decoded with numpy/cv2 rather than `cv_bridge`, deliberately:
ros2_pipeline/README.md documents that cv_bridge breaks against a NumPy 2.x in
the user site, and a frame source has no reason to inherit that fragility.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from tlr_autolabel.frames import Frame, FrameSource

IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")

# encoding -> (channels, dtype); what `sensor_msgs/Image` layouts we can reshape
# without cv_bridge. Anything else is an explicit error rather than a guess.
RAW_ENCODINGS = {
    "bgr8": (3, np.uint8), "rgb8": (3, np.uint8),
    "bgra8": (4, np.uint8), "rgba8": (4, np.uint8),
    "mono8": (1, np.uint8), "8UC1": (1, np.uint8), "8UC3": (3, np.uint8),
    "mono16": (1, np.uint16), "16UC1": (1, np.uint16),
}


def channel_of_topic(topic: str) -> str:
    """`/sensing/camera/camera6/image_rect_color` -> `camera6`.

    The segment before the image topic name is the camera identity in every
    Autoware camera graph; fall back to the last segment."""
    parts = [p for p in topic.strip("/").split("/") if p]
    if not parts:
        return "image"
    if len(parts) > 1 and parts[-1].startswith("image"):
        return parts[-2]
    return parts[-1]


def decode_image_msg(msg, msg_type: str) -> np.ndarray | None:
    """ROS image message -> BGR ndarray (what cv2.imread would have given)."""
    if msg_type.endswith("CompressedImage"):
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    enc = msg.encoding
    if enc not in RAW_ENCODINGS:
        raise SystemExit(
            f"unsupported image encoding {enc!r}; supported: "
            f"{', '.join(sorted(RAW_ENCODINGS))} (+ any CompressedImage). "
            "Republish the topic as bgr8/rgb8 or use a compressed topic.")
    nch, dtype = RAW_ENCODINGS[enc]
    arr = np.frombuffer(bytes(msg.data), dtype=dtype)
    expected = msg.height * msg.width * nch
    if arr.size < expected:
        return None
    arr = arr[:expected].reshape(msg.height, msg.width, nch)
    if dtype == np.uint16:
        arr = cv2.convertScaleAbs(arr, alpha=255.0 / 65535.0)
    if nch == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if enc.startswith("rgb"):
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if nch == 3 else cv2.COLOR_RGBA2BGR)
    if nch == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(arr)


@dataclass
class RosbagSource(FrameSource):
    uri: str = ""
    topics: list[str] | None = None          # default: every image topic in the bag
    channel_map: dict | None = None          # topic -> channel override
    stride: int = 1
    max_frames: int | None = None            # per topic
    storage_id: str | None = None
    kind: str = field(default="rosbag", init=False)
    has_files = False

    def __post_init__(self):
        if not self.uri:
            raise SystemExit("rosbag source: 'uri' is required")
        self.uri = os.path.expanduser(self.uri)
        if not os.path.exists(self.uri):
            raise SystemExit(f"bag not found: {self.uri}")
        if isinstance(self.topics, str):
            self.topics = [t.strip() for t in self.topics.split(",") if t.strip()]
        if self.stride < 1:
            raise SystemExit(f"rosbag source: stride must be >= 1, got {self.stride}")
        self.name = os.path.basename(self.uri.rstrip("/"))
        self._typemap = None

    def _storage(self):
        if self.storage_id:
            return self.storage_id
        return "mcap" if self.uri.endswith(".mcap") else "sqlite3"

    def _reader(self):
        try:
            import rosbag2_py
        except ImportError as exc:  # pragma: no cover - needs a sourced ROS 2
            raise SystemExit(
                "rosbag2_py is not importable — source ROS 2 first:\n"
                "  source /opt/ros/humble/setup.bash") from exc
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=self.uri, storage_id=self._storage()),
                    rosbag2_py.ConverterOptions("", ""))
        self._typemap = {t.name: t.type for t in reader.get_all_topics_and_types()}
        return reader

    def image_topics(self) -> dict:
        """{topic: type} for the image topics this source will read."""
        if self._typemap is None:
            self._reader()
        available = {t: ty for t, ty in self._typemap.items() if ty in IMAGE_TYPES}
        if not self.topics:
            if not available:
                raise SystemExit(
                    f"no sensor_msgs/Image or CompressedImage topics in {self.uri}; "
                    f"topics present: {', '.join(sorted(self._typemap))}")
            return available
        missing = [t for t in self.topics if t not in self._typemap]
        if missing:
            raise SystemExit(f"topic(s) not in bag: {', '.join(missing)}; "
                             f"image topics available: {', '.join(sorted(available))}")
        return {t: self._typemap[t] for t in self.topics}

    def iter_frames(self, skip=None):
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        reader = self._reader()
        wanted = self.image_topics()
        multi = len(wanted) > 1
        channel_map = dict(self.channel_map or {})
        counts = {t: -1 for t in wanted}
        emitted = {t: 0 for t in wanted}
        while reader.has_next():
            topic, data, bag_stamp_ns = reader.read_next()
            if topic not in wanted:
                continue
            counts[topic] += 1
            number = counts[topic]
            if number % self.stride:
                continue
            if self.max_frames is not None and emitted[topic] >= self.max_frames:
                if all(self.max_frames is not None and emitted[t] >= self.max_frames
                       for t in wanted):
                    break
                continue
            channel = channel_map.get(topic) or channel_of_topic(topic)
            frame_id = f"{channel}/{number:06d}" if multi else f"{number:06d}"
            if skip is not None and skip(frame_id):
                emitted[topic] += 1
                continue
            msg = deserialize_message(data, get_message(wanted[topic]))
            img = decode_image_msg(msg, wanted[topic])
            if img is None:
                print(f"[skip] {topic}#{number:06d} (decode failed)")
                continue
            stamp = getattr(msg, "header", None)
            if stamp is not None:
                stamp_ns = stamp.stamp.sec * 1_000_000_000 + stamp.stamp.nanosec
            else:  # pragma: no cover - all image msgs carry a header
                stamp_ns = bag_stamp_ns
            emitted[topic] += 1
            yield Frame(
                frame_id=frame_id,
                frame_index=number,
                image=img,
                rel_path=f"{self.name}{topic}/{number:06d}",
                realpath=None,
                channel=channel,
                timestamp_us=stamp_ns // 1000,
                source={"kind": "rosbag", "uri": self.uri, "topic": topic,
                        "message_index": number, "stride": self.stride,
                        "stamp_ns": stamp_ns, "bag_stamp_ns": bag_stamp_ns},
            )

    def describe(self) -> dict:
        return {"kind": self.kind, "uri": self.uri,
                "topics": sorted(self.image_topics()), "stride": self.stride,
                "max_frames": self.max_frames, "storage_id": self._storage()}
