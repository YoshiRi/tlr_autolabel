"""Lightweight T4-format dataset reader.

A self-contained, dependency-free replacement for the ad hoc per-script
`json.load(open(annotation/...))` calls scattered across this repo. Reads
the standard T4 annotation tables into token-indexed dicts and exposes the
few derived views (camera frame index, channel lookups) that every L3+
consumer needs, without pulling in t4devkit.

Every table is optional at the file level except `sample_data.json`,
`calibrated_sensor.json`, and `sensor.json` (needed to resolve which
sample_data rows are cameras and what channel they belong to). Missing
optional tables load as an empty dict/list rather than raising, since a
derived/partial dataset (e.g. a B-only export) may not carry them all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED_TABLES = ("sample_data", "calibrated_sensor", "sensor")
INDEXED_TABLES = (
    "sample",
    "sample_data",
    "ego_pose",
    "calibrated_sensor",
    "sensor",
    "scene",
    "log",
    "map",
    "category",
    "attribute",
    "instance",
    "visibility",
)
LIST_TABLES = ("object_ann", "sample_annotation", "surface_ann")


@dataclass
class CameraFrame:
    sample_data: dict
    ego_pose: dict
    calibrated_sensor: dict
    channel: str

    @property
    def token(self) -> str:
        return self.sample_data["token"]

    @property
    def timestamp(self) -> int:
        return self.sample_data.get("timestamp", 0)

    @property
    def stem(self) -> str:
        return Path(self.sample_data["filename"]).stem


@dataclass
class T4Dataset:
    """Token-indexed view of a T4-format `annotation/` directory.

    `by_token(table)` gives {token: row} for any of INDEXED_TABLES;
    `rows(table)` gives the raw list for any of LIST_TABLES. Camera frames
    are pre-joined (sample_data + ego_pose + calibrated_sensor + channel)
    since nearly every L3+ tool needs that join.
    """

    root: Path
    tables: dict[str, dict[str, dict]] = field(default_factory=dict)
    lists: dict[str, list[dict]] = field(default_factory=dict)
    camera_frames_by_token: dict[str, CameraFrame] = field(default_factory=dict)
    camera_frames_by_channel_stem: dict[tuple[str, str], CameraFrame] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> "T4Dataset":
        root = Path(root)
        ann_dir = root / "annotation"
        missing_required = [
            t for t in REQUIRED_TABLES if not (ann_dir / f"{t}.json").exists()
        ]
        if missing_required:
            raise FileNotFoundError(
                f"{ann_dir} is missing required table(s): "
                + ", ".join(f"{t}.json" for t in missing_required)
            )

        tables: dict[str, dict[str, dict]] = {}
        for name in INDEXED_TABLES:
            rows = _load_json_list(ann_dir / f"{name}.json")
            tables[name] = {row["token"]: row for row in rows}

        lists: dict[str, list[dict]] = {}
        for name in LIST_TABLES:
            lists[name] = _load_json_list(ann_dir / f"{name}.json")

        ds = cls(root=root, tables=tables, lists=lists)
        ds._build_camera_frames()
        return ds

    def _build_camera_frames(self) -> None:
        sensor_by_calibrated = {
            calib["token"]: self.tables["sensor"].get(calib["sensor_token"], {})
            for calib in self.tables["calibrated_sensor"].values()
        }
        for row in self.tables["sample_data"].values():
            calib = self.tables["calibrated_sensor"].get(row["calibrated_sensor_token"])
            if calib is None:
                continue
            sensor = sensor_by_calibrated.get(calib["token"], {})
            if sensor.get("modality") != "camera":
                continue
            frame = CameraFrame(
                sample_data=row,
                ego_pose=self.tables["ego_pose"].get(row.get("ego_pose_token"), {}),
                calibrated_sensor=calib,
                channel=sensor.get("channel", ""),
            )
            self.camera_frames_by_token[frame.token] = frame
            self.camera_frames_by_channel_stem[(frame.channel, frame.stem)] = frame

    def by_token(self, table: str) -> dict[str, dict]:
        if table not in INDEXED_TABLES:
            raise ValueError(f"{table!r} is not a token-indexed table; use rows() instead")
        return self.tables[table]

    def rows(self, table: str) -> list[dict]:
        if table in LIST_TABLES:
            return self.lists[table]
        if table in INDEXED_TABLES:
            return list(self.tables[table].values())
        raise ValueError(f"unknown table {table!r}")

    def channel_of(self, sample_data_token: str) -> str:
        frame = self.camera_frames_by_token.get(sample_data_token)
        return frame.channel if frame else ""

    def timestamp_of_sample(self, sample_token: str) -> int | None:
        row = self.tables["sample"].get(sample_token)
        return row.get("timestamp") if row else None

    def scene_name(self) -> str:
        scenes = list(self.tables["scene"].values())
        return scenes[0].get("name") or "scene" if scenes else "scene"


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())
