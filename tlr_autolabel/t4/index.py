"""T4 camera sample_data indexing (REFACTOR_PLAN.md phase 5).

Extracted from match_traffic_lights.py.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_t4_index(root: Path):
    """Index camera sample_data rows by (channel, filename stem)."""
    ann = root / "annotation"
    ego_by_token = {r["token"]: r for r in json.loads((ann / "ego_pose.json").read_text())}
    calib_by_token = {r["token"]: r for r in json.loads((ann / "calibrated_sensor.json").read_text())}
    sensor_by_token = {r["token"]: r for r in json.loads((ann / "sensor.json").read_text())}

    frames: dict[tuple[str, str], dict] = {}
    frames_by_token: dict[str, dict] = {}
    for row in json.loads((ann / "sample_data.json").read_text()):
        calib = calib_by_token[row["calibrated_sensor_token"]]
        sensor = sensor_by_token[calib["sensor_token"]]
        if sensor["modality"] != "camera":
            continue
        stem = Path(row["filename"]).stem
        entry = {
            "sample_data": row,
            "ego_pose": ego_by_token[row["ego_pose_token"]],
            "calib": calib,
            "channel": sensor["channel"],
        }
        frames[(sensor["channel"], stem)] = entry
        frames_by_token[row["token"]] = entry
    return frames, frames_by_token
