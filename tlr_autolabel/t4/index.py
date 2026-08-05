"""T4 camera sample_data indexing (REFACTOR_PLAN.md phase 5).

Extracted from match_traffic_lights.py; now backed by the shared
tlr_autolabel/t4/dataset.py reader.
"""
from __future__ import annotations

from pathlib import Path

from tlr_autolabel.t4.dataset import T4Dataset


def load_t4_index(root: Path):
    """Index camera sample_data rows by (channel, filename stem).

    Returns the same {"sample_data", "ego_pose", "calib", "channel"} shape
    as before the T4Dataset migration, so existing callers (map/projection.py's
    project_traffic_lights) don't need to change.
    """
    ds = T4Dataset.load(root)
    frames: dict[tuple[str, str], dict] = {}
    frames_by_token: dict[str, dict] = {}
    for frame in ds.camera_frames_by_token.values():
        entry = {
            "sample_data": frame.sample_data,
            "ego_pose": frame.ego_pose,
            "calib": frame.calibrated_sensor,
            "channel": frame.channel,
        }
        frames[(frame.channel, frame.stem)] = entry
        frames_by_token[frame.token] = entry
    return frames, frames_by_token
