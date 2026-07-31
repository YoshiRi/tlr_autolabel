"""Camera pose math and lanelet2-traffic-light-to-image projection
(REFACTOR_PLAN.md phase 5). Extracted from match_traffic_lights.py.
"""
from __future__ import annotations

import numpy as np


def quat_to_rot(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def pose_to_matrix(translation, rotation_wxyz) -> np.ndarray:
    mat = np.eye(4)
    mat[:3, :3] = quat_to_rot(rotation_wxyz)
    mat[:3, 3] = translation
    return mat


def project_traffic_lights(frame, traffic_lights, max_distance, image_wh, margin=100.0,
                           max_incidence_deg=75.0, readable_incidence_deg=60.0):
    """Project all map traffic lights into this frame -> list of candidates.

    Facing classification per candidate (incidence = angle between the signed
    face normal and the sight line, horizontal plane):
      "front"        <= readable_incidence_deg -- lamps read cleanly
      "front_oblique" readable_incidence_deg .. max_incidence_deg -- still the
                     front face but steeply angled, hard to read (empirical
                     matched-rate collapses here: 12% at 70-80, 6% at 80-90).
                     Kept but flagged so L2 can mark it occlusion_state.partial.
      (edge-on)      max_incidence_deg .. 180-max_incidence_deg -- lamps
                     unreadable and box degenerate; candidate dropped.
      "back"         >= 180 - max_incidence_deg -- the housing's back.
    """
    ego = frame["ego_pose"]
    calib = frame["calib"]
    t_map_base = pose_to_matrix(ego["translation"], ego["rotation"])
    t_base_cam = pose_to_matrix(calib["translation"], calib["rotation"])
    t_cam_map = np.linalg.inv(t_map_base @ t_base_cam)
    intrinsic = np.array(calib["camera_intrinsic"])
    width, height = image_wh
    ego_xy = np.array(ego["translation"][:2])
    cos_max = np.cos(np.radians(max_incidence_deg))
    cos_readable = np.cos(np.radians(readable_incidence_deg))

    candidates = []
    for way_id, tl in traffic_lights.items():
        center = tl["corners"].mean(axis=0)
        distance = float(np.linalg.norm(center[:2] - ego_xy))
        if distance > max_distance:
            continue
        facing = ""
        facing_deg = None
        if tl["facing_axis"] is not None and distance > 1e-6:
            sight = (ego_xy - center[:2]) / distance
            cos_face = float(np.dot(tl["facing_axis"], sight))
            facing_deg = float(np.degrees(np.arccos(np.clip(cos_face, -1.0, 1.0))))
            if cos_face >= cos_readable:
                facing = "front"
            elif cos_face >= cos_max:
                facing = "front_oblique"  # front face but steeply angled, hard to read
            elif cos_face <= -cos_max:
                facing = "back"
            else:
                continue  # edge-on: lamps unreadable and box degenerate
        pts_cam = (t_cam_map[:3, :3] @ tl["corners"].T + t_cam_map[:3, 3:4]).T
        if np.any(pts_cam[:, 2] < 1.0):  # behind or grazing the image plane
            continue
        uv = (intrinsic @ pts_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        x0, y0 = uv.min(axis=0)
        x1, y1 = uv.max(axis=0)
        if x1 < -margin or y1 < -margin or x0 > width + margin or y0 > height + margin:
            continue
        candidates.append(
            {
                "way_id": way_id,
                "subtype": tl["subtype"],
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "distance_m": distance,
                "facing": facing,
                "facing_deg": None if facing_deg is None else round(facing_deg, 1),
                "proj_min_side_px": round(float(min(x1 - x0, y1 - y0)), 1),
            }
        )
    return candidates


class MapProjector:
    """Map projection adapter: project(frame, image_wh) -> map signal candidates."""

    def __init__(self, traffic_lights, max_distance, max_incidence_deg, readable_incidence_deg):
        self.traffic_lights = traffic_lights
        self.max_distance = max_distance
        self.max_incidence_deg = max_incidence_deg
        self.readable_incidence_deg = readable_incidence_deg

    def project(self, frame, image_wh):
        return project_traffic_lights(
            frame,
            self.traffic_lights,
            self.max_distance,
            image_wh,
            max_incidence_deg=self.max_incidence_deg,
            readable_incidence_deg=self.readable_incidence_deg,
        )
