#!/usr/bin/env python3
"""Associate 2D traffic light detections with lanelet2 map traffic lights.

Pipeline per camera frame:
  1. T_map_cam = T_map_base(ego_pose) * T_base_cam(calibrated_sensor)
  2. Project each `way type=traffic_light` (bottom linestring + `height` tag
     extruded upward, Autoware convention) into the image -> predicted bbox.
  3. Hungarian matching between predicted bboxes and detected bboxes from
     tlr_autolabel/*.json (cost = 1 - IoU + normalized center distance, gated).
  4. Resolve `relation type=regulatory_element subtype=traffic_light` ids via
     role=refers membership of the matched traffic_light way.

Outputs:
  - annotation/traffic_signal_2d_ann.json (traffic_signal_2d/v1 sidecar table)
  - build/tl_match/match_report.json      (per-frame diagnostics)
  - build/tl_match/vis/*.jpg              (optional overlays via --vis)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.optimize import linear_sum_assignment

from state_tokens import elements_key, parse_state
from temporal_association import TemporalAssociator, TemporalTrackingConfig


# ---------------------------------------------------------------- lanelet2 map


def load_lanelet2_traffic_lights(osm_path: Path):
    """Return (traffic_lights, regulatory_by_way).

    traffic_lights: {way_id: {"corners": (4,3) array in map frame, "subtype": str}}
    regulatory_by_way: {way_id: [relation_id, ...]}
    """
    tree = ET.parse(osm_path)
    root = tree.getroot()

    nodes: dict[str, np.ndarray] = {}
    for node in root.iter("node"):
        tags = {t.get("k"): t.get("v") for t in node.findall("tag")}
        if "local_x" in tags and "local_y" in tags:
            nodes[node.get("id")] = np.array(
                [float(tags["local_x"]), float(tags["local_y"]), float(tags.get("ele", 0.0))]
            )

    traffic_lights: dict[str, dict] = {}
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if tags.get("type") != "traffic_light":
            continue
        refs = [nd.get("ref") for nd in way.findall("nd")]
        pts = [nodes[r] for r in refs if r in nodes]
        if len(pts) < 2:
            continue
        height = float(tags.get("height", 0.5))
        bottom = np.array(pts)
        up = np.array([0.0, 0.0, height])
        corners = np.vstack([bottom, bottom + up])
        # signed face normal: linestring direction rotated -90 deg ([dy, -dx]).
        # Empirically verified on this map: 99% of matches whose lamps were
        # readable (colored state) lie on this side; the opposite side only
        # collects `unknown` boxes = detections of the housing's back.
        direction = pts[-1][:2] - pts[0][:2]
        normal = np.array([direction[1], -direction[0]])
        norm = np.linalg.norm(normal)
        traffic_lights[way.get("id")] = {
            "corners": corners,
            "subtype": tags.get("subtype", ""),
            "height": height,
            "facing_axis": normal / norm if norm > 1e-9 else None,
        }

    regulatory_by_way: dict[str, list[str]] = defaultdict(list)
    for rel in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
        if tags.get("type") != "regulatory_element" or tags.get("subtype") != "traffic_light":
            continue
        for member in rel.findall("member"):
            if member.get("role") == "refers" and member.get("type") == "way":
                regulatory_by_way[member.get("ref")].append(rel.get("id"))

    return traffic_lights, regulatory_by_way


# ------------------------------------------------------------------ T4 tables


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


# ------------------------------------------------------------------ projection


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


# -------------------------------------------------------------------- matching


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def center(box):
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])


def _solve_assignment(cost, n_cand, unmatch_cost):
    rows, cols = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(rows, cols) if j < n_cand and cost[i, j] < unmatch_cost}


def match_boxes_legacy(detections, candidates, gate_factor=1.5, min_iou=0.05, unmatch_cost=1.5):
    """One-to-one Hungarian matching. Returns {det_index: cand_index}.

    Costs: overlapping pairs cost 1 - IoU (0..1); non-overlapping pairs within
    the center-distance gate cost 1 + dist/gate (1..2). Dummy "unmatched"
    columns cost `unmatch_cost`, so a poor pairing is only chosen when it beats
    leaving both boxes unmatched -- this stops the assignment from maximizing
    match count at the expense of quality.
    """
    if not detections or not candidates:
        return {}, {}
    big = 1e6
    n_det, n_cand = len(detections), len(candidates)
    cost = np.full((n_det, n_cand + n_det), big)
    cost[:, n_cand:] = unmatch_cost
    for i, det in enumerate(detections):
        dbox = det["box_xyxy"]
        ddiag = np.hypot(dbox[2] - dbox[0], dbox[3] - dbox[1])
        for j, cand in enumerate(candidates):
            cbox = cand["bbox"]
            cdiag = np.hypot(cbox[2] - cbox[0], cbox[3] - cbox[1])
            overlap = iou(dbox, cbox)
            dist = float(np.linalg.norm(center(dbox) - center(cbox)))
            gate = gate_factor * max(cdiag, ddiag)
            size_ratio = max(ddiag, cdiag) / max(min(ddiag, cdiag), 1e-6)
            if overlap >= min_iou:
                cost[i, j] = 1.0 - overlap
            elif dist <= gate and size_ratio <= 2.5:
                # offset fallback: sizes must agree, since projection error
                # shifts boxes but barely changes their scale
                cost[i, j] = 1.0 + dist / max(gate, 1e-6)
    matches = _solve_assignment(cost, n_cand, unmatch_cost)
    return matches, {i: "legacy" for i in matches}


def _build_cost(detections, candidates, det_indices, cand_indices, rules, unmatch_cost):
    big = 1e6
    cost = np.full((len(det_indices), len(cand_indices) + len(det_indices)), big)
    cost[:, len(cand_indices):] = unmatch_cost
    stages = {}
    for ri, det_i in enumerate(det_indices):
        dbox = detections[det_i]["box_xyxy"]
        ddiag = np.hypot(dbox[2] - dbox[0], dbox[3] - dbox[1])
        for cj, cand_j in enumerate(cand_indices):
            cbox = candidates[cand_j]["bbox"]
            cdiag = np.hypot(cbox[2] - cbox[0], cbox[3] - cbox[1])
            overlap = iou(dbox, cbox)
            dist = float(np.linalg.norm(center(dbox) - center(cbox)))
            size_ratio = max(ddiag, cdiag) / max(min(ddiag, cdiag), 1e-6)
            for rule in rules:
                if rule["kind"] == "iou":
                    if overlap < rule["min_iou"]:
                        continue
                    cost[ri, cj] = 1.0 - overlap
                    stages[(ri, cj)] = rule["stage"]
                    break
                if rule["kind"] == "distance":
                    gate = rule["gate_factor"] * max(cdiag, ddiag)
                    if dist > gate or size_ratio > rule["max_size_ratio"]:
                        continue
                    cost[ri, cj] = 1.0 + dist / max(gate, 1e-6)
                    stages[(ri, cj)] = rule["stage"]
                    break
    return cost, stages


def match_boxes_staged(detections, candidates, gate_factor=1.5, strict_min_iou=0.05,
                       relaxed_min_iou=0.01, relaxed_size_ratio=8.0,
                       relaxed_unmatch_cost=2.25):
    """Two-pass one-to-one matching.

    Pass 1 uses only IoU, so high-quality overlaps are claimed first. Pass 2
    runs on the remaining detections/candidates and allows either a weaker IoU
    or a center-distance gate with a looser size-ratio bound. This is intended
    for map QA / low-quality map geometry cases where projection boxes are
    visibly near the detection but too small or offset for the legacy cost.
    """
    if not detections or not candidates:
        return {}, {}

    det_indices = list(range(len(detections)))
    cand_indices = list(range(len(candidates)))
    strict_cost, strict_stages = _build_cost(
        detections,
        candidates,
        det_indices,
        cand_indices,
        [{"kind": "iou", "min_iou": strict_min_iou, "stage": "strict_iou"}],
        unmatch_cost=1.5,
    )
    strict_local = _solve_assignment(strict_cost, len(cand_indices), 1.5)
    matches = {det_indices[i]: cand_indices[j] for i, j in strict_local.items()}
    stages = {
        det_indices[i]: strict_stages.get((i, j), "strict_iou")
        for i, j in strict_local.items()
    }

    rem_det = [i for i in det_indices if i not in matches]
    used_cands = set(matches.values())
    rem_cand = [j for j in cand_indices if j not in used_cands]
    if not rem_det or not rem_cand:
        return matches, stages

    relaxed_cost, relaxed_stages = _build_cost(
        detections,
        candidates,
        rem_det,
        rem_cand,
        [
            {"kind": "iou", "min_iou": relaxed_min_iou, "stage": "relaxed_iou"},
            {
                "kind": "distance",
                "gate_factor": gate_factor,
                "max_size_ratio": relaxed_size_ratio,
                "stage": "relaxed_distance",
            },
        ],
        unmatch_cost=relaxed_unmatch_cost,
    )
    relaxed_local = _solve_assignment(relaxed_cost, len(rem_cand), relaxed_unmatch_cost)
    for i, j in relaxed_local.items():
        det_i = rem_det[i]
        cand_j = rem_cand[j]
        matches[det_i] = cand_j
        stages[det_i] = relaxed_stages.get((i, j), "relaxed")
    return matches, stages


def match_boxes(detections, candidates, mode="legacy", gate_factor=1.5,
                strict_min_iou=0.05, relaxed_min_iou=0.01,
                relaxed_size_ratio=8.0, relaxed_unmatch_cost=2.25):
    if mode == "legacy":
        return match_boxes_legacy(detections, candidates, gate_factor=gate_factor,
                                  min_iou=strict_min_iou)
    return match_boxes_staged(
        detections,
        candidates,
        gate_factor=gate_factor,
        strict_min_iou=strict_min_iou,
        relaxed_min_iou=relaxed_min_iou,
        relaxed_size_ratio=relaxed_size_ratio,
        relaxed_unmatch_cost=relaxed_unmatch_cost,
    )


def unmatched_reason(det, det_state, candidates, matches, gate_factor=1.5):
    """Classify why a detection stayed unmatched (review triage; report-only).
    Categories, checked in order:
      state_unknown_backside  — classifier saw no lamps (likely back/side view)
      no_map_candidate_in_view — no projected map TL in this frame at all
      candidate_taken         — nearest candidate was assigned to another detection
      beyond_gate             — nearest candidate exists but is too far
      geometry_mismatch       — within gate yet rejected (size ratio / cost)
    """
    if det_state == "unknown":
        return "state_unknown_backside"
    if not candidates:
        return "no_map_candidate_in_view"
    dbox = det["box_xyxy"]
    ddiag = np.hypot(dbox[2] - dbox[0], dbox[3] - dbox[1])
    dists = [float(np.linalg.norm(center(dbox) - center(c["bbox"]))) for c in candidates]
    j = int(np.argmin(dists))
    if j in set(matches.values()):
        return "candidate_taken"
    cbox = candidates[j]["bbox"]
    cdiag = np.hypot(cbox[2] - cbox[0], cbox[3] - cbox[1])
    if dists[j] > gate_factor * max(cdiag, ddiag):
        return "beyond_gate"
    return "geometry_mismatch"


# ---------------------------------------------------------------- state labels


def raw_state(signal_entry) -> str:
    """Detector state string as written by L1 (v1 `state`, pre-v1 `signal`)."""
    return signal_entry.get("state") or signal_entry.get("signal") or "unknown"


def canonical_state(signal_entry) -> str:
    """Canonical normalized state: parse (handles legacy tokens too) and
    re-serialize sorted; 'unknown' when no lamp carries state."""
    return elements_key(parse_state(raw_state(signal_entry))) or "unknown"


def signal_kind(elements: list[dict]) -> str:
    if any(e["shape"] == "ped" for e in elements):
        return "pedestrian"
    return "vehicle" if elements else "unknown"


def detector_score(signal_entry, default=1.0) -> float:
    score = signal_entry.get("detector_score", default)
    if score is None or score == "":
        return default
    return float(score)


def load_tracking_config(args) -> TemporalTrackingConfig:
    cfg = TemporalTrackingConfig()
    if args.tracking_config:
        import yaml
        config_path = args.tracking_config
        if not config_path.is_absolute():
            candidates = [
                config_path,
                Path(__file__).resolve().parent / config_path,
                args.dataset_root / config_path,
            ]
            config_path = next((p for p in candidates if p.exists()), candidates[0])
        data = yaml.safe_load(config_path.read_text()) or {}
        if "temporal_tracking" in data:
            data = data["temporal_tracking"] or {}
        # Config files are parameter profiles only. Keep the feature gate on the
        # CLI so passing --tracking-config never changes the legacy path by
        # accident.
        data.pop("enabled", None)
        cfg = TemporalTrackingConfig.from_mapping(data)
    cfg = TemporalTrackingConfig.from_mapping({
        **cfg.__dict__,
        "enabled": bool(args.temporal_tracking),
    })
    cli_overrides = {
        "low_score": args.tracking_low_score,
        "max_lost_frames": args.tracking_max_lost_frames,
        "min_iou": args.tracking_min_iou,
        "center_gate_factor": args.tracking_center_gate_factor,
        "projection_gate_factor": args.tracking_projection_gate_factor,
        "max_size_ratio": args.tracking_max_size_ratio,
    }
    cfg = TemporalTrackingConfig.from_mapping({
        **cfg.__dict__,
        **{k: v for k, v in cli_overrides.items() if v is not None},
    })
    if args.no_tracking_propagation:
        cfg = TemporalTrackingConfig.from_mapping({**cfg.__dict__, "propagate": False})
    if args.tracking_propagate_state:
        cfg = TemporalTrackingConfig.from_mapping({**cfg.__dict__, "propagate_state": True})
    return cfg


def collect_low_tracking_candidates(payload, high_threshold, low_threshold):
    """Return classified low-confidence candidates without duplicating highs.

    Newer L1 JSONs may carry raw_detections down to --det-low-score-thr. Older
    runs only have signals; those can still serve as low candidates when L1 was
    run below match_traffic_lights.py --min-score.
    """
    low = []
    high_boxes = []
    for det in payload.get("signals", []):
        score = detector_score(det)
        if score >= high_threshold:
            high_boxes.append(det["box_xyxy"])
        elif score >= low_threshold:
            low.append(det)

    for det in payload.get("raw_detections", []):
        score = detector_score(det)
        if score < low_threshold or score >= high_threshold:
            continue
        box = det["box_xyxy"]
        if any(iou(box, existing) >= 0.98 for existing in high_boxes):
            continue
        if any(iou(box, existing["box_xyxy"]) >= 0.98 for existing in low):
            continue
        low.append(det)
    return low


# ------------------------------------------------------------------------ main


def stable_token(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


FILENAME_RE = re.compile(r"^(?P<channel>.+)_(?P<frame>\d+)$")


def _load_font(size=26):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_with_bg(draw, xy, text, fill, font):
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle([left - 3, top - 2, right + 3, bottom + 2], fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill, font=font)


def draw_overlay(root, frame, detections, candidates, matches, out_path):
    from PIL import Image, ImageDraw

    img = Image.open(root / frame["sample_data"]["filename"]).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font()
    for j, cand in enumerate(candidates):
        box = cand["bbox"]
        draw.rectangle(box, outline=(0, 160, 255), width=3)
        _text_with_bg(draw, (box[0], max(box[1] - 34, 0)),
                      f"map:{cand['way_id']} {cand['distance_m']:.0f}m", (80, 190, 255), font)
    matched_cands = set(matches.values())
    for i, det in enumerate(detections):
        box = det["box_xyxy"]
        color = (0, 220, 0) if i in matches else (255, 60, 60)
        draw.rectangle(list(box), outline=color, width=3)
        label = raw_state(det)
        if i in matches:
            label += f" -> {candidates[matches[i]]['way_id']}"
            draw.line(
                [tuple(center(box)), tuple(center(candidates[matches[i]]["bbox"]))],
                fill=(255, 255, 0), width=3,
            )
        _text_with_bg(draw, (box[0], box[3] + 8), label, color, font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=88)
    return len(matched_cands)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--autolabel-dir", default=Path("tlr_autolabel"), type=Path)
    parser.add_argument("--output", default=None, type=Path,
                        help="Sidecar table path (default: annotation/traffic_signal_2d_ann.json; "
                             "not written when --frames is used unless given explicitly).")
    parser.add_argument("--report", default=None, type=Path,
                        help="Diagnostics path (default: build/tl_match/match_report.json; "
                             "not written when --frames is used unless given explicitly).")
    parser.add_argument("--max-distance", default=200.0, type=float,
                        help="Max ego-to-signal distance for a map candidate to be projected.")
    parser.add_argument("--max-incidence-deg", default=85.0, type=float,
                        help="Drop map candidates seen closer to edge-on than this "
                             "(unsigned face-normal vs sight-line angle). 85 keeps "
                             "side-on but still-visible signals; only near-90 (a signal "
                             "truly seen edge-on) is dropped. The eval layer can filter "
                             "high-incidence candidates separately via facing_deg.")
    parser.add_argument("--readable-incidence-deg", default=60.0, type=float,
                        help="Incidence up to which a front-facing signal reads cleanly "
                             "(facing='front'). Between this and --max-incidence-deg the "
                             "signal is steeply angled (facing='front_oblique') and L2 marks "
                             "it occlusion_state.partial. 60 ~ the readable/unreadable border.")
    parser.add_argument("--gate-factor", default=1.5, type=float)
    parser.add_argument("--match-mode", choices=["legacy", "staged"], default="legacy",
                        help="legacy: original single Hungarian cost; staged: strict IoU pass "
                             "then relaxed IoU/distance pass on leftovers.")
    parser.add_argument("--strict-min-iou", default=0.05, type=float,
                        help="Minimum IoU for legacy matching and staged strict pass.")
    parser.add_argument("--relaxed-min-iou", default=0.01, type=float,
                        help="Minimum IoU for staged relaxed pass.")
    parser.add_argument("--relaxed-size-ratio", default=8.0, type=float,
                        help="Maximum bbox diagonal ratio for staged distance fallback.")
    parser.add_argument("--relaxed-unmatch-cost", default=2.25, type=float,
                        help="Dummy assignment cost in staged relaxed pass; >2 allows "
                             "any in-gate distance match.")
    parser.add_argument("--min-score", default=0.5, type=float,
                        help="Ignore detections below this detector_score.")
    tracking_group = parser.add_mutually_exclusive_group()
    tracking_group.add_argument("--temporal-tracking", dest="temporal_tracking",
                                action="store_true", default=False,
                                help="Enable ByteTrack-like temporal association: high-score "
                                     "detections can create/update map-way tracks; low-score "
                                     "detections only update existing tracks; short misses can "
                                     "be written as propagated boxes.")
    tracking_group.add_argument("--no-temporal-tracking", dest="temporal_tracking",
                                action="store_false",
                                help="Disable temporal association explicitly. This is the "
                                     "default and also overrides any enabled field in a "
                                     "tracking config.")
    parser.add_argument("--tracking-config", type=Path, default=None,
                        help="YAML/JSON parameter profile for temporal association thresholds "
                             "and TTL. May be flat or nested under temporal_tracking; it does "
                             "not enable tracking without --temporal-tracking.")
    parser.add_argument("--tracking-low-score", type=float, default=None,
                        help="Low detector score floor used only for existing-track recovery "
                             "(default from config, otherwise 0.2).")
    parser.add_argument("--tracking-max-lost-frames", type=int, default=None,
                        help="Lost-track TTL in frames (default from config, otherwise 3).")
    parser.add_argument("--tracking-min-iou", type=float, default=None,
                        help="Minimum IoU for low detection to projection/previous bbox "
                             "(default from config, otherwise 0.01).")
    parser.add_argument("--tracking-center-gate-factor", type=float, default=None,
                        help="Center-distance gate vs previous track bbox diagonal "
                             "(default from config, otherwise 2.0).")
    parser.add_argument("--tracking-projection-gate-factor", type=float, default=None,
                        help="Center-distance gate vs map projection bbox diagonal "
                             "(default from config, otherwise 2.0).")
    parser.add_argument("--tracking-max-size-ratio", type=float, default=None,
                        help="Maximum bbox diagonal ratio for low-track association "
                             "(default from config, otherwise 8.0).")
    parser.add_argument("--no-tracking-propagation", action="store_true",
                        help="Keep lost tracks internally but do not write propagated boxes.")
    parser.add_argument("--tracking-propagate-state", action="store_true",
                        help="Write the last observed state on propagated boxes. Default is "
                             "unknown because signal state can change while detection is lost.")
    parser.add_argument("--vis", default=0, type=int,
                        help="Save overlay images for the first N frames that have detections.")
    parser.add_argument("--frames", default="",
                        help="Comma-separated frame stems (e.g. CAM_FRONT_00076). "
                             "Restricts processing to these frames and renders their overlays.")
    parser.add_argument("--vis-dir", default=Path("build/tl_match/vis"), type=Path)
    parser.add_argument("--limit", default=0, type=int, help="Process only the first N files (debug).")
    parser.add_argument("--fill-gaps", dest="fill_gaps", action="store_true", default=True,
                        help="Fill temporal detection gaps for a regulatory element seen "
                             "before and after (default on): project the map traffic light "
                             "into the missing frames as interpolated boxes.")
    parser.add_argument("--no-fill-gaps", dest="fill_gaps", action="store_false")
    parser.add_argument("--fill-mode", choices=["strict", "bracketed", "all"],
                        default="bracketed",
                        help="strict: only same-state interior gaps -> that state. "
                             "bracketed (default): all interior gaps between two detections "
                             "-> same-state gives the state, differing gives unknown. "
                             "all: also fill one-sided leading/trailing in-view frames with unknown.")
    parser.add_argument("--max-gap-frames", default=5, type=int,
                        help="Only bridge gaps up to this many consecutive missing frames.")
    parser.add_argument("--map-fill", dest="map_fill", action="store_true", default=True,
                        help="Trust the map (default on): a near, front-facing signal the "
                             "detector missed between real detections still gets an unknown "
                             "box at its projected map position, for review. Independent of "
                             "state gap filling.")
    parser.add_argument("--no-map-fill", dest="map_fill", action="store_false")
    parser.add_argument("--map-fill-max-distance", default=130.0, type=float,
                        help="Only map-fill signals within this distance. Corroboration "
                             "(a real detection of the same signal within --map-fill-window) "
                             "keeps the projection on the real signal even far out (verified "
                             "to ~123m at the next intersection), covering center-far signals "
                             "the detector drops while approaching. Boxes get small/less "
                             "precise with distance; the eval layer bins by distance.")
    parser.add_argument("--map-fill-window", default=30, type=int,
                        help="Map-fill a near/front frame only if the SAME signal was "
                             "actually detected before and after within this many frames — "
                             "the projection is trusted only for bracketed misses, so it "
                             "does not extrapolate after a signal leaves the image.")
    return parser.parse_args()


def clip_box_in_image(box, wh, min_frac=0.5, min_side=6.0):
    """Clip a projected box to the image; return the clipped box only if the
    signal is actually reviewable there — most of it inside the frame and not a
    sliver. Near signals directly overhead/beside the car project off-frame;
    those must not become annotations. Returns None when unusable."""
    w, h = wh
    x0, y0, x1, y1 = box
    cx0, cy0 = max(0.0, min(x0, x1)), max(0.0, min(y0, y1))
    cx1, cy1 = min(float(w), max(x0, x1)), min(float(h), max(y0, y1))
    if cx1 - cx0 < min_side or cy1 - cy0 < min_side:
        return None
    orig_area = abs((x1 - x0) * (y1 - y0))
    if orig_area <= 0 or (cx1 - cx0) * (cy1 - cy0) < min_frac * orig_area:
        return None
    return [cx0, cy0, cx1, cy1]


def plan_gap_fills(track, max_gap, mode):
    """Given one (channel, way) timeline (each entry has state=None when the RE
    was in view but not detected), decide which missing frames to fill and with
    what state. Yields (entry, fill_state). source order preserved; caller dedups.

      strict    : interior gap, both sides same state, len<=max_gap -> that state
      bracketed : + interior gaps whose sides differ (or exceed max_gap)  -> unknown
      all       : + one-sided leading/trailing in-view frames (<=max_gap) -> unknown
    """
    matched = [i for i, e in enumerate(track) if e["state"] is not None]
    if not matched:
        return
    for a, b in zip(matched, matched[1:]):
        gap = track[a + 1:b]
        if not gap:
            continue
        sa, sb = track[a]["state"], track[b]["state"]
        if len(gap) <= max_gap:
            if sa == sb:
                state = sa                       # othello: both sides agree
            elif mode in ("bracketed", "all"):
                state = "unknown"                # sides differ: presence only
            else:
                continue                         # strict: leave differing gaps
            for e in gap:
                yield e, state
        elif mode == "all":
            # long gap: only fill a bounded margin next to each detection
            for e in gap[:max_gap]:
                yield e, "unknown"
            for e in gap[-max_gap:]:
                yield e, "unknown"
    if mode == "all":
        for e in track[max(0, matched[0] - max_gap):matched[0]]:
            yield e, "unknown"
        for e in track[matched[-1] + 1:matched[-1] + 1 + max_gap]:
            yield e, "unknown"


def bracketed_by_matches(index: int, matched_indices: list[int], window: int) -> bool:
    """True when an unmatched projected frame is bounded by real detections.

    Map-presence fill is intentionally not one-sided extrapolation: once a
    signal leaves the frame (commonly through the top edge), the last detection
    must not keep authorizing projected boxes for the next `window` frames.
    """
    has_prev = any(0 < index - m <= window for m in matched_indices)
    has_next = any(0 < m - index <= window for m in matched_indices)
    return has_prev and has_next


def main():
    args = parse_args()
    root = args.dataset_root.resolve()
    args.dataset_root = root
    tracking_cfg = load_tracking_config(args)
    tracker = TemporalAssociator(tracking_cfg) if tracking_cfg.enabled else None
    channel_frame_numbers: dict[str, int] = defaultdict(int)

    traffic_lights, regulatory_by_way = load_lanelet2_traffic_lights(root / "map/lanelet2_map.osm")
    map_projector = MapProjector(
        traffic_lights,
        args.max_distance,
        args.max_incidence_deg,
        args.readable_incidence_deg,
    )
    frames, frames_by_token = load_t4_index(root)
    print(f"map traffic lights: {len(traffic_lights)}, "
          f"ways with regulatory element: {len(regulatory_by_way)}, camera frames: {len(frames)}")
    if tracking_cfg.enabled:
        print("temporal tracking: "
              f"low_score={tracking_cfg.low_score:g}, "
              f"max_lost_frames={tracking_cfg.max_lost_frames}, "
              f"min_iou={tracking_cfg.min_iou:g}")

    # accept both a flat dir of <CHANNEL>_<frame>.json and per-channel subdirs
    # of <frame>.json (tlr_autolabel.py --out-dir layout)
    files = sorted((root / args.autolabel_dir).rglob("*.json"))
    if args.frames:
        wanted = {s.strip() for s in args.frames.split(",") if s.strip()}
        files = [f for f in files if f.stem in wanted]
        args.vis = max(args.vis, len(files))
    if args.limit > 0:
        files = files[: args.limit]

    annotations = []
    report_frames = []
    vis_left = args.vis
    stats = defaultdict(int)
    # per (channel, way_id) timeline for gap filling: every frame the way is
    # in view (a projected candidate), with the matched state if any.
    way_track: dict[tuple[str, str], list[dict]] = defaultdict(list)
    annotated_way_keys: set[tuple[str, str]] = set()

    for path in files:
        payload = json.loads(path.read_text())
        # tlr_autolabel/v1 carries the sample_data_token -> exact frame lookup,
        # independent of file naming. Filename parsing stays as the legacy path.
        frame = frames_by_token.get(payload.get("sample_data_token") or "")
        if frame is None:
            m = FILENAME_RE.match(path.stem)
            if not m:
                stats["bad_filename"] += 1
                continue
            frame = frames.get((m.group("channel"), m.group("frame")))
        if frame is None:
            stats["no_sample_data"] += 1
            continue

        channel_frame_numbers[frame["channel"]] += 1
        tracking_frame_number = channel_frame_numbers[frame["channel"]]

        detections = [d for d in payload.get("signals", [])
                      if detector_score(d) >= args.min_score]
        low_detections = (
            collect_low_tracking_candidates(payload, args.min_score, tracking_cfg.low_score)
            if tracking_cfg.enabled else []
        )
        image_wh = (payload.get("width", 2880), payload.get("height", 1860))

        candidates = map_projector.project(frame, image_wh)
        matches, match_stages = match_boxes(
            detections,
            candidates,
            mode=args.match_mode,
            gate_factor=args.gate_factor,
            strict_min_iou=args.strict_min_iou,
            relaxed_min_iou=args.relaxed_min_iou,
            relaxed_size_ratio=args.relaxed_size_ratio,
            relaxed_unmatch_cost=args.relaxed_unmatch_cost,
        )
        tracking_result = None
        tracked_low_matches = {}
        if tracker is not None:
            tracking_result = tracker.update(
                frame["channel"],
                tracking_frame_number,
                detections,
                matches,
                low_detections,
                candidates,
                state_fn=canonical_state,
                raw_state_fn=raw_state,
            )
            tracked_low_matches = tracking_result.low_matches

        stats["frames"] += 1
        stats["detections"] += len(detections)
        if tracking_cfg.enabled:
            stats["tracking_low_candidates"] += len(low_detections)
            stats["tracking_low_matched"] += len(tracked_low_matches)
            stats["tracking_terminated"] += len(tracking_result.terminated_tracks)
        for i, det in enumerate(detections):
            if i not in matches:
                stats["unmatched:" + unmatched_reason(
                    det, canonical_state(det), candidates, matches, args.gate_factor)] += 1
        stats["map_candidates"] += len(candidates)
        stats["matched"] += len(matches)
        for stage in match_stages.values():
            stats["matched:" + stage] += 1

        sd = frame["sample_data"]
        tracked_current_cand_idx = {
            match.candidate_index for match in tracked_low_matches.values()
            if match.candidate_index is not None
        }
        matched_cand_idx = set(matches.values()) | tracked_current_cand_idx
        frame_report = {
            "file": path.name,
            "channel": frame["channel"],
            "sample_data_token": sd["token"],
            "n_detections": len(detections),
            "n_map_candidates": len(candidates),
            # all projected map candidates incl. undetected ones -- the
            # evaluation layer bins detection coverage by distance from these
            "candidates": [
                {"way_id": c["way_id"], "distance_m": c["distance_m"],
                 "bbox": c["bbox"], "facing": c["facing"],
                 "facing_deg": c["facing_deg"],
                 "proj_min_side_px": c["proj_min_side_px"],
                 "matched": j in matched_cand_idx}
                for j, c in enumerate(candidates)
            ],
            "pairs": [],
        }
        if tracking_cfg.enabled:
            frame_report["n_low_tracking_candidates"] = len(low_detections)
            frame_report["n_low_tracking_matches"] = len(tracked_low_matches)

        observed_cand_state = (
            dict(tracking_result.observed_candidate_states)
            if tracking_result is not None else {}
        )
        high_tracks = {
            obs.det_index: obs.track for obs in tracking_result.observed_tracks
            if obs.source_type == "auto"
        } if tracking_result is not None else {}
        low_observations = [
            obs for obs in tracking_result.observed_tracks
            if obs.source_type == "tracked"
        ] if tracking_result is not None else []

        def append_detection_annotation(det, token, cand, source_type, *,
                                        match_stage=None, track=None,
                                        real_detection=True):
            reg_ids = regulatory_by_way.get(cand["way_id"], []) if cand else []
            pair_iou = iou(det["box_xyxy"], cand["bbox"]) if cand else 0.0
            # For an unmatched detection, keep the nearest in-view map candidate
            # as a *soft* association (its way + RE + why the match was rejected),
            # so the info isn't lost — a reviewer can promote it. The authoritative
            # map_traffic_light_id stays empty; the _candidate fields carry the hint.
            reason, cand_way, cand_re = "", "", ""
            if cand is None and candidates:
                reason = unmatched_reason(det, canonical_state(det), candidates,
                                          matches, args.gate_factor)
                dists = [float(np.linalg.norm(center(det["box_xyxy"]) - center(c["bbox"])))
                         for c in candidates]
                nc = candidates[int(np.argmin(dists))]
                cand_way = nc["way_id"]
                cand_re = ",".join(regulatory_by_way.get(nc["way_id"], []))
            elif cand is None:
                reason = unmatched_reason(det, canonical_state(det), candidates,
                                          matches, args.gate_factor)
            state = canonical_state(det) if real_detection else (det.get("state") or "unknown")
            raw = raw_state(det) if real_detection else ""
            if (
                real_detection
                and source_type == "tracked"
                and state == "unknown"
                and track is not None
                and track.last_state != "unknown"
            ):
                state = track.last_state
            score = det.get("detector_score") if real_detection else None
            kind_elements = parse_state(raw)
            if not kind_elements:
                kind_elements = parse_state(state)
            attrs = {
                "state": state,
                "signal_kind": signal_kind(kind_elements),
                # the detector fired here, so the signal is visible; propagated
                # boxes are only a short-lived presence hint and need review.
                "visibility": "full" if real_detection else "unknown",
                "review_status": "unchecked",
                "map_traffic_light_id": cand["way_id"] if cand else "",
                "regulatory_element_id": ",".join(reg_ids),
                "map_candidate_id": cand_way,
                "regulatory_element_id_candidate": cand_re,
                "unmatched_reason": reason,
                "facing": cand["facing"] if cand else "",
                "raw_state": raw,
                "detector_score": "" if score is None else f"{score}",
                "source_type": source_type,
            }
            if tracking_cfg.enabled:
                attrs.update({
                    "temporal_source": "observed" if real_detection else "propagated",
                    "track_id": track.track_id if track else "",
                    "tracking_status": (
                        "observed" if real_detection and track else
                        (track.status if track else "")
                    ),
                    "tracking_lost_frames": (
                        str(track.lost_frames) if track else ""
                    ),
                })
            annotations.append(
                {
                    "token": token,
                    "sample_token": sd["sample_token"],
                    "sample_data_token": sd["token"],
                    "channel": frame["channel"],
                    "filename": sd["filename"],
                    "timestamp": sd["timestamp"],
                    "label": "traffic_light",
                    "box2d": [float(v) for v in det["box_xyxy"]],
                    "occluded": False,
                    "z_order": 0,
                    "attributes": attrs,
                }
            )
            if cand:
                annotated_way_keys.add((sd["token"], cand["way_id"]))
            frame_report["pairs"].append(
                {
                    "detection_box": det["box_xyxy"],
                    "detector_score": score,
                    "signal": raw or state,
                    "map_traffic_light_id": cand["way_id"] if cand else None,
                    "map_subtype": cand["subtype"] if cand else None,
                    "regulatory_element_ids": reg_ids,
                    "projected_box": cand["bbox"] if cand else None,
                    "distance_m": cand["distance_m"] if cand else None,
                    "iou": round(pair_iou, 3),
                    "match_stage": match_stage,
                    "source_type": source_type,
                    "track_id": track.track_id if track else None,
                    "unmatched_reason": None if cand else reason,
                }
            )

        for i, det in enumerate(detections):
            token = stable_token(sd["token"], i, *det["box_xyxy"])
            cand = candidates[matches[i]] if i in matches else None
            track = None
            if cand is not None:
                observed_cand_state[matches[i]] = canonical_state(det)
                track = high_tracks.get(i)
            append_detection_annotation(
                det, token, cand, "auto",
                match_stage=match_stages.get(i), track=track
            )

        for obs in low_observations:
            i = obs.det_index
            det = low_detections[i]
            cand = obs.candidate
            track = obs.track
            token = stable_token(
                sd["token"], "tracked",
                det.get("raw_detection_id") or det.get("signal_id") or i,
                *det["box_xyxy"],
            )
            append_detection_annotation(
                det, token, cand, "tracked",
                match_stage=f"temporal_low:{obs.association_source}", track=track
            )

        if tracking_result is not None:
            for propagated in tracking_result.propagated_tracks:
                track = propagated.track
                state = track.last_state if tracking_cfg.propagate_state else "unknown"
                det = {
                    "box_xyxy": propagated.bbox,
                    "state": state,
                    "lamps": [],
                    "detector_score": None,
                }
                token = stable_token(sd["token"], "propagated", track.track_id)
                append_detection_annotation(
                    det, token, propagated.candidate, "propagated",
                    match_stage="temporal_propagated", track=track,
                    real_detection=False
                )
                stats["tracking_propagated"] += 1
        report_frames.append(frame_report)

        # record every in-view map candidate for gap filling (matched or not)
        for j, c in enumerate(candidates):
            way_track[(frame["channel"], c["way_id"])].append({
                "timestamp": sd["timestamp"],
                "sample_token": sd["sample_token"],
                "sample_data_token": sd["token"],
                "filename": sd["filename"],
                "channel": frame["channel"],
                "proj_box": c["bbox"],
                "wh": image_wh,
                "facing": c["facing"],
                "distance": c["distance_m"],
                # Only real high/low observations bracket later gap filling.
                # Propagated boxes are review aids, not evidence for more fill.
                "state": observed_cand_state.get(j),
            })

        if vis_left > 0 and detections:
            draw_overlay(root, frame, detections, candidates, matches,
                         root / args.vis_dir / f"{path.stem}.jpg")
            vis_left -= 1

    # ---- gap filling: bridge short detection dropouts of a regulatory element
    # that was matched before and after. The map traffic light is projected into
    # each missing frame (accurate box + RE id), state copied from the bracketing
    # matched frames when they agree. Marked source_type=interpolated for review.
    if args.fill_gaps or args.map_fill:
        # Two backfill sources, merged per (frame, way) with priority so a signal
        # never gets two boxes: a concrete gap-filled state beats an unknown.
        #   priority 2 = temporal gap fill with a concrete state
        #   priority 1 = unknown (temporal presence, or map-presence)
        # value: (priority, source_type, state, entry)
        fills: dict[tuple[str, str], tuple] = {}

        def offer(key, priority, source, state, entry):
            if key in annotated_way_keys:
                return
            clipped = clip_box_in_image(entry["proj_box"], entry["wh"])
            if clipped is None:                      # projected off-frame: not reviewable
                return
            entry = {**entry, "proj_box": clipped}
            if key not in fills or priority > fills[key][0]:
                fills[key] = (priority, source, state, entry)

        for (channel, way_id), track in way_track.items():
            track.sort(key=lambda r: r["timestamp"])
            if args.fill_gaps:
                for g, state in plan_gap_fills(track, args.max_gap_frames, args.fill_mode):
                    offer((g["sample_data_token"], way_id),
                          2 if state != "unknown" else 1, "interpolated", state, g)
            if args.map_fill:
                # Near, front-facing frames the detector missed get an unknown box
                # at the projected position -- but ONLY for misses bracketed by
                # real detections of the same signal. One-sided extrapolation
                # leaves stale boxes after a signal exits through the image top.
                matched_idx = [i for i, e in enumerate(track) if e["state"] is not None]
                for i, e in enumerate(track):
                    if (e["state"] is None and e["facing"] == "front"
                            and e["distance"] <= args.map_fill_max_distance
                            and bracketed_by_matches(i, matched_idx, args.map_fill_window)):
                        offer((e["sample_data_token"], way_id), 1,
                              "map_presence", "unknown", e)

        n_state = n_unknown = n_mappres = 0
        for (sd_token, way_id), (_prio, source, state, e) in fills.items():
            reg_ids = regulatory_by_way.get(way_id, [])
            attrs = {
                "state": state,
                "signal_kind": signal_kind(parse_state(state)),
                # interpolated boxes are bracketed by detections (visible);
                # map_presence boxes are detector-missed (small/occluded) so
                # leave visibility for the reviewer to decide.
                "visibility": "full" if source == "interpolated" else "unknown",
                "review_status": "unchecked",
                "map_traffic_light_id": way_id,
                "regulatory_element_id": ",".join(reg_ids),
                "map_candidate_id": "",
                "regulatory_element_id_candidate": "",
                "unmatched_reason": "",
                "facing": e["facing"],
                "raw_state": "",       # not a detection
                "detector_score": "",
                "source_type": source,
            }
            if tracking_cfg.enabled:
                attrs.update({
                    "temporal_source": (
                        "map_presence" if source == "map_presence" else "propagated"
                    ),
                    "track_id": "",
                    "tracking_status": "",
                    "tracking_lost_frames": "",
                })
            annotations.append({
                "token": stable_token(sd_token, source, way_id),
                "sample_token": e["sample_token"],
                "sample_data_token": sd_token,
                "channel": e.get("channel", ""),
                "filename": e["filename"],
                "timestamp": e["timestamp"],
                "label": "traffic_light",
                "box2d": [float(v) for v in e["proj_box"]],
                "occluded": False,
                "z_order": 0,
                "attributes": attrs,
            })
            n_state += (state != "unknown")
            n_unknown += (state == "unknown")
            n_mappres += (source == "map_presence")
        stats["backfilled"] = len(fills)
        stats["backfilled_state"] = n_state
        stats["backfilled_map_presence"] = n_mappres
        print(f"backfill: {len(fills)} boxes "
              f"({n_state} gap-filled with state, {n_unknown} unknown; "
              f"of which {n_mappres} bracketed map-presence "
              f"@<= {args.map_fill_max_distance:g}m front; "
              f"gap-mode={args.fill_mode}, max_gap_frames={args.max_gap_frames})")

    # With --frames we only render overlays; don't clobber the full-run tables
    # unless output paths were given explicitly.
    if args.output or not args.frames:
        out_path = root / (args.output or Path("annotation/traffic_signal_2d_ann.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"schema_version": "traffic_signal_2d/v2", "source": "map_projection_auto",
             "annotations": annotations}, indent=2))
        print(f"wrote {out_path}")
    if args.report or not args.frames:
        report_path = root / (args.report or Path("build/tl_match/match_report.json"))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_params = {"max_distance": args.max_distance, "gate_factor": args.gate_factor,
                         "match_mode": args.match_mode,
                         "strict_min_iou": args.strict_min_iou,
                         "relaxed_min_iou": args.relaxed_min_iou,
                         "relaxed_size_ratio": args.relaxed_size_ratio,
                         "relaxed_unmatch_cost": args.relaxed_unmatch_cost,
                         "min_score": args.min_score}
        if tracking_cfg.enabled:
            report_params["temporal_tracking"] = tracking_cfg.__dict__
        report_path.write_text(json.dumps(
            {"params": report_params, "stats": dict(stats), "frames": report_frames}, indent=2))
        print(f"wrote {report_path}")

    matched_pct = 100.0 * stats["matched"] / max(stats["detections"], 1)
    print(f"frames={stats['frames']} detections={stats['detections']} "
          f"matched={stats['matched']} ({matched_pct:.1f}%)")
    if tracking_cfg.enabled:
        print(f"tracking: low_candidates={stats['tracking_low_candidates']} "
              f"low_matched={stats['tracking_low_matched']} "
              f"propagated={stats['tracking_propagated']}")


if __name__ == "__main__":
    main()
