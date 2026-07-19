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
                           max_incidence_deg=75.0):
    """Project all map traffic lights into this frame -> list of candidates.

    Facing classification per candidate (angle between the signed face normal
    and the sight line): "front" (<= max_incidence_deg -- lamps readable),
    "back" (>= 180 - max_incidence_deg -- the housing's back, detector may
    still fire an `unknown` box on it). Near-edge-on candidates in between are
    dropped: empirical matched-rate collapses there (12% at 70-80, 6% at
    80-90 unsigned incidence).
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
            if cos_face >= cos_max:
                facing = "front"
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


def match_boxes(detections, candidates, gate_factor=1.5, min_iou=0.05, unmatch_cost=1.5):
    """One-to-one Hungarian matching. Returns {det_index: cand_index}.

    Costs: overlapping pairs cost 1 - IoU (0..1); non-overlapping pairs within
    the center-distance gate cost 1 + dist/gate (1..2). Dummy "unmatched"
    columns cost `unmatch_cost`, so a poor pairing is only chosen when it beats
    leaving both boxes unmatched -- this stops the assignment from maximizing
    match count at the expense of quality.
    """
    if not detections or not candidates:
        return {}
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
    rows, cols = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(rows, cols) if j < n_cand and cost[i, j] < unmatch_cost}


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
    parser.add_argument("--max-distance", default=150.0, type=float)
    parser.add_argument("--max-incidence-deg", default=75.0, type=float,
                        help="Drop map candidates seen closer to edge-on than this "
                             "(unsigned face-normal vs sight-line angle).")
    parser.add_argument("--gate-factor", default=1.5, type=float)
    parser.add_argument("--min-score", default=0.5, type=float,
                        help="Ignore detections below this detector_score.")
    parser.add_argument("--vis", default=0, type=int,
                        help="Save overlay images for the first N frames that have detections.")
    parser.add_argument("--frames", default="",
                        help="Comma-separated frame stems (e.g. CAM_FRONT_00076). "
                             "Restricts processing to these frames and renders their overlays.")
    parser.add_argument("--vis-dir", default=Path("build/tl_match/vis"), type=Path)
    parser.add_argument("--limit", default=0, type=int, help="Process only the first N files (debug).")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.dataset_root.resolve()

    traffic_lights, regulatory_by_way = load_lanelet2_traffic_lights(root / "map/lanelet2_map.osm")
    frames, frames_by_token = load_t4_index(root)
    print(f"map traffic lights: {len(traffic_lights)}, "
          f"ways with regulatory element: {len(regulatory_by_way)}, camera frames: {len(frames)}")

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

        detections = [d for d in payload.get("signals", [])
                      if d.get("detector_score", 1.0) >= args.min_score]
        image_wh = (payload.get("width", 2880), payload.get("height", 1860))

        candidates = project_traffic_lights(frame, traffic_lights, args.max_distance, image_wh,
                                            max_incidence_deg=args.max_incidence_deg)
        matches = match_boxes(detections, candidates, gate_factor=args.gate_factor)

        stats["frames"] += 1
        stats["detections"] += len(detections)
        for i, det in enumerate(detections):
            if i not in matches:
                stats["unmatched:" + unmatched_reason(
                    det, canonical_state(det), candidates, matches, args.gate_factor)] += 1
        stats["map_candidates"] += len(candidates)
        stats["matched"] += len(matches)

        sd = frame["sample_data"]
        matched_cand_idx = set(matches.values())
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

        for i, det in enumerate(detections):
            token = stable_token(sd["token"], i, *det["box_xyxy"])
            cand = candidates[matches[i]] if i in matches else None
            reg_ids = regulatory_by_way.get(cand["way_id"], []) if cand else []
            pair_iou = iou(det["box_xyxy"], cand["bbox"]) if cand else 0.0
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
                    "attributes": {
                        "state": canonical_state(det),
                        "signal_kind": signal_kind(parse_state(raw_state(det))),
                        "visibility": "unknown",
                        "review_status": "unchecked",
                        "map_traffic_light_id": cand["way_id"] if cand else "",
                        "regulatory_element_id": ",".join(reg_ids),
                        "facing": cand["facing"] if cand else "",
                        "raw_state": raw_state(det),
                        "detector_score": ("" if det.get("detector_score") is None
                                           else f"{det['detector_score']}"),
                        "source_type": "auto",
                    },
                }
            )
            frame_report["pairs"].append(
                {
                    "detection_box": det["box_xyxy"],
                    "detector_score": det.get("detector_score"),
                    "signal": raw_state(det),
                    "map_traffic_light_id": cand["way_id"] if cand else None,
                    "map_subtype": cand["subtype"] if cand else None,
                    "regulatory_element_ids": reg_ids,
                    "projected_box": cand["bbox"] if cand else None,
                    "distance_m": cand["distance_m"] if cand else None,
                    "iou": round(pair_iou, 3),
                    "unmatched_reason": (None if cand else unmatched_reason(
                        det, canonical_state(det), candidates, matches, args.gate_factor)),
                }
            )
        report_frames.append(frame_report)

        if vis_left > 0 and detections:
            draw_overlay(root, frame, detections, candidates, matches,
                         root / args.vis_dir / f"{path.stem}.jpg")
            vis_left -= 1

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
        report_path.write_text(json.dumps(
            {"params": {"max_distance": args.max_distance, "gate_factor": args.gate_factor,
                        "min_score": args.min_score},
             "stats": dict(stats), "frames": report_frames}, indent=2))
        print(f"wrote {report_path}")

    matched_pct = 100.0 * stats["matched"] / max(stats["detections"], 1)
    print(f"frames={stats['frames']} detections={stats['detections']} "
          f"matched={stats['matched']} ({matched_pct:.1f}%)")


if __name__ == "__main__":
    main()
