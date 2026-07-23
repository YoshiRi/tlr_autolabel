#!/usr/bin/env python3
"""Aggregate 2D traffic light annotations into regulatory-element time series.

Consumes annotation/traffic_signal_2d_ann.json (bboxes already associated to
lanelet2 `traffic_light` ways by tools/match_traffic_lights.py) and produces:

  1. Way-level observations: per (sample, traffic_light way) fusion of all
     camera detections, with cross-camera agreement voting.
  2. Map bulb feasibility checks: detected colors/arrow directions are
     validated against the `light_bulbs` ways of the lanelet2 map.
  3. Regulatory-element-level time series: per relation
     (type=regulatory_element subtype=traffic_light), fusing all member heads
     (role=refers), with cross-head agreement voting.
  4. Temporal checks: single-frame state flips inside otherwise-stable runs.

Outputs:
  - annotation/traffic_signal_re_timeseries.json  (traffic_signal_re/v1)
  - build/tl_match/re_verification_report.json    (flag summary for review)

Extension points (not implemented here): fusing an onboard TLR topic from the
rosbag, and per-bulb projection checks using lamp positions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


# ------------------------------------------------------------- detector signal
# Token parsing is shared with the rest of the repo (canonical + legacy forms).
from state_tokens import bulb_color, elements_key, parse_state

# box shorter side (px) at/above which a lamp read is treated as fully reliable;
# smaller boxes get proportionally less vote weight (cross-camera resolution
# weighting, e.g. wide vs telephoto observing the same head — see way fusion).
SIZE_REF_PX = 30.0


# ---------------------------------------------------------------- lanelet2 map


def load_map(osm_path: Path):
    """Return (bulbs_by_way, ways_by_relation).

    bulbs_by_way: {tl_way_id: {"colors": set, "arrows": set of (color, dir)}}
    ways_by_relation: {relation_id: [tl_way_id, ...]} for regulatory elements.
    """
    root = ET.parse(osm_path).getroot()
    node_tags = {
        n.get("id"): {t.get("k"): t.get("v") for t in n.findall("tag")}
        for n in root.iter("node")
    }

    bulbs_by_way: dict[str, dict] = defaultdict(lambda: {"colors": set(), "arrows": set()})
    for way in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if tags.get("type") != "light_bulbs":
            continue
        tl_id = tags.get("traffic_light_id")
        if not tl_id:
            continue
        entry = bulbs_by_way[tl_id]
        for nd in way.findall("nd"):
            ntags = node_tags.get(nd.get("ref"), {})
            color = ntags.get("color")
            if not color:
                continue
            entry["colors"].add(color)
            if "arrow" in ntags:
                entry["arrows"].add((color, ntags["arrow"]))

    ways_by_relation: dict[str, list[str]] = {}
    for rel in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
        if tags.get("type") != "regulatory_element" or tags.get("subtype") != "traffic_light":
            continue
        refs = [m.get("ref") for m in rel.findall("member")
                if m.get("role") == "refers" and m.get("type") == "way"]
        if refs:
            ways_by_relation[rel.get("id")] = refs

    return bulbs_by_way, ways_by_relation


# ------------------------------------------------------------------ bulb check


def bulb_flags(elements: list[dict], bulbs: dict | None) -> list[str]:
    if bulbs is None:
        return ["no_bulb_info_in_map"] if elements else []
    flags = []
    for e in elements:
        if e["shape"] == "ped":
            # a pedestrian lamp reading on a 3-color/arrow (vehicle) bulb row
            # points at a wrong association or detector confusion
            if "yellow" in bulbs["colors"] or bulbs["arrows"]:
                flags.append("ped_on_vehicle_bulbs")
            continue
        if bulb_color(e["color"]) not in bulbs["colors"]:
            flags.append(f"color_not_in_map_bulbs:{e['color']}")
        if e["shape"] == "arrow":
            if not bulbs["arrows"]:
                flags.append(f"arrow_without_map_bulb:{e['arrow']}")
            elif e["arrow"] and (bulb_color(e["color"]), e["arrow"]) not in bulbs["arrows"]:
                known = ",".join(sorted(d for _, d in bulbs["arrows"]))
                flags.append(f"arrow_dir_mismatch:{e['arrow']}(map:{known})")
    return sorted(set(flags))


def snap_arrow_dirs(elements: list[dict], bulbs: dict | None):
    """Snap an arrow's direction to the map bulb layout when the detected
    direction is infeasible AND the map offers exactly one direction for that
    color (8-way classifiers often miss by one sector; the map is authoritative
    for which arrows physically exist). Colors/shapes are never snapped — a
    color mismatch may equally be a stale map. Returns (elements, snaps)."""
    if not bulbs or not bulbs["arrows"]:
        return elements, []
    snapped, snaps = [], []
    for e in elements:
        if e["shape"] == "arrow" and e.get("arrow"):
            bc = bulb_color(e["color"])
            if (bc, e["arrow"]) not in bulbs["arrows"]:
                dirs = sorted({d for c, d in bulbs["arrows"] if c == bc})
                if len(dirs) == 1:
                    snaps.append(f"arrow_dir_snapped:{e['arrow']}->{dirs[0]}")
                    e = dict(e, arrow=dirs[0])
        snapped.append(e)
    return snapped, snaps


def bulb_weight(elements: list[dict], bulbs: dict | None) -> float:
    """Vote weight: states inconsistent with the map's bulb layout count less."""
    return 0.25 ** len(bulb_flags(elements, bulbs))


# -------------------------------------------------------------------- fusion


def vote(groups: list[tuple[str, list[dict], float]]):
    """Vote on (key, elements, weight) triples. Unknown ('') abstains.

    Returns (winner_key, winner_elements, votes Counter, confidence, flags).
    """
    votes = Counter()
    by_key = {}
    for key, elements, weight in groups:
        if key == "":
            continue
        votes[key] += weight
        by_key[key] = elements
    flags = []
    if not votes:
        return "", [], votes, 0.0, flags
    winner, winner_weight = max(votes.items(), key=lambda kv: kv[1])
    total = sum(votes.values())
    if len(votes) > 1:
        flags.append("state_disagreement")
    if any(key == "" for key, _e, _w in groups):
        flags.append("partial_unknown")
    return winner, by_key[winner], votes, winner_weight / total, flags


# ------------------------------------------------------------------------ main


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--input", default=Path("annotation/traffic_signal_2d_ann.json"), type=Path)
    parser.add_argument("--output", default=Path("annotation/traffic_signal_re_timeseries.json"), type=Path)
    parser.add_argument("--report", default=Path("build/tl_match/re_verification_report.json"), type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.dataset_root.resolve()

    bulbs_by_way, ways_by_relation = load_map(root / "map/lanelet2_map.osm")
    sidecar = json.loads((root / args.input).read_text())
    annotations = [a for a in sidecar["annotations"]
                   if a["attributes"].get("map_traffic_light_id")]

    sample_ts = {s["token"]: s["timestamp"]
                 for s in json.loads((root / "annotation/sample.json").read_text())}

    # ---- way-level fusion: (sample_token, way_id) -> observation
    by_sample_way = defaultdict(list)
    for a in annotations:
        by_sample_way[(a["sample_token"], a["attributes"]["map_traffic_light_id"])].append(a)

    way_obs: dict[tuple[str, str], dict] = {}
    for (sample_token, way_id), anns in by_sample_way.items():
        bulbs = bulbs_by_way.get(way_id)
        groups = []
        snap_flags = []
        for a in anns:
            elements = parse_state(a["attributes"].get("state")
                                   or a["attributes"].get("raw_state")
                                   or a["attributes"].get("detector_signal", ""))
            elements, snaps = snap_arrow_dirs(elements, bulbs)
            snap_flags += snaps
            weight = bulb_weight(elements, bulbs)
            # cross-camera resolution weighting: a head seen by both a wide and
            # a telephoto (e.g. CAM_FRONT vs CAM_FRONT_FAR) at distance gives a
            # ~10px vs ~44px box for the same signal. More pixels on the lamp =
            # more reliable classification, so the bigger box should win a
            # disagreement. Saturates at SIZE_REF_PX (above it, both reads are
            # reliable and near-field behaviour is unchanged); floored so a tiny
            # box still votes a little rather than abstaining.
            box = a.get("box2d")
            if box:
                min_side = min(box[2] - box[0], box[3] - box[1])
                weight *= min(max(min_side / SIZE_REF_PX, 0.2), 1.0)
            # a colored reading through the housing's back is physically
            # impossible -> almost certainly a misassociation; flag + demote
            if elements and a["attributes"].get("facing") == "back":
                snap_flags.append("colored_state_on_back_face")
                weight *= 0.25
            groups.append((elements_key(elements), elements, weight))
        winner_key, winner_elements, votes, confidence, flags = vote(groups)
        flags = ["cross_camera_" + f if f == "state_disagreement" else f for f in flags]
        flags += snap_flags
        flags += bulb_flags(winner_elements, bulbs)
        way_obs[(sample_token, way_id)] = {
            "timestamp": sample_ts.get(sample_token),
            "state": winner_key or "unknown",
            "elements": winner_elements,
            "n_sources": len(anns),
            "channels": sorted({a["channel"] for a in anns}),
            "annotation_tokens": [a["token"] for a in anns],
            "votes": {k: round(v, 3) for k, v in votes.items()},
            "confidence": round(confidence, 3),
            "flags": sorted(set(flags)),
        }

    # ---- regulatory-element-level fusion and time series
    obs_ways_by_sample = defaultdict(set)
    for (sample_token, way_id) in way_obs:
        obs_ways_by_sample[sample_token].add(way_id)

    series = []
    all_flag_counter = Counter()
    flagged_observations = []
    for rel_id, member_ways in sorted(ways_by_relation.items(), key=lambda kv: kv[0]):
        observations = []
        for sample_token, seen_ways in obs_ways_by_sample.items():
            active = [w for w in member_ways if w in seen_ways]
            if not active:
                continue
            groups, head_states, sub_flags = [], {}, []
            for w in active:
                o = way_obs[(sample_token, w)]
                key = "" if o["state"] == "unknown" else o["state"]
                # head weight: its own fused confidence, scaled again by how
                # well its state fits this head's map bulb layout
                weight = max(o["confidence"], 0.1) * bulb_weight(
                    o["elements"], bulbs_by_way.get(w))
                groups.append((key, o["elements"], weight))
                head_states[w] = o["state"]
                sub_flags += o["flags"]
            winner_key, winner_elements, votes, confidence, flags = vote(groups)
            flags = ["cross_head_" + f if f == "state_disagreement" else f for f in flags]
            flags += sub_flags
            observations.append({
                "sample_token": sample_token,
                "timestamp": sample_ts.get(sample_token),
                "state": winner_key or "unknown",
                "elements": winner_elements,
                "head_states": head_states,
                "n_heads": len(active),
                "confidence": round(confidence, 3),
                "flags": sorted(set(flags)),
            })
        if not observations:
            continue
        observations.sort(key=lambda o: o["timestamp"] or 0)

        # temporal repair: a single-frame flip inside an otherwise-stable run is
        # corrected by its neighbors (raw value kept in state_original; per-head
        # states stay untouched). Decided 2026-07-19: repair, don't just flag.
        before = [o["state"] for o in observations]
        for i in range(1, len(observations) - 1):
            prev_s, cur, next_s = before[i - 1], observations[i], before[i + 1]
            if prev_s == next_s and before[i] != prev_s and "unknown" not in (
                    prev_s, before[i]):
                cur["state_original"] = cur["state"]
                cur["state"] = prev_s
                cur["elements"] = [dict(e) for e in observations[i - 1]["elements"]]
                cur["state_source"] = "temporal_fix"
                cur["flags"] = sorted(set(cur["flags"] + ["single_frame_flip_fixed"]))

        # run-length segments for readability
        segments = []
        for o in observations:
            if segments and segments[-1]["state"] == o["state"]:
                segments[-1]["t_end"] = o["timestamp"]
                segments[-1]["n_frames"] += 1
            else:
                segments.append({"state": o["state"], "t_start": o["timestamp"],
                                 "t_end": o["timestamp"], "n_frames": 1})

        for o in observations:
            for f in o["flags"]:
                all_flag_counter[f] += 1
            if o["flags"]:
                flagged_observations.append({
                    "regulatory_element_id": rel_id,
                    "sample_token": o["sample_token"],
                    "timestamp": o["timestamp"],
                    "state": o["state"],
                    "head_states": o["head_states"],
                    "flags": o["flags"],
                    # review triage: more flags and less voting agreement first
                    "review_priority": round(len(o["flags"]) + (1 - o["confidence"]), 3),
                })
        series.append({
            "regulatory_element_id": rel_id,
            "member_ways": member_ways,
            "n_observations": len(observations),
            "segments": segments,
            "observations": observations,
        })

    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"schema_version": "traffic_signal_re/v1",
         "source": str(args.input),
         "series": series}, indent=2))

    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(
        {"flag_counts": dict(all_flag_counter),
         "n_series": len(series),
         "n_way_observations": len(way_obs),
         "flagged_observations": sorted(
             flagged_observations,
             key=lambda f: (-f["review_priority"], f["regulatory_element_id"], f["timestamp"] or 0)),
         }, indent=2))

    print(f"series (regulatory elements observed): {len(series)}")
    print(f"way-level observations: {len(way_obs)}")
    print(f"flag counts: {dict(all_flag_counter)}")
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
