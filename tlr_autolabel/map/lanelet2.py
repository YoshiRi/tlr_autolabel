"""Lanelet2 traffic-light map parsing (REFACTOR_PLAN.md phase 5).

Extracted from match_traffic_lights.py.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


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


# Way types worth drawing as road context. Lane boundaries are skipped: the
# lanelet polygons below already cover them, and drawing both is unreadable.
CONTEXT_WAY_TYPES = ("intersection_area", "crosswalk_polygon", "stop_line")


def load_lanelet2_context(osm_path: Path, way_types=CONTEXT_WAY_TYPES):
    """Return (lanelets, ways) for drawing road shape around the ego path.

    lanelets: [{"id", "subtype", "left": [(x, y)...], "right": [(x, y)...]}]
      -- a drivable/walkable area is the polygon left + reversed(right).
    ways: [{"id", "type", "points": [(x, y)...]}] for `way_types`.

    Separate from load_lanelet2_traffic_lights() because matching needs only
    the signals, while visualisation needs the surrounding geometry.
    """
    root = ET.parse(osm_path).getroot()

    nodes: dict[str, tuple[float, float]] = {}
    for node in root.iter("node"):
        tags = {t.get("k"): t.get("v") for t in node.findall("tag")}
        if "local_x" in tags and "local_y" in tags:
            nodes[node.get("id")] = (float(tags["local_x"]), float(tags["local_y"]))

    wanted = set(way_types)
    way_points: dict[str, list[tuple[float, float]]] = {}
    context_ways: list[dict] = []
    for way in root.iter("way"):
        refs = [nd.get("ref") for nd in way.findall("nd")]
        pts = [nodes[r] for r in refs if r in nodes]
        if len(pts) < 2:
            continue
        way_points[way.get("id")] = pts
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if tags.get("type") in wanted:
            context_ways.append(
                {"id": way.get("id"), "type": tags.get("type"), "points": pts}
            )

    lanelets: list[dict] = []
    for rel in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
        if tags.get("type") != "lanelet":
            continue
        bounds = {
            member.get("role"): way_points.get(member.get("ref"))
            for member in rel.findall("member")
            if member.get("role") in ("left", "right")
        }
        if not bounds.get("left") or not bounds.get("right"):
            continue
        lanelets.append({
            "id": rel.get("id"),
            "subtype": tags.get("subtype", "") or "",
            "left": bounds["left"],
            "right": bounds["right"],
        })

    return lanelets, context_ways
