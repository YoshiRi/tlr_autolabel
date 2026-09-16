"""Two signals bolted back to back on one mast.

They are two separate map ways whose face normals point opposite ways -- which
is correct mapping, not a winding error -- but they sit under a metre apart and
so project to nearly the same box. Geometry has no reason to prefer either, and
the assignment was picking the back one often enough to produce every
"colored state on a back face" flag in the Odaiba runs (222 of them across
three recordings, all on the back member of such a pair).

A readable colour settles it: you cannot read a lamp through the back of the
housing. That is knowledge no box overlap contains, so it enters as a cost
penalty rather than as geometry -- and only when the front partner is also in
view, so a match is re-routed and never lost.

Fixture: a 1.2 x 0.45 m housing 40 m ahead plus its back-to-back partner 0.8 m
further on, seen by a telephoto camera (f=6240 on 2880x1860).
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tlr_autolabel.map.association import match_boxes
from tlr_autolabel.map.lanelet2 import (
    find_back_to_back_pairs,
    load_lanelet2_traffic_lights,
)
from tlr_autolabel.map.projection import project_traffic_lights
from tlr_autolabel.t4.index import load_t4_index

ROOT = Path(__file__).resolve().parents[1]
FRONT_WAY, BACK_WAY = "101", "102"
IMAGE_WH = (2880, 1860)
FOCAL_PX = 6240.0
CAM_ROTATION_WXYZ = [0.7071067811865476, -0.7071067811865476, 0.0, 0.0]


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def tl(way_id, corners, facing_axis, subtype="red_yellow_green"):
    """A traffic_lights[] entry as load_lanelet2_traffic_lights() builds it."""
    return way_id, {
        "corners": np.asarray(corners, dtype=float),
        "subtype": subtype,
        "height": 0.45,
        "facing_axis": None if facing_axis is None else np.asarray(facing_axis, dtype=float),
    }


def bar(x, y, z, width=1.2):
    """Four corners of a housing centred at (x, y, z)."""
    h = width / 2
    return [[x - h, y, z], [x + h, y, z], [x - h, y, z + 0.45], [x + h, y, z + 0.45]]


class FindPairsTest(unittest.TestCase):
    def test_opposed_normals_within_a_metre_are_paired(self):
        lights = dict([tl("a", bar(0, 40, 4), (0, -1)),
                       tl("b", bar(0, 40.8, 4), (0, 1))])
        self.assertEqual(find_back_to_back_pairs(lights), {"a": "b", "b": "a"})

    def test_result_is_symmetric(self):
        lights = dict([tl("a", bar(0, 40, 4), (0, -1)),
                       tl("b", bar(0, 40.8, 4), (0, 1))])
        pairs = find_back_to_back_pairs(lights)
        self.assertTrue(all(pairs[v] == k for k, v in pairs.items()))

    def test_signals_facing_the_same_way_are_not_a_mast(self):
        # two signals over adjacent lanes: close, but both face the traffic
        lights = dict([tl("a", bar(0, 40, 4), (0, -1)),
                       tl("b", bar(2, 40, 4), (0, -1))])
        self.assertEqual(find_back_to_back_pairs(lights), {})

    def test_perpendicular_normals_are_not_a_mast(self):
        # the cross-street signal at the same intersection
        lights = dict([tl("a", bar(0, 40, 4), (0, -1)),
                       tl("b", bar(1, 40, 4), (1, 0))])
        self.assertEqual(find_back_to_back_pairs(lights), {})

    def test_too_far_apart_is_not_a_mast(self):
        lights = dict([tl("a", bar(0, 40, 4), (0, -1)),
                       tl("b", bar(0, 45, 4), (0, 1))])
        self.assertEqual(find_back_to_back_pairs(lights), {})

    def test_height_difference_rules_out_a_mast(self):
        # a signal directly above another is not its back face
        lights = dict([tl("a", bar(0, 40, 4), (0, -1)),
                       tl("b", bar(0, 40.5, 9), (0, 1))])
        self.assertEqual(find_back_to_back_pairs(lights), {})

    def test_three_on_one_mast_yield_one_pair_not_a_chain(self):
        lights = dict([tl("a", bar(0, 40.0, 4), (0, -1)),
                       tl("b", bar(0, 40.4, 4), (0, 1)),
                       tl("c", bar(0, 41.2, 4), (0, 1))])
        pairs = find_back_to_back_pairs(lights)
        # a-b are closest, so they pair and c is left alone rather than
        # producing a half-relation
        self.assertEqual(pairs, {"a": "b", "b": "a"})
        self.assertTrue(all(pairs[v] == k for k, v in pairs.items()))

    def test_way_without_a_facing_axis_is_skipped(self):
        lights = dict([tl("a", bar(0, 40, 4), None),
                       tl("b", bar(0, 40.8, 4), (0, 1))])
        self.assertEqual(find_back_to_back_pairs(lights), {})

    def test_thresholds_are_adjustable(self):
        lights = dict([tl("a", bar(0, 40, 4), (0, -1)),
                       tl("b", bar(0, 44, 4), (0, 1))])
        self.assertEqual(find_back_to_back_pairs(lights), {})
        self.assertEqual(find_back_to_back_pairs(lights, max_distance=5.0),
                         {"a": "b", "b": "a"})


class PenaltyHookTest(unittest.TestCase):
    """The hook itself: a penalty must be able to move an assignment."""

    def setUp(self):
        # two candidates on top of each other; the detection sits exactly on B
        self.detections = [{"box_xyxy": [200, 100, 300, 140], "state": "red-circle"}]
        self.candidates = [
            {"way_id": "front", "bbox": [190, 100, 290, 140], "facing": "front",
             "subtype": "red_yellow_green", "distance_m": 40.0},
            {"way_id": "back", "bbox": [200, 100, 300, 140], "facing": "back",
             "subtype": "red_yellow_green", "distance_m": 40.0},
        ]

    def test_without_a_penalty_the_better_overlap_wins(self):
        matches, _ = match_boxes(self.detections, self.candidates)
        self.assertEqual(self.candidates[matches[0]]["way_id"], "back")

    def test_a_penalty_on_the_back_reroutes_to_the_front(self):
        def penalty(det, cand):
            return 1.0 if cand["facing"] == "back" else 0.0
        matches, _ = match_boxes(self.detections, self.candidates, penalty_fn=penalty)
        self.assertEqual(self.candidates[matches[0]]["way_id"], "front")

    def test_zero_penalty_changes_nothing(self):
        matches, _ = match_boxes(self.detections, self.candidates,
                                 penalty_fn=lambda d, c: 0.0)
        self.assertEqual(self.candidates[matches[0]]["way_id"], "back")

    def test_the_hook_also_reaches_the_staged_matcher(self):
        def penalty(det, cand):
            return 1.0 if cand["facing"] == "back" else 0.0
        matches, _ = match_boxes(self.detections, self.candidates, mode="staged",
                                 penalty_fn=penalty)
        self.assertEqual(self.candidates[matches[0]]["way_id"], "front")


def write_dataset(root: Path, *, with_partner=True):
    """One camera frame; way 101 faces the ego, way 102 is its back-to-back
    partner 0.8 m further away (omitted when with_partner is False)."""
    ann = root / "annotation"
    osm = root / "map" / "lanelet2_map.osm"
    osm.parent.mkdir(parents=True, exist_ok=True)
    partner = """
  <node id="3"><tag k="local_x" v="0.6"/><tag k="local_y" v="40.8"/><tag k="ele" v="4.0"/></node>
  <node id="4"><tag k="local_x" v="-0.6"/><tag k="local_y" v="40.8"/><tag k="ele" v="4.0"/></node>
  <way id="102">
    <nd ref="3"/><nd ref="4"/>
    <tag k="type" v="traffic_light"/>
    <tag k="subtype" v="red_yellow_green"/>
    <tag k="height" v="0.45"/>
  </way>
  <relation id="202">
    <member type="way" ref="102" role="refers"/>
    <tag k="type" v="regulatory_element"/>
    <tag k="subtype" v="traffic_light"/>
  </relation>
""" if with_partner else ""
    osm.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1"><tag k="local_x" v="-0.6"/><tag k="local_y" v="40.0"/><tag k="ele" v="4.0"/></node>
  <node id="2"><tag k="local_x" v="0.6"/><tag k="local_y" v="40.0"/><tag k="ele" v="4.0"/></node>
  <way id="101">
    <nd ref="1"/><nd ref="2"/>
    <tag k="type" v="traffic_light"/>
    <tag k="subtype" v="red_yellow_green"/>
    <tag k="height" v="0.45"/>
  </way>
  <relation id="201">
    <member type="way" ref="101" role="refers"/>
    <tag k="type" v="regulatory_element"/>
    <tag k="subtype" v="traffic_light"/>
  </relation>{partner}
</osm>
""")
    write_json(ann / "sensor.json", [
        {"token": "sensor-cam", "channel": "CAM_TRAFFIC_LIGHT_FAR", "modality": "camera"}])
    write_json(ann / "calibrated_sensor.json", [{
        "token": "calib-cam", "sensor_token": "sensor-cam",
        "translation": [0.0, 0.0, 4.0], "rotation": CAM_ROTATION_WXYZ,
        "camera_intrinsic": [[FOCAL_PX, 0.0, IMAGE_WH[0] / 2],
                             [0.0, FOCAL_PX, IMAGE_WH[1] / 2], [0.0, 0.0, 1.0]]}])
    write_json(ann / "ego_pose.json", [
        {"token": "ego-0", "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}])
    write_json(ann / "sample_data.json", [{
        "token": "sd-0", "sample_token": "sample-0", "ego_pose_token": "ego-0",
        "calibrated_sensor_token": "calib-cam",
        "filename": "CAM_TRAFFIC_LIGHT_FAR/00000.jpg", "timestamp": 0,
        "width": IMAGE_WH[0], "height": IMAGE_WH[1]}])
    (root / "tlr_autolabel").mkdir(parents=True, exist_ok=True)


def candidates_of(root: Path) -> dict:
    traffic_lights, _ = load_lanelet2_traffic_lights(root / "map/lanelet2_map.osm")
    _, by_token = load_t4_index(root)
    cands = project_traffic_lights(by_token["sd-0"], traffic_lights, 200.0, IMAGE_WH)
    return {c["way_id"]: c for c in cands}


def write_l1(root: Path, box, state, lamp_confidence=0.9):
    lamps = []
    if state != "unknown":
        color, shape = state.split("-", 1)
        lamps = [{"label": state, "color": color, "shape": shape, "arrow": None,
                  "confidence": lamp_confidence}]
    write_json(root / "tlr_autolabel" / "00000.json", {
        "schema_version": "tlr_autolabel/v1",
        "image": "CAM_TRAFFIC_LIGHT_FAR/00000.jpg",
        "sample_data_token": "sd-0", "channel": "CAM_TRAFFIC_LIGHT_FAR",
        "frame_index": 0, "width": IMAGE_WH[0], "height": IMAGE_WH[1],
        "meta": {"run_id": "back-to-back"},
        "signals": [{"signal_id": "00000-00", "detector_score": 0.9,
                     "box_xyxy": [round(v, 2) for v in box], "state": state,
                     "lamps": lamps}]})


def match(root: Path, penalty: str):
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "match_traffic_lights.py"),
         "--dataset-root", str(root),
         "--output", f"annotation/p{penalty}.json",
         "--report", f"build/p{penalty}.json",
         "--no-fill-gaps", "--no-map-fill", "--min-score", "0.5",
         "--back-to-back-penalty", penalty],
        cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    anns = json.loads((root / f"annotation/p{penalty}.json").read_text())["annotations"]
    stats = json.loads((root / f"build/p{penalty}.json").read_text())["stats"]
    return [a for a in anns if a["attributes"]["source_type"] == "auto"], stats, out.stdout


class FixtureGeometryTest(unittest.TestCase):
    def test_the_pair_projects_as_a_front_and_a_back_almost_on_top(self):
        with tempfile.TemporaryDirectory(prefix="tlr_btb_geom_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            c = candidates_of(root)
            self.assertEqual(set(c), {FRONT_WAY, BACK_WAY})
            self.assertEqual(c[FRONT_WAY]["facing"], "front")
            self.assertEqual(c[BACK_WAY]["facing"], "back")
            fx = (c[FRONT_WAY]["bbox"][0] + c[FRONT_WAY]["bbox"][2]) / 2
            bx = (c[BACK_WAY]["bbox"][0] + c[BACK_WAY]["bbox"][2]) / 2
            self.assertLess(abs(fx - bx), 5.0, "the two should nearly coincide")


class BackToBackMatchTest(unittest.TestCase):
    def _on_the_back(self, root):
        """A detection sitting exactly on the back way's projection, so without
        the penalty the back way is the better overlap."""
        return candidates_of(root)[BACK_WAY]["bbox"]

    def test_penalty_off_lands_on_the_back_way(self):
        with tempfile.TemporaryDirectory(prefix="tlr_btb_off_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, self._on_the_back(root), "red-circle")
            rows, _, _ = match(root, "0")
            self.assertEqual(rows[0]["attributes"]["map_traffic_light_id"], BACK_WAY)
            self.assertEqual(rows[0]["attributes"]["facing"], "back")

    def test_penalty_reroutes_the_same_detection_to_the_front_way(self):
        with tempfile.TemporaryDirectory(prefix="tlr_btb_on_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, self._on_the_back(root), "red-circle")
            rows, stats, stdout = match(root, "1.0")
            self.assertEqual(rows[0]["attributes"]["map_traffic_light_id"], FRONT_WAY)
            self.assertEqual(rows[0]["attributes"]["facing"], "front")
            # re-routed, not dropped
            self.assertEqual(stats["matched"], 1)
            self.assertEqual(stats["back_face_penalty_applied"], 1)
            self.assertIn("back-to-back map pairs: 1", stdout)

    def test_an_unreadable_detection_is_left_alone(self):
        # no colour was read, so nothing rules out the back face
        with tempfile.TemporaryDirectory(prefix="tlr_btb_unknown_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, self._on_the_back(root), "unknown")
            rows, stats, _ = match(root, "1.0")
            self.assertEqual(rows[0]["attributes"]["map_traffic_light_id"], BACK_WAY)
            self.assertEqual(stats.get("back_face_penalty_applied", 0), 0)

    def test_a_lone_back_way_keeps_its_match(self):
        # a back face with no partner in the map is not a back-to-back case;
        # penalising it would lose the match for nothing
        with tempfile.TemporaryDirectory(prefix="tlr_btb_lone_") as tmp:
            root = Path(tmp)
            write_dataset(root, with_partner=False)
            box = candidates_of(root)[FRONT_WAY]["bbox"]
            write_l1(root, box, "red-circle")
            rows, stats, _ = match(root, "1.0")
            self.assertEqual(rows[0]["attributes"]["map_traffic_light_id"], FRONT_WAY)
            self.assertEqual(stats["matched"], 1)
            self.assertEqual(stats.get("back_face_penalty_applied", 0), 0)

    def test_no_colored_state_survives_on_a_back_face(self):
        with tempfile.TemporaryDirectory(prefix="tlr_btb_flag_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, self._on_the_back(root), "green-circle")
            rows, _, _ = match(root, "1.0")
            offenders = [r for r in rows
                         if r["attributes"]["facing"] == "back"
                         and r["attributes"]["state"] not in ("unknown", "")]
            self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
