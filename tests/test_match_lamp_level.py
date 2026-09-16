"""End-to-end: a lamp-level L1 payload through match_traffic_lights.py.

Reproduces the failure the lamp-level matching mode was built for, with the
real scene's geometry rather than a toy one, because the failure only appears
at realistic proportions:

  * The map projects the whole housing. A lamp-level L1 (CoMLOps-style
    detectors, CoMET `--only-tlr`) boxes one lit lamp, ~0.22 of the housing
    width, so the legacy cost's size-ratio bound (2.5) is blown ~4x and the
    distance fallback never runs.
  * The projection is also laterally offset -- measured at ~0.36 of the housing
    width on the Komatsu run, uniform across all four ways, so it is a
    projection offset, not per-signal map error. A lamp at the end of the
    housing (red is rightmost in a JP horizontal signal) therefore falls
    *outside* the projected box, and IoU is 0 rather than merely small.

Together those two put red-circle at 0/483 matched. Comparing the lamp against
its slot inside the projection fixes both: 483/483, and the run overall goes
21.9% -> 61.1% (the same scene's housing-level L1 scores 52.5%, or 62.8% once
its 257 stateless back-of-housing detections -- which a lamp-level producer
never emits -- are excluded).

The fixture below is a 1.2m x 0.45m vehicle housing 40m ahead of a telephoto
camera (f=6240px on 2880x1860), which projects to ~187x70px: the same size the
real CAM_TRAFFIC_LIGHT_FAR run produced at 30-50m.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tlr_autolabel.map.lamp_geometry import expected_lamp_box, housing_axis
from tlr_autolabel.map.lanelet2 import load_lanelet2_traffic_lights
from tlr_autolabel.map.projection import project_traffic_lights
from tlr_autolabel.t4.index import load_t4_index

ROOT = Path(__file__).resolve().parents[1]

HOUSING_WIDTH_M = 1.2
HOUSING_HEIGHT_M = 0.45
SIGNAL_DEPTH_M = 40.0
FOCAL_PX = 6240.0
IMAGE_WH = (2880, 1860)
# lamp box width as a fraction of the projected housing width (measured 0.25)
LAMP_WIDTH_FRACTION = 0.22
# lateral projection offset as a fraction of the housing width (measured 0.36)
PROJECTION_OFFSET_FRACTION = 0.36

# camera looking along map +y with map +z as image up: a -90 deg roll about x
CAM_ROTATION_WXYZ = [0.7071067811865476, -0.7071067811865476, 0.0, 0.0]


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_dataset(root: Path) -> None:
    """One camera frame, one 1.2x0.45m vehicle signal 40m straight ahead."""
    ann = root / "annotation"
    half = HOUSING_WIDTH_M / 2
    osm = root / "map" / "lanelet2_map.osm"
    osm.parent.mkdir(parents=True, exist_ok=True)
    osm.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1"><tag k="local_x" v="{-half}"/><tag k="local_y" v="{SIGNAL_DEPTH_M}"/><tag k="ele" v="4.0"/></node>
  <node id="2"><tag k="local_x" v="{half}"/><tag k="local_y" v="{SIGNAL_DEPTH_M}"/><tag k="ele" v="4.0"/></node>
  <way id="101">
    <nd ref="1"/><nd ref="2"/>
    <tag k="type" v="traffic_light"/>
    <tag k="subtype" v="red_yellow_green"/>
    <tag k="height" v="{HOUSING_HEIGHT_M}"/>
  </way>
  <relation id="201">
    <member type="way" ref="101" role="refers"/>
    <tag k="type" v="regulatory_element"/>
    <tag k="subtype" v="traffic_light"/>
  </relation>
</osm>
""")
    write_json(ann / "sensor.json", [
        {"token": "sensor-cam", "channel": "CAM_TRAFFIC_LIGHT_FAR", "modality": "camera"}
    ])
    write_json(ann / "calibrated_sensor.json", [{
        "token": "calib-cam",
        "sensor_token": "sensor-cam",
        "translation": [0.0, 0.0, 4.0],
        "rotation": CAM_ROTATION_WXYZ,
        "camera_intrinsic": [[FOCAL_PX, 0.0, IMAGE_WH[0] / 2],
                             [0.0, FOCAL_PX, IMAGE_WH[1] / 2],
                             [0.0, 0.0, 1.0]],
    }])
    write_json(ann / "ego_pose.json", [
        {"token": "ego-0", "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
    ])
    write_json(ann / "sample_data.json", [{
        "token": "sd-0",
        "sample_token": "sample-0",
        "ego_pose_token": "ego-0",
        "calibrated_sensor_token": "calib-cam",
        "filename": "CAM_TRAFFIC_LIGHT_FAR/00000.jpg",
        "timestamp": 0,
        "width": IMAGE_WH[0],
        "height": IMAGE_WH[1],
    }])
    (root / "tlr_autolabel").mkdir(parents=True, exist_ok=True)


def projected_housing(root: Path) -> dict:
    traffic_lights, _ = load_lanelet2_traffic_lights(root / "map/lanelet2_map.osm")
    _, frames_by_token = load_t4_index(root)
    candidates = project_traffic_lights(frames_by_token["sd-0"], traffic_lights,
                                        200.0, IMAGE_WH)
    assert len(candidates) == 1, candidates
    return candidates[0]


def write_lamp_level_l1(root: Path, color: str, *,
                        offset_fraction: float = PROJECTION_OFFSET_FRACTION) -> list[float]:
    """Write frame 0 as one lamp-level detection of `color`, placed where the
    lamp really appears: its slot inside the projection, shifted right by the
    measured projection offset. Returns the lamp box written."""
    cand = projected_housing(root)
    box = cand["bbox"]
    width = box[2] - box[0]
    slot = expected_lamp_box(cand, [{"color": color, "shape": "circle", "arrow": None}])
    cx = (slot[0] + slot[2]) / 2 + offset_fraction * width
    cy = (slot[1] + slot[3]) / 2
    half = LAMP_WIDTH_FRACTION * width / 2
    lamp = [round(cx - half, 2), round(cy - half, 2),
            round(cx + half, 2), round(cy + half, 2)]
    write_json(root / "tlr_autolabel" / "00000.json", {
        "schema_version": "tlr_autolabel/v1",
        "box_level": "lamp",
        "image": "CAM_TRAFFIC_LIGHT_FAR/00000.jpg",
        "sample_data_token": "sd-0",
        "channel": "CAM_TRAFFIC_LIGHT_FAR",
        "frame_index": 0,
        "width": IMAGE_WH[0],
        "height": IMAGE_WH[1],
        "meta": {"run_id": "synthetic-lamp-level"},
        "signals": [{
            "signal_id": "00000-lamp-00",
            "detector_score": 0.9,
            "box_xyxy": lamp,
            "state": f"{color}-circle",
            "lamps": [{"label": f"{color}-circle", "color": color, "shape": "circle",
                       "arrow": None, "confidence": 0.9}],
            "source_instance_token": "comet-instance-1",
        }],
    })
    return lamp


def write_housing_level_l1(root: Path, color: str) -> list[float]:
    """The same signal as a housing-level L1: box the whole housing."""
    box = [round(v, 2) for v in projected_housing(root)["bbox"]]
    write_json(root / "tlr_autolabel" / "00000.json", {
        "schema_version": "tlr_autolabel/v1",
        "image": "CAM_TRAFFIC_LIGHT_FAR/00000.jpg",
        "sample_data_token": "sd-0",
        "channel": "CAM_TRAFFIC_LIGHT_FAR",
        "frame_index": 0,
        "width": IMAGE_WH[0],
        "height": IMAGE_WH[1],
        "meta": {"run_id": "synthetic-housing-level"},
        "signals": [{
            "signal_id": "00000-00",
            "detector_score": 0.9,
            "box_xyxy": box,
            "state": f"{color}-circle",
            "lamps": [{"label": f"{color}-circle", "color": color, "shape": "circle",
                       "arrow": None, "confidence": 0.9}],
        }],
    })
    return box


def match(root: Path, mode: str, *extra: str) -> list[dict]:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "match_traffic_lights.py"),
         "--dataset-root", str(root),
         "--output", f"annotation/{mode}.json",
         "--report", f"build/{mode}_report.json",
         "--no-fill-gaps", "--no-map-fill",
         "--min-score", "0.5",
         "--lamp-level-matching", mode, *extra],
        cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    anns = json.loads((root / f"annotation/{mode}.json").read_text())["annotations"]
    return [a for a in anns if a["attributes"]["source_type"] == "auto"]


class FixtureGeometryTest(unittest.TestCase):
    """The scenario only tests anything if its proportions are the real ones."""

    def test_projection_matches_the_real_runs_scale(self):
        with tempfile.TemporaryDirectory(prefix="tlr_lamp_fixture_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            cand = projected_housing(root)
            box = cand["bbox"]
            width, height = box[2] - box[0], box[3] - box[1]
            self.assertEqual(housing_axis(box), "horizontal")
            self.assertAlmostEqual(width, HOUSING_WIDTH_M * FOCAL_PX / SIGNAL_DEPTH_M,
                                   delta=1.0)
            self.assertAlmostEqual(height, HOUSING_HEIGHT_M * FOCAL_PX / SIGNAL_DEPTH_M,
                                   delta=1.0)
            self.assertAlmostEqual(width / height, 2.67, delta=0.05)
            self.assertEqual(cand["facing"], "front")

    def test_the_offset_lamp_falls_outside_the_projection(self):
        # this is why IoU is 0 and not merely small -- the premise of the mode
        with tempfile.TemporaryDirectory(prefix="tlr_lamp_outside_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            lamp = write_lamp_level_l1(root, "red")
            box = projected_housing(root)["bbox"]
            self.assertGreater(lamp[0], box[2],
                               "red lamp should sit right of the projected housing")


class LampLevelMatchingTest(unittest.TestCase):
    def test_red_lamp_matches_only_with_the_mode_on(self):
        with tempfile.TemporaryDirectory(prefix="tlr_lamp_red_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_lamp_level_l1(root, "red")

            off = match(root, "off")
            on = match(root, "on")

            self.assertEqual(len(off), 1)
            self.assertEqual(off[0]["attributes"]["map_traffic_light_id"], "")
            self.assertEqual(off[0]["attributes"]["unmatched_reason"], "geometry_mismatch")

            self.assertEqual(len(on), 1)
            self.assertEqual(on[0]["attributes"]["map_traffic_light_id"], "101")
            self.assertEqual(on[0]["attributes"]["regulatory_element_id"], "201")
            self.assertEqual(on[0]["attributes"]["unmatched_reason"], "")

    def test_every_lamp_colour_lands_on_the_same_signal(self):
        for color in ("red", "amber", "green"):
            with tempfile.TemporaryDirectory(prefix=f"tlr_lamp_{color}_") as tmp:
                root = Path(tmp)
                write_dataset(root)
                write_lamp_level_l1(root, color)
                on = match(root, "on")
                self.assertEqual(on[0]["attributes"]["map_traffic_light_id"], "101",
                                 f"{color} did not match")

    def test_box_level_and_provenance_reach_the_sidecar(self):
        with tempfile.TemporaryDirectory(prefix="tlr_lamp_attrs_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_lamp_level_l1(root, "red")
            attrs = match(root, "on")[0]["attributes"]
            self.assertEqual(attrs["box_level"], "lamp")
            self.assertEqual(attrs["source_track_id"], "comet-instance-1")
            self.assertEqual(attrs["state"], "red-circle")

    def test_the_mode_does_not_change_housing_level_l1(self):
        results = {}
        for mode in ("off", "on"):
            with tempfile.TemporaryDirectory(prefix=f"tlr_housing_{mode}_") as tmp:
                root = Path(tmp)
                write_dataset(root)
                write_housing_level_l1(root, "red")
                anns = match(root, mode)
                results[mode] = [(a["attributes"]["map_traffic_light_id"],
                                  a["attributes"]["state"],
                                  a["attributes"]["box_level"]) for a in anns]
        self.assertEqual(results["off"], results["on"])
        self.assertEqual(results["on"], [("101", "red-circle", "housing")])


if __name__ == "__main__":
    unittest.main()
