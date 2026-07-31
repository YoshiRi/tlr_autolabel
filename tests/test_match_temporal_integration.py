import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from match_traffic_lights import load_lanelet2_traffic_lights, load_t4_index, project_traffic_lights


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_synthetic_dataset(root: Path, *, low_state: str = "green-circle",
                            low_lamps: list[dict] | None = None) -> list[float]:
    ann = root / "annotation"
    map_dir = root / "map"
    autolabel = root / "tlr_autolabel"
    ann.mkdir(parents=True)
    map_dir.mkdir()
    autolabel.mkdir()

    (map_dir / "lanelet2_map.osm").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1"><tag k="local_x" v="-0.5"/><tag k="local_y" v="5.0"/><tag k="ele" v="10.0"/></node>
  <node id="2"><tag k="local_x" v="0.5"/><tag k="local_y" v="5.0"/><tag k="ele" v="10.0"/></node>
  <way id="101">
    <nd ref="1"/><nd ref="2"/>
    <tag k="type" v="traffic_light"/>
    <tag k="subtype" v="red_yellow_green"/>
    <tag k="height" v="5.0"/>
  </way>
  <relation id="201">
    <member type="way" ref="101" role="refers"/>
    <tag k="type" v="regulatory_element"/>
    <tag k="subtype" v="traffic_light"/>
  </relation>
</osm>
"""
    )
    write_json(ann / "sensor.json", [
        {"token": "sensor-cam", "channel": "CAM_FRONT", "modality": "camera"}
    ])
    write_json(ann / "calibrated_sensor.json", [
        {
            "token": "calib-cam",
            "sensor_token": "sensor-cam",
            "translation": [0.0, 0.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "camera_intrinsic": [[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]],
        }
    ])
    write_json(ann / "ego_pose.json", [
        {"token": f"ego-{i}", "translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
        for i in range(3)
    ])
    write_json(ann / "sample_data.json", [
        {
            "token": f"sd-{i}",
            "sample_token": f"sample-{i}",
            "ego_pose_token": f"ego-{i}",
            "calibrated_sensor_token": "calib-cam",
            "filename": f"CAM_FRONT/{i:05d}.jpg",
            "timestamp": i,
        }
        for i in range(3)
    ])

    traffic_lights, _ = load_lanelet2_traffic_lights(map_dir / "lanelet2_map.osm")
    _, frames_by_token = load_t4_index(root)
    bbox = project_traffic_lights(frames_by_token["sd-0"], traffic_lights, 200.0, (200, 200))[0]["bbox"]
    high_box = [round(v, 2) for v in bbox]
    low_box = [round(bbox[0] + 1.0, 2), round(bbox[1] + 1.0, 2),
               round(bbox[2] + 1.0, 2), round(bbox[3] + 1.0, 2)]

    def payload(index: int, signals: list[dict], raw_detections: list[dict] | None = None):
        data = {
            "schema_version": "tlr_autolabel/v1",
            "image": f"CAM_FRONT/{index:05d}.jpg",
            "sample_data_token": f"sd-{index}",
            "channel": "CAM_FRONT",
            "frame_index": index,
            "width": 200,
            "height": 200,
            "meta": {"run_id": "synthetic"},
            "signals": signals,
        }
        if raw_detections is not None:
            data["raw_detections"] = raw_detections
        write_json(autolabel / f"{index:05d}.json", data)

    payload(0, [{
        "signal_id": "00000-00",
        "detector_score": 0.8,
        "box_xyxy": high_box,
        "lamps": [{"label": "red-circle", "color": "red", "shape": "circle", "arrow": None, "confidence": 0.9}],
        "state": "red-circle",
    }])
    if low_lamps is None:
        low_lamps = [{
            "label": "green-circle",
            "color": "green",
            "shape": "circle",
            "arrow": None,
            "confidence": 0.9,
        }]
    payload(1, [], [{
        "raw_detection_id": "00001-raw-00",
        "detector_score": 0.3,
        "box_xyxy": low_box,
        "lamps": low_lamps,
        "state": low_state,
        "detection_level": "low",
    }])
    payload(2, [], [])
    return high_box


def run_match(dataset_root: Path, *args):
    subprocess.run(
        [sys.executable, str(ROOT / "match_traffic_lights.py"), "--dataset-root", str(dataset_root), *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class MatchTemporalIntegrationTest(unittest.TestCase):
    def test_tracking_off_keeps_existing_high_only_contract(self):
        with tempfile.TemporaryDirectory(prefix="tlr_match_off_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root)

            run_match(
                root,
                "--output", "annotation/off.json",
                "--report", "build/off_report.json",
                "--no-fill-gaps",
                "--no-map-fill",
                "--min-score", "0.5",
            )

            sidecar = json.loads((root / "annotation/off.json").read_text())
            self.assertEqual([a["attributes"]["source_type"] for a in sidecar["annotations"]], ["auto"])
            self.assertNotIn("temporal_source", sidecar["annotations"][0]["attributes"])

            report = json.loads((root / "build/off_report.json").read_text())
            self.assertNotIn("temporal_tracking", report["params"])

    def test_tracking_config_alone_does_not_enable_tracking(self):
        with tempfile.TemporaryDirectory(prefix="tlr_match_cfg_off_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root)
            cfg = root / "tracking_enabled.yaml"
            cfg.write_text("temporal_tracking:\n  enabled: true\n  low_score: 0.2\n")

            run_match(
                root,
                "--output", "annotation/config_only.json",
                "--report", "build/config_only_report.json",
                "--no-fill-gaps",
                "--no-map-fill",
                "--min-score", "0.5",
                "--tracking-config", str(cfg),
            )

            sidecar = json.loads((root / "annotation/config_only.json").read_text())
            self.assertEqual([a["attributes"]["source_type"] for a in sidecar["annotations"]], ["auto"])

            report = json.loads((root / "build/config_only_report.json").read_text())
            self.assertNotIn("temporal_tracking", report["params"])

    def test_tracking_recovers_low_detection_and_propagates_short_miss(self):
        with tempfile.TemporaryDirectory(prefix="tlr_match_on_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root)

            run_match(
                root,
                "--output", "annotation/on.json",
                "--report", "build/on_report.json",
                "--no-fill-gaps",
                "--no-map-fill",
                "--min-score", "0.5",
                "--temporal-tracking",
                "--tracking-config", "configs/tracking/bytetrack-lite.yaml",
                "--tracking-max-lost-frames", "2",
            )

            sidecar = json.loads((root / "annotation/on.json").read_text())
            anns = sidecar["annotations"]
            self.assertEqual([a["attributes"]["source_type"] for a in anns], ["auto", "tracked", "propagated"])
            self.assertEqual([a["attributes"]["temporal_source"] for a in anns], ["observed", "observed", "propagated"])
            self.assertEqual({a["attributes"]["map_traffic_light_id"] for a in anns}, {"101"})
            self.assertEqual(len({(a["sample_data_token"], a["attributes"]["map_traffic_light_id"]) for a in anns}), 3)
            self.assertEqual(anns[2]["attributes"]["state"], "unknown")
            self.assertEqual(anns[0]["attributes"]["track_id"], anns[1]["attributes"]["track_id"])
            self.assertEqual(anns[1]["attributes"]["track_id"], anns[2]["attributes"]["track_id"])

            report = json.loads((root / "build/on_report.json").read_text())
            self.assertEqual(report["stats"]["tracking_low_matched"], 1)
            self.assertEqual(report["stats"]["tracking_propagated"], 1)
            self.assertTrue(report["params"]["temporal_tracking"]["enabled"])

    def test_tracking_carries_previous_state_when_low_detection_is_unknown(self):
        with tempfile.TemporaryDirectory(prefix="tlr_match_state_carry_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root, low_state="unknown", low_lamps=[])

            run_match(
                root,
                "--output", "annotation/on.json",
                "--report", "build/on_report.json",
                "--no-fill-gaps",
                "--no-map-fill",
                "--min-score", "0.5",
                "--temporal-tracking",
                "--tracking-config", "configs/tracking/bytetrack-lite.yaml",
                "--tracking-max-lost-frames", "2",
            )

            anns = json.loads((root / "annotation/on.json").read_text())["annotations"]
            tracked = [a for a in anns if a["attributes"]["source_type"] == "tracked"][0]
            self.assertEqual(tracked["attributes"]["state"], "red-circle")
            self.assertEqual(tracked["attributes"]["raw_state"], "unknown")
            self.assertEqual(tracked["attributes"]["signal_kind"], "vehicle")


if __name__ == "__main__":
    unittest.main()
