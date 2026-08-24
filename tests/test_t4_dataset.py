"""Unit tests for the shared T4 dataset reader (tlr_autolabel/t4/dataset.py).

Built to replace ad hoc per-script annotation/*.json loading without
depending on t4devkit.
"""
import json
import tempfile
import unittest
from pathlib import Path

from tlr_autolabel.t4.dataset import T4Dataset


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def write_minimal_dataset(root: Path) -> None:
    ann = root / "annotation"
    write_json(ann / "sensor.json", [
        {"token": "sensor-cam", "channel": "CAM_FRONT", "modality": "camera"},
        {"token": "sensor-lidar", "channel": "LIDAR_TOP", "modality": "lidar"},
    ])
    write_json(ann / "calibrated_sensor.json", [
        {"token": "calib-cam", "sensor_token": "sensor-cam"},
        {"token": "calib-lidar", "sensor_token": "sensor-lidar"},
    ])
    write_json(ann / "ego_pose.json", [
        {"token": "ego-0", "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]},
    ])
    write_json(ann / "sample.json", [
        {"token": "sample-0", "timestamp": 1000},
    ])
    write_json(ann / "sample_data.json", [
        {"token": "sd-cam-0", "sample_token": "sample-0", "ego_pose_token": "ego-0",
         "calibrated_sensor_token": "calib-cam", "filename": "CAM_FRONT/00000.jpg", "timestamp": 1000},
        {"token": "sd-lidar-0", "sample_token": "sample-0", "ego_pose_token": "ego-0",
         "calibrated_sensor_token": "calib-lidar", "filename": "LIDAR_TOP/00000.pcd.bin", "timestamp": 1000},
    ])


class T4DatasetLoadTest(unittest.TestCase):
    def test_missing_required_table_raises(self):
        with tempfile.TemporaryDirectory(prefix="tlr_t4_dataset_") as tmp:
            root = Path(tmp)
            (root / "annotation").mkdir()
            with self.assertRaises(FileNotFoundError):
                T4Dataset.load(root)

    def test_missing_optional_table_loads_empty(self):
        with tempfile.TemporaryDirectory(prefix="tlr_t4_dataset_") as tmp:
            root = Path(tmp)
            write_minimal_dataset(root)
            ds = T4Dataset.load(root)
            self.assertEqual(ds.by_token("category"), {})
            self.assertEqual(ds.rows("object_ann"), [])

    def test_camera_frames_only_include_camera_modality(self):
        with tempfile.TemporaryDirectory(prefix="tlr_t4_dataset_") as tmp:
            root = Path(tmp)
            write_minimal_dataset(root)
            ds = T4Dataset.load(root)
            self.assertIn("sd-cam-0", ds.camera_frames_by_token)
            self.assertNotIn("sd-lidar-0", ds.camera_frames_by_token)
            frame = ds.camera_frames_by_token["sd-cam-0"]
            self.assertEqual(frame.channel, "CAM_FRONT")
            self.assertEqual(frame.stem, "00000")
            self.assertEqual(ds.channel_of("sd-cam-0"), "CAM_FRONT")

    def test_channel_stem_index(self):
        with tempfile.TemporaryDirectory(prefix="tlr_t4_dataset_") as tmp:
            root = Path(tmp)
            write_minimal_dataset(root)
            ds = T4Dataset.load(root)
            frame = ds.camera_frames_by_channel_stem[("CAM_FRONT", "00000")]
            self.assertEqual(frame.token, "sd-cam-0")

    def test_timestamp_of_sample(self):
        with tempfile.TemporaryDirectory(prefix="tlr_t4_dataset_") as tmp:
            root = Path(tmp)
            write_minimal_dataset(root)
            ds = T4Dataset.load(root)
            self.assertEqual(ds.timestamp_of_sample("sample-0"), 1000)
            self.assertIsNone(ds.timestamp_of_sample("missing"))

    def test_scene_name_defaults(self):
        with tempfile.TemporaryDirectory(prefix="tlr_t4_dataset_") as tmp:
            root = Path(tmp)
            write_minimal_dataset(root)
            ds = T4Dataset.load(root)
            self.assertEqual(ds.scene_name(), "scene")


if __name__ == "__main__":
    unittest.main()
