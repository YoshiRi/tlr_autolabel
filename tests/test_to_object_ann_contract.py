"""I/O contract test for Tier B/B' `traffic_light.json`, object_ann.json,
category.json, attribute.json, and instance.json.

Pins the t4devkit-defined relation-row shape produced by to_object_ann.py:
`{token, instance_token, primitive_id}`, and the golden shape of the other
generated Tier B tables. See REFACTOR_PLAN.md and
docs/existing_annotation_if.md for the agreed contract.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_minimal_t4_dataset(root: Path) -> None:
    ann = root / "annotation"
    ann.mkdir(parents=True)
    write_json(ann / "sensor.json", [
        {"token": "sensor-cam", "channel": "CAM_FRONT", "modality": "camera"}
    ])
    write_json(ann / "calibrated_sensor.json", [
        {"token": "calib-cam", "sensor_token": "sensor-cam"}
    ])
    write_json(ann / "sample_data.json", [
        {"token": "sd-0", "calibrated_sensor_token": "calib-cam",
         "width": 200, "height": 200, "timestamp": 0}
    ])
    write_json(ann / "scene.json", [])
    write_json(ann / "category.json", [])
    write_json(ann / "attribute.json", [])
    write_json(ann / "instance.json", [])
    write_json(ann / "object_ann.json", [])
    write_json(ann / "sample_annotation.json", [])


def write_sidecar(path: Path, *, map_traffic_light_id: str) -> None:
    write_json(path, {
        "schema_version": "traffic_signal_2d/v2",
        "annotations": [{
            "token": "ann-0",
            "sample_data_token": "sd-0",
            "box2d": [10.0, 10.0, 30.0, 30.0],
            "attributes": {
                "state": "red-circle",
                "visibility": "full",
                "map_traffic_light_id": map_traffic_light_id,
            },
        }],
    })


def run_to_object_ann(dataset_root: Path, out_dir: Path, sidecar: Path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "to_object_ann.py"),
         "--t4-dataset", str(dataset_root), "--out", str(out_dir),
         "--sidecar", str(sidecar)],
        cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


class TrafficLightJsonContractTest(unittest.TestCase):
    def test_relation_rows_use_token_instance_token_primitive_id(self):
        with tempfile.TemporaryDirectory(prefix="tlr_to_object_ann_") as tmp:
            root = Path(tmp)
            src = root / "src"
            out = root / "out"
            write_minimal_t4_dataset(src)
            sidecar = root / "sidecar.json"
            write_sidecar(sidecar, map_traffic_light_id="501")

            run_to_object_ann(src, out, sidecar)

            traffic_light_path = out / "annotation" / "traffic_light.json"
            self.assertTrue(traffic_light_path.exists(),
                             "traffic_light.json must be written when a map "
                             "relation exists (B' contract)")
            rows = json.loads(traffic_light_path.read_text())
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(set(row.keys()), {"token", "instance_token", "primitive_id"})
            self.assertIsInstance(row["token"], str)
            self.assertIsInstance(row["instance_token"], str)
            self.assertEqual(row["primitive_id"], "501")

    def test_no_relation_when_map_id_absent(self):
        with tempfile.TemporaryDirectory(prefix="tlr_to_object_ann_") as tmp:
            root = Path(tmp)
            src = root / "src"
            out = root / "out"
            write_minimal_t4_dataset(src)
            sidecar = root / "sidecar.json"
            write_sidecar(sidecar, map_traffic_light_id="")

            run_to_object_ann(src, out, sidecar)

            traffic_light_path = out / "annotation" / "traffic_light.json"
            self.assertFalse(traffic_light_path.exists(),
                              "Tier B (no map relation) must not write "
                              "traffic_light.json")

    def test_object_ann_category_attribute_instance_golden_shape(self):
        with tempfile.TemporaryDirectory(prefix="tlr_to_object_ann_") as tmp:
            root = Path(tmp)
            src = root / "src"
            out = root / "out"
            write_minimal_t4_dataset(src)
            sidecar = root / "sidecar.json"
            write_sidecar(sidecar, map_traffic_light_id="501")

            run_to_object_ann(src, out, sidecar)
            ann_out = out / "annotation"

            object_ann = json.loads((ann_out / "object_ann.json").read_text())
            self.assertEqual(len(object_ann), 1)
            oa = object_ann[0]
            self.assertEqual(set(oa.keys()), {
                "token", "sample_data_token", "instance_token", "category_token",
                "attribute_tokens", "bbox", "mask",
            })
            self.assertEqual(oa["sample_data_token"], "sd-0")
            self.assertEqual(oa["bbox"], [10.0, 10.0, 30.0, 30.0])
            self.assertIsNone(oa["mask"])
            self.assertEqual(len(oa["attribute_tokens"]), 3)

            categories = json.loads((ann_out / "category.json").read_text())
            self.assertEqual([c["name"] for c in categories], ["red"])
            cat_token = categories[0]["token"]
            self.assertEqual(oa["category_token"], cat_token)

            attributes = {a["token"]: a["name"] for a in
                          json.loads((ann_out / "attribute.json").read_text())}
            attr_names = {attributes[t] for t in oa["attribute_tokens"]}
            self.assertEqual(attr_names, {
                "occlusion_state.none",
                "Truncation_State.non-truncated",
                "light_status.on",
            })

            instances = json.loads((ann_out / "instance.json").read_text())
            self.assertEqual(len(instances), 1)
            inst = instances[0]
            self.assertEqual(inst["token"], oa["instance_token"])
            self.assertEqual(inst["category_token"], cat_token)
            self.assertEqual(inst["nbr_annotations"], 1)
            self.assertEqual(inst["first_annotation_token"], oa["token"])
            self.assertEqual(inst["last_annotation_token"], oa["token"])

    def test_deprecated_map_association_not_written_by_default(self):
        with tempfile.TemporaryDirectory(prefix="tlr_to_object_ann_") as tmp:
            root = Path(tmp)
            src = root / "src"
            out = root / "out"
            write_minimal_t4_dataset(src)
            sidecar = root / "sidecar.json"
            write_sidecar(sidecar, map_traffic_light_id="501")

            run_to_object_ann(src, out, sidecar)

            legacy_path = out / "annotation" / "traffic_light_map_association.json"
            stale_path = out / "annotation" / "traffic_light_instance_map.json"
            self.assertFalse(legacy_path.exists())
            self.assertFalse(stale_path.exists())


if __name__ == "__main__":
    unittest.main()
