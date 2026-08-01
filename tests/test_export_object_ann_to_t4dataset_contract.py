"""Golden test for export_object_ann_to_t4dataset.py (REFACTOR_PLAN.md phase 4).

This script had no prior test coverage; this pins its B/B' output shape
(instance splitting on ambiguous map association, traffic_light.json rows)
and doubles as a regression check for the tlr_autolabel/t4/adapters.py
extraction that now backs it.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_source_dataset(root: Path) -> None:
    ann = root / "annotation"
    write_json(ann / "sample_data.json", [
        {"token": "sd-0", "timestamp": 0},
        {"token": "sd-1", "timestamp": 1},
    ])
    write_json(ann / "scene.json", [{"name": "test-scene"}])
    write_json(ann / "category.json", [{"token": "cat-red", "name": "red"}])
    write_json(ann / "attribute.json", [])
    write_json(ann / "sample_annotation.json", [])
    write_json(ann / "instance.json", [{
        "token": "inst-shared", "category_token": "cat-red",
        "instance_name": "orig", "nbr_annotations": 0,
        "first_annotation_token": "", "last_annotation_token": "",
    }])
    # Both object_ann rows share one instance_token, but the deprecated
    # association maps them to two different map ids -> must split on export.
    write_json(ann / "object_ann.json", [
        {"token": "oa-0", "sample_data_token": "sd-0", "instance_token": "inst-shared",
         "category_token": "cat-red", "attribute_tokens": [], "bbox": [0, 0, 10, 10], "mask": None},
        {"token": "oa-1", "sample_data_token": "sd-1", "instance_token": "inst-shared",
         "category_token": "cat-red", "attribute_tokens": [], "bbox": [0, 0, 10, 10], "mask": None},
    ])
    write_json(ann / "traffic_light_map_association.json", [
        {"object_ann_token": "oa-0", "map_traffic_light_id": "101"},
        {"object_ann_token": "oa-1", "map_traffic_light_id": "202"},
    ])


def run_export(src: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_object_ann_to_t4dataset.py"),
         "--dataset-root", str(src), "--out", str(out)],
        cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


class ExportObjectAnnContractTest(unittest.TestCase):
    def test_ambiguous_instance_splits_and_traffic_light_rows_shaped(self):
        with tempfile.TemporaryDirectory(prefix="tlr_export_oa_") as tmp:
            root = Path(tmp)
            src = root / "src"
            out = root / "out"
            write_source_dataset(src)

            run_export(src, out)

            object_ann = json.loads((out / "annotation/object_ann.json").read_text())
            self.assertEqual(len(object_ann), 2)
            instance_tokens = {row["instance_token"] for row in object_ann}
            self.assertEqual(len(instance_tokens), 2,
                              "one instance mapped to two map ids must split into two instances")

            traffic_light = json.loads((out / "annotation/traffic_light.json").read_text())
            self.assertEqual(len(traffic_light), 2)
            for row in traffic_light:
                self.assertEqual(set(row.keys()), {"token", "instance_token", "primitive_id"})
            self.assertEqual({row["primitive_id"] for row in traffic_light}, {"101", "202"})

            instances = json.loads((out / "annotation/instance.json").read_text())
            self.assertEqual({i["token"] for i in instances}, instance_tokens)
            for inst in instances:
                self.assertEqual(inst["nbr_annotations"], 1)

            legacy_path = out / "annotation/traffic_light_map_association.json"
            self.assertFalse(legacy_path.exists())

    def test_out_must_differ_from_dataset_root(self):
        with tempfile.TemporaryDirectory(prefix="tlr_export_oa_same_") as tmp:
            root = Path(tmp)
            write_source_dataset(root)
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "export_object_ann_to_t4dataset.py"),
                 "--dataset-root", str(root), "--out", str(root)],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
