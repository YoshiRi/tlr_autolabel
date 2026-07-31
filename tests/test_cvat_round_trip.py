"""CVAT round-trip contract test (A' sidecar -> CVAT zip -> A' sidecar).

See REFACTOR_PLAN.md section 2 and docs/cvat_interop.md. Reuses the
synthetic dataset fixture from the temporal integration tests, produces a
real A' sidecar via match_traffic_lights.py, exports it to a CVAT task zip
(no images, so no real image files are needed), then imports the XML back
and checks the editable fields survive losslessly with no manual edits in
between.
"""
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from test_match_temporal_integration import ROOT, run_match, write_synthetic_dataset


def add_image_dims(root: Path, width: int = 200, height: int = 200) -> None:
    # export_cvat_signal_task.py's XML writer needs width/height per
    # sample_data row; the shared match_traffic_lights fixture omits them
    # since the matcher itself doesn't need them.
    sd_path = root / "annotation/sample_data.json"
    rows = json.loads(sd_path.read_text())
    for row in rows:
        row["width"] = width
        row["height"] = height
    sd_path.write_text(json.dumps(rows, indent=2))


def run_script(name: str, *args) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / name), *args],
        cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


class CvatRoundTripTest(unittest.TestCase):
    def test_editable_fields_survive_export_import_unmodified(self):
        with tempfile.TemporaryDirectory(prefix="tlr_cvat_round_trip_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root)
            add_image_dims(root)

            run_match(
                root,
                "--output", "annotation/a_prime.json",
                "--report", "build/report.json",
                "--no-fill-gaps",
                "--no-map-fill",
                "--min-score", "0.5",
            )
            original = json.loads((root / "annotation/a_prime.json").read_text())["annotations"]

            zip_path = root / "build/cvat_signal/CAM_FRONT.zip"
            run_script(
                "export_cvat_signal_task.py",
                "--dataset-root", str(root),
                "--camera", "CAM_FRONT",
                "--count", "0",
                "--signal-ann", "annotation/a_prime.json",
                "--output", str(zip_path),
                "--no-images",
            )
            self.assertTrue(zip_path.exists())

            xml_path = root / "annotations.xml"
            with zipfile.ZipFile(zip_path) as archive:
                xml_path.write_bytes(archive.read("annotations.xml"))

            imported_path = root / "annotation/a_prime.imported.json"
            run_script(
                "import_cvat_signal_annotations.py",
                str(xml_path),
                "--dataset-root", str(root),
                "--output", str(imported_path),
            )
            imported = json.loads(imported_path.read_text())["annotations"]

            self.assertEqual(len(original), len(imported))
            by_token = {a["token"]: a for a in imported}
            for orig in original:
                got = by_token[orig["token"]]
                for key in ("state", "signal_kind", "visibility", "review_status",
                            "map_traffic_light_id"):
                    self.assertEqual(got["attributes"][key], orig["attributes"][key],
                                     f"{key} mismatch for {orig['token']}")
                self.assertEqual(got["box2d"], orig["box2d"])

    def test_import_fails_loud_on_unparsable_state_token(self):
        with tempfile.TemporaryDirectory(prefix="tlr_cvat_bad_state_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root)
            (root / "annotation").mkdir(exist_ok=True)

            xml_path = root / "annotations.xml"
            xml_path.write_text(
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<annotations>\n'
                '<image id="0" name="images/CAM_FRONT_00000.jpg" width="200" height="200">\n'
                '<box label="traffic_light" xtl="10.0" ytl="10.0" xbr="30.0" ybr="30.0" '
                'occluded="0" z_order="0">\n'
                '<attribute name="state">not-a-real-token</attribute>\n'
                '<attribute name="signal_kind">vehicle</attribute>\n'
                '<attribute name="visibility">full</attribute>\n'
                '<attribute name="review_status">accepted</attribute>\n'
                '<attribute name="map_traffic_light_id"></attribute>\n'
                '<attribute name="annotation_uid"></attribute>\n'
                '</box>\n'
                '</image>\n'
                '</annotations>\n'
            )
            result = subprocess.run(
                [sys.executable, str(ROOT / "import_cvat_signal_annotations.py"),
                 str(xml_path), "--dataset-root", str(root)],
                cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not-a-real-token", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
