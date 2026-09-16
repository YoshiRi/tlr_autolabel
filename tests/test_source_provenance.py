"""L1 observation identity must survive into the A' sidecar.

CoMET's `--only-tlr --tlr-tracking` produces a map-free 2D track and the
interop adapter passes it down as source_instance_token/_name on each
tlr_autolabel/v1 signal. Before this contract, match_traffic_lights.py
dropped unknown L1 fields, so a reviewer could not tell which CoMET track a
box came from. These tests pin that the fields are copied, that they stay
distinct from the map-assisted L3 `track_id`, and that synthetic boxes
(propagated / interpolated / map_presence) leave them empty rather than
inheriting someone else's identity.
"""
import json
import tempfile
import unittest
from pathlib import Path

from test_match_temporal_integration import run_match, write_synthetic_dataset

PROVENANCE_KEYS = ("source_track_id", "source_track_name", "source_detection_id")


def tag_l1_signals(root: Path, **fields):
    """Add upstream provenance fields to every L1 detection in the fixture."""
    for path in sorted((root / "tlr_autolabel").rglob("*.json")):
        payload = json.loads(path.read_text())
        for group in ("signals", "raw_detections"):
            for signal in payload.get(group, []):
                signal.update(fields)
        path.write_text(json.dumps(payload, indent=2))


class SourceProvenanceTest(unittest.TestCase):
    def test_l1_identity_is_copied_onto_detection_rows(self):
        with tempfile.TemporaryDirectory(prefix="tlr_provenance_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root)
            tag_l1_signals(
                root,
                source_instance_token="comet-instance-7",
                source_instance_name="traffic_light_7",
                source_object_ann_token="comet-oann-42",
            )

            run_match(
                root,
                "--output", "annotation/a_prime.json",
                "--report", "build/report.json",
                "--no-fill-gaps",
                "--no-map-fill",
                "--min-score", "0.5",
            )

            anns = json.loads((root / "annotation/a_prime.json").read_text())["annotations"]
            self.assertTrue(anns)
            for ann in anns:
                attrs = ann["attributes"]
                self.assertEqual(attrs["source_track_id"], "comet-instance-7")
                self.assertEqual(attrs["source_track_name"], "traffic_light_7")
                self.assertEqual(attrs["source_detection_id"], "comet-oann-42")

    def test_signal_id_is_the_detection_id_fallback(self):
        with tempfile.TemporaryDirectory(prefix="tlr_provenance_fallback_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root)

            run_match(
                root,
                "--output", "annotation/a_prime.json",
                "--report", "build/report.json",
                "--no-fill-gaps",
                "--no-map-fill",
                "--min-score", "0.5",
            )

            anns = json.loads((root / "annotation/a_prime.json").read_text())["annotations"]
            attrs = anns[0]["attributes"]
            self.assertEqual(attrs["source_detection_id"], "00000-00")
            self.assertEqual(attrs["source_track_id"], "")
            self.assertEqual(attrs["source_track_name"], "")

    def test_source_track_id_is_independent_of_l3_track_id(self):
        with tempfile.TemporaryDirectory(prefix="tlr_provenance_tracking_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root)
            tag_l1_signals(root, source_instance_token="comet-instance-7")

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
            by_source = {a["attributes"]["source_type"]: a["attributes"] for a in anns}
            self.assertEqual({"auto", "tracked", "propagated"}, set(by_source))

            # real observations (high and low-score recovery) keep L1 identity
            for source_type in ("auto", "tracked"):
                attrs = by_source[source_type]
                self.assertEqual(attrs["source_track_id"], "comet-instance-7")
                self.assertTrue(attrs["track_id"])
                self.assertNotEqual(attrs["track_id"], attrs["source_track_id"])

            # a propagated box is not an observation: no upstream identity
            propagated = by_source["propagated"]
            self.assertTrue(propagated["track_id"])
            for key in PROVENANCE_KEYS:
                self.assertEqual(propagated[key], "")

    def test_backfilled_boxes_carry_no_upstream_identity(self):
        with tempfile.TemporaryDirectory(prefix="tlr_provenance_backfill_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root)
            tag_l1_signals(root, source_instance_token="comet-instance-7")

            run_match(
                root,
                "--output", "annotation/fill.json",
                "--report", "build/fill_report.json",
                "--min-score", "0.5",
                "--fill-mode", "all",
            )

            anns = json.loads((root / "annotation/fill.json").read_text())["annotations"]
            backfilled = [a["attributes"] for a in anns
                          if a["attributes"]["source_type"] in ("interpolated", "map_presence")]
            self.assertTrue(backfilled)
            for attrs in backfilled:
                for key in PROVENANCE_KEYS:
                    self.assertIn(key, attrs)
                    self.assertEqual(attrs[key], "")


if __name__ == "__main__":
    unittest.main()
