"""I/O contract test for the A' `traffic_signal_2d/v2` sidecar.

Pins the row shape written by match_traffic_lights.py: schema_version,
per-annotation required keys, and the attributes sub-keys covering state,
visibility, review fields, and map cache fields. See REFACTOR_PLAN.md
section 2 and docs/cvat_interop.md for the agreed contract. Reuses the
synthetic dataset/lanelet2 fixture already built for the temporal
integration tests instead of duplicating it.
"""
import json
import tempfile
import unittest
from pathlib import Path

from test_match_temporal_integration import run_match, write_synthetic_dataset

REQUIRED_ANNOTATION_KEYS = {
    "token", "sample_token", "sample_data_token", "channel", "filename",
    "timestamp", "label", "box2d", "occluded", "z_order", "attributes",
}
REQUIRED_ATTRIBUTE_KEYS = {
    "state", "signal_kind", "visibility", "review_status",
    "map_traffic_light_id", "regulatory_element_id", "map_candidate_id",
    "regulatory_element_id_candidate", "unmatched_reason", "facing",
    "raw_state", "detector_score", "source_type",
}


class APrimeSidecarContractTest(unittest.TestCase):
    def test_annotation_rows_have_required_keys(self):
        with tempfile.TemporaryDirectory(prefix="tlr_a_prime_contract_") as tmp:
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

            sidecar = json.loads((root / "annotation/a_prime.json").read_text())
            self.assertEqual(sidecar["schema_version"], "traffic_signal_2d/v2")
            self.assertIn("annotations", sidecar)
            self.assertTrue(sidecar["annotations"])

            for ann in sidecar["annotations"]:
                self.assertTrue(REQUIRED_ANNOTATION_KEYS.issubset(ann.keys()),
                                 f"missing keys: {REQUIRED_ANNOTATION_KEYS - ann.keys()}")
                self.assertTrue(REQUIRED_ATTRIBUTE_KEYS.issubset(ann["attributes"].keys()),
                                 f"missing attribute keys: "
                                 f"{REQUIRED_ATTRIBUTE_KEYS - ann['attributes'].keys()}")
                self.assertIsInstance(ann["box2d"], list)
                self.assertEqual(len(ann["box2d"]), 4)


if __name__ == "__main__":
    unittest.main()
