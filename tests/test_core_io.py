"""Unit tests for the shared JSON/table helpers (REFACTOR_PLAN.md phase 4/7).

token_of and load_json used to be redefined identically in several scripts
(to_object_ann.py, export_object_ann_to_t4dataset.py, apply_re_review.py,
export_cvat_signal_task.py, import_cvat_signal_annotations.py); all now
import the single copy in tlr_autolabel/core/io.py. Pin behavior so future
changes to any one caller can't silently drift the shared helper.
"""
import json
import tempfile
import unittest
from pathlib import Path

from tlr_autolabel.core.io import load_json, token_of


class TokenOfTest(unittest.TestCase):
    def test_deterministic_for_same_parts(self):
        self.assertEqual(token_of("a", "b", 1), token_of("a", "b", 1))

    def test_distinguishes_part_boundaries(self):
        self.assertNotEqual(token_of("a", "b1"), token_of("a", "b", 1))

    def test_returns_md5_hex_string(self):
        token = token_of("traffic_light", "instance-1", "501")
        self.assertEqual(len(token), 32)
        int(token, 16)  # raises ValueError if not hex


class LoadJsonTest(unittest.TestCase):
    def test_round_trips_a_list(self):
        with tempfile.TemporaryDirectory(prefix="tlr_core_io_") as tmp:
            path = Path(tmp) / "rows.json"
            path.write_text(json.dumps([{"token": "a"}, {"token": "b"}]))
            self.assertEqual(load_json(path), [{"token": "a"}, {"token": "b"}])

    def test_round_trips_a_dict(self):
        with tempfile.TemporaryDirectory(prefix="tlr_core_io_") as tmp:
            path = Path(tmp) / "payload.json"
            path.write_text(json.dumps({"schema_version": "v1"}))
            self.assertEqual(load_json(path), {"schema_version": "v1"})


if __name__ == "__main__":
    unittest.main()
