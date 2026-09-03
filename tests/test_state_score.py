"""Localization confidence vs state confidence.

A two-stage producer reports two independent numbers: the detector says "a
signal is here" and the lamp classifier says "and its state is this". Folding
them into one `detector_score` hid a whole class of rows -- on a 2400-frame
run, 11368 of 18772 housings had no readable lamp but carried a median
detector_score of 0.950, so they sailed through a 0.5 gate as confident
detections with state `unknown`. These pin the three fixes: the state
confidence reaches A', `--min-state-score` withdraws an unsupported state
without discarding the box, and the match rate is reported separately for
detections whose state was actually read.
"""
import json
import tempfile
import unittest
from pathlib import Path

from tlr_autolabel.core.l1_ingest import state_score
from test_match_lamp_level import (
    match as run_match,
    projected_housing,
    write_dataset,
    write_json,
)

IMAGE_WH = (2880, 1860)


def write_l1(root: Path, signals: list[dict]):
    write_json(root / "tlr_autolabel" / "00000.json", {
        "schema_version": "tlr_autolabel/v1",
        "image": "CAM_TRAFFIC_LIGHT_FAR/00000.jpg",
        "sample_data_token": "sd-0",
        "channel": "CAM_TRAFFIC_LIGHT_FAR",
        "frame_index": 0,
        "width": IMAGE_WH[0], "height": IMAGE_WH[1],
        "meta": {"run_id": "state-score"},
        "signals": signals,
    })


def housing_signal(root: Path, *, state, lamp_confidence, detector_score=0.95,
                   signal_id="00000-00"):
    """One housing-level detection on the projected signal, so it matches."""
    box = [round(v, 2) for v in projected_housing(root)["bbox"]]
    lamps = []
    if state != "unknown":
        color, shape = state.split("-", 1)
        lamps = [{"label": state, "color": color, "shape": shape, "arrow": None,
                  "confidence": lamp_confidence}]
    return {
        "signal_id": signal_id,
        "detector_score": detector_score,
        "box_xyxy": box,
        "state": state,
        "lamps": lamps,
    }


def attrs_of(root: Path, *args) -> dict:
    rows = run_match(root, "on", *args) if args else run_match(root, "on")
    return rows[0]["attributes"]


class StateScoreHelperTest(unittest.TestCase):
    def test_strongest_lamp_wins(self):
        self.assertEqual(state_score({"lamps": [{"confidence": 0.4},
                                                {"confidence": 0.91}]}), 0.91)

    def test_explicit_field_overrides_the_lamps(self):
        self.assertEqual(state_score({"state_score": 0.2,
                                      "lamps": [{"confidence": 0.99}]}), 0.2)

    def test_no_readable_lamp_is_none_not_zero(self):
        # None means "unread", which is actionable; 0.0 would read as "certain
        # it has no state" and would sort alongside a genuine low confidence
        self.assertIsNone(state_score({"lamps": []}))
        self.assertIsNone(state_score({}))
        self.assertIsNone(state_score({"lamps": [{"confidence": None}]}))

    def test_detector_score_is_not_used_as_a_fallback(self):
        self.assertIsNone(state_score({"detector_score": 0.99, "lamps": []}))


class APrimeStateScoreTest(unittest.TestCase):
    def test_state_score_reaches_the_sidecar(self):
        with tempfile.TemporaryDirectory(prefix="tlr_state_score_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, [housing_signal(root, state="red-circle",
                                           lamp_confidence=0.87,
                                           detector_score=0.95)])
            a = attrs_of(root)
            self.assertEqual(a["state"], "red-circle")
            self.assertEqual(a["detector_score"], "0.95")
            self.assertEqual(a["state_score"], "0.87")

    def test_unread_housing_keeps_its_detector_score_and_has_no_state_score(self):
        with tempfile.TemporaryDirectory(prefix="tlr_state_unread_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, [housing_signal(root, state="unknown",
                                           lamp_confidence=None,
                                           detector_score=0.95)])
            a = attrs_of(root)
            self.assertEqual(a["state"], "unknown")
            self.assertEqual(a["detector_score"], "0.95")
            self.assertEqual(a["state_score"], "")


class MinStateScoreTest(unittest.TestCase):
    def test_disabled_by_default(self):
        with tempfile.TemporaryDirectory(prefix="tlr_mss_off_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, [housing_signal(root, state="red-circle",
                                           lamp_confidence=0.30)])
            a = attrs_of(root)
            self.assertEqual(a["state"], "red-circle")

    def test_weak_state_is_withdrawn_but_the_box_survives(self):
        with tempfile.TemporaryDirectory(prefix="tlr_mss_on_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, [housing_signal(root, state="red-circle",
                                           lamp_confidence=0.30)])
            a = attrs_of(root, "--min-state-score", "0.5")
            self.assertEqual(a["state"], "unknown")
            # provenance is kept: the box, the score and what the detector said
            self.assertEqual(a["raw_state"], "red-circle")
            self.assertEqual(a["state_score"], "0.3")
            self.assertEqual(a["map_traffic_light_id"], "101")

    def test_strong_state_is_untouched(self):
        with tempfile.TemporaryDirectory(prefix="tlr_mss_keep_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, [housing_signal(root, state="red-circle",
                                           lamp_confidence=0.80)])
            a = attrs_of(root, "--min-state-score", "0.5")
            self.assertEqual(a["state"], "red-circle")

    def test_threshold_is_reported_in_the_run_params(self):
        with tempfile.TemporaryDirectory(prefix="tlr_mss_param_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, [housing_signal(root, state="red-circle",
                                           lamp_confidence=0.30)])
            run_match(root, "on", "--min-state-score", "0.5")
            report = json.loads((root / "build/on_report.json").read_text())
            self.assertEqual(report["params"]["min_state_score"], 0.5)
            self.assertEqual(report["stats"]["state_suppressed"], 1)


class SplitMetricsTest(unittest.TestCase):
    def test_stated_and_unread_detections_are_counted_separately(self):
        with tempfile.TemporaryDirectory(prefix="tlr_split_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            box = [round(v, 2) for v in projected_housing(root)["bbox"]]
            # one read housing on the projection, one unread housing elsewhere
            elsewhere = [box[0] - 600, box[1], box[2] - 600, box[3]]
            write_l1(root, [
                housing_signal(root, state="red-circle", lamp_confidence=0.9,
                               signal_id="read"),
                {"signal_id": "unread", "detector_score": 0.95,
                 "box_xyxy": elsewhere, "state": "unknown", "lamps": []},
            ])
            run_match(root, "on")
            stats = json.loads((root / "build/on_report.json").read_text())["stats"]
            self.assertEqual(stats["detections"], 2)
            self.assertEqual(stats["detections_stated"], 1)
            self.assertEqual(stats["detections_state_unknown"], 1)
            # the read one is on the projection, so it is the one that matched
            self.assertEqual(stats["matched_stated"], 1)
            self.assertEqual(stats.get("matched_state_unknown", 0), 0)

    def test_all_read_leaves_the_unknown_bucket_empty(self):
        with tempfile.TemporaryDirectory(prefix="tlr_split_read_") as tmp:
            root = Path(tmp)
            write_dataset(root)
            write_l1(root, [housing_signal(root, state="red-circle",
                                           lamp_confidence=0.9)])
            run_match(root, "on")
            stats = json.loads((root / "build/on_report.json").read_text())["stats"]
            self.assertEqual(stats["detections_stated"], 1)
            self.assertEqual(stats.get("detections_state_unknown", 0), 0)


if __name__ == "__main__":
    unittest.main()
