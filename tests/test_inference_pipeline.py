"""Pipeline payload contract (Tier A `tlr_autolabel/v1`).

The pipeline now assembles the payload that `cli/autolabel.py:main` used to
build inline, and it is what the comparison runner writes too — so the schema
promise is pinned here: the historical keys are always present, and the new keys
(`source`, `timestamp_us`, `timing_ms`, model digests) only appear when the
frame or the configuration actually carries them.
"""
import unittest

import numpy as np

from tlr_autolabel.frames import Frame
from tlr_autolabel.inference.config import InferenceConfig
from tlr_autolabel.inference.pipeline import Pipeline

HISTORICAL_KEYS = ("schema_version", "image", "image_realpath", "sample_data_token",
                   "channel", "frame_index", "width", "height", "meta", "signals")


class FakeDetector:
    kind = "yolox"
    w = h = 64
    sess = None

    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else [
            {"prob": 0.9, "x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0},
            {"prob": 0.2, "x1": 40.0, "y1": 40.0, "x2": 60.0, "y2": 60.0},
        ]

    def detect(self, _img, score_thr):
        return [b for b in self.boxes if b["prob"] >= score_thr], 1.0


class FakeClassifier:
    width = height = 32
    backend = "fake"
    kind = "lamp_recognizer"

    def classify(self, _img, _bbox):
        return [{"label": "red-circle", "color": "red", "shape": "circle",
                 "arrow": None, "confidence": 0.95}]


def make_frame(**kw):
    defaults = dict(frame_id="00000", frame_index=0,
                    image=np.zeros((100, 100, 3), dtype=np.uint8),
                    rel_path="CAM_FRONT/00000.png",
                    realpath="/abs/CAM_FRONT/00000.png",
                    channel="CAM_FRONT", sample_data_token="sd-0")
    defaults.update(kw)
    return Frame(**defaults)


def build(cfg=None, classifier=FakeClassifier(), detector=None):
    cfg = cfg or InferenceConfig(detector="d.onnx")
    return Pipeline(cfg=cfg, detector=detector or FakeDetector(),
                    classifier=classifier, run_id="test-run")


class PayloadShapeTest(unittest.TestCase):
    def test_historical_keys_and_order(self):
        payload = build().run(make_frame())
        self.assertEqual(list(payload)[:len(HISTORICAL_KEYS)], list(HISTORICAL_KEYS))
        self.assertEqual(payload["schema_version"], "tlr_autolabel/v1")
        self.assertEqual(payload["image"], "CAM_FRONT/00000.png")
        self.assertEqual(payload["channel"], "CAM_FRONT")
        self.assertEqual(payload["sample_data_token"], "sd-0")
        self.assertEqual((payload["width"], payload["height"]), (100, 100))
        self.assertEqual(payload["signals"][0]["signal_id"], "00000-00")
        self.assertEqual(payload["signals"][0]["state"], "red-circle")

    def test_optional_keys_absent_for_a_plain_image_frame(self):
        payload = build().run(make_frame())
        for key in ("source", "timestamp_us", "timing_ms", "raw_detections"):
            self.assertNotIn(key, payload)
        for key in ("detector_sha256", "classifier_sha256"):
            self.assertNotIn(key, payload["meta"])

    def test_source_and_timestamp_carried_from_the_frame(self):
        frame = make_frame(realpath=None, timestamp_us=123456,
                           source={"kind": "video", "uri": "a.mp4"})
        payload = build().run(frame)
        self.assertEqual(payload["source"], {"kind": "video", "uri": "a.mp4"})
        self.assertEqual(payload["timestamp_us"], 123456)
        self.assertIsNone(payload["image_realpath"])

    def test_frame_id_with_a_channel_prefix_keeps_signal_ids_short(self):
        payload = build().run(make_frame(frame_id="CAM_FRONT/00012"))
        self.assertEqual(payload["signals"][0]["signal_id"], "00012-00")

    def test_meta_records_families_and_thresholds(self):
        cfg = InferenceConfig(detector="d.onnx", det_score_thr=0.4, tiles=True,
                              preset="some-preset")
        meta = build(cfg).run(make_frame())["meta"]
        self.assertEqual(meta["run_id"], "test-run")
        self.assertEqual(meta["preset"], "some-preset")
        self.assertEqual(meta["det_score_thr"], 0.4)
        self.assertTrue(meta["tiles"])
        self.assertEqual(meta["detector_type"], "yolox")
        self.assertEqual(meta["classifier_type"], "lamp_recognizer")

    def test_timing_recorded_only_when_asked(self):
        cfg = InferenceConfig(detector="d.onnx", record_timing=True)
        payload = build(cfg).run(make_frame())
        timing = payload["timing_ms"]
        self.assertEqual(timing["crops"], 1)
        for key in ("detector", "classifier", "total"):
            self.assertGreaterEqual(timing[key], 0.0)
        self.assertGreaterEqual(timing["total"], timing["detector"])

    def test_raw_detections_only_with_a_low_threshold(self):
        cfg = InferenceConfig(detector="d.onnx", det_low_score_thr=0.1)
        payload = build(cfg).run(make_frame())
        self.assertEqual(len(payload["signals"]), 1, "low candidate must not become a signal")
        self.assertEqual(len(payload["raw_detections"]), 2)
        levels = [r["detection_level"] for r in payload["raw_detections"]]
        self.assertEqual(sorted(levels), ["high", "low"])
        low = next(r for r in payload["raw_detections"] if r["detection_level"] == "low")
        self.assertEqual(low["lamps"], [], "low candidates are not classified by default")
        self.assertEqual(low["raw_detection_id"], "00000-raw-01")

    def test_low_candidates_classified_on_request(self):
        cfg = InferenceConfig(detector="d.onnx", det_low_score_thr=0.1,
                              classify_low_detections=True)
        payload = build(cfg).run(make_frame())
        low = next(r for r in payload["raw_detections"] if r["detection_level"] == "low")
        self.assertEqual(low["state"], "red-circle")


class DetectorOnlyTest(unittest.TestCase):
    def test_no_classifier_gives_unknown_state_and_no_lamps(self):
        cfg = InferenceConfig(detector="d.onnx", classifier=None)
        pipeline = build(cfg, classifier=None)
        payload = pipeline.run(make_frame())
        self.assertEqual(payload["signals"][0]["state"], "unknown")
        self.assertEqual(payload["signals"][0]["lamps"], [])
        self.assertIsNone(payload["meta"]["classifier"])
        self.assertNotIn("classifier_type", payload["meta"])
        self.assertIn("detector-only", pipeline.describe())


if __name__ == "__main__":
    unittest.main()
