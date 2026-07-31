import unittest
from types import SimpleNamespace

import numpy as np

from tlr_autolabel import process_image_with_candidates


class FakeDetector:
    h = 100
    w = 100

    def __init__(self):
        self.thresholds = []
        self.boxes = [
            {"prob": 0.8, "x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0},
            {"prob": 0.3, "x1": 50.0, "y1": 50.0, "x2": 70.0, "y2": 70.0},
        ]

    def detect(self, _img, score_thr):
        self.thresholds.append(score_thr)
        return [b for b in self.boxes if b["prob"] > score_thr], 1.0


class FakeClassifier:
    def __init__(self, lamps=None):
        self.calls = []
        self.lamps = lamps or []

    def classify(self, img, bbox):
        self.calls.append((img.shape, bbox))
        return [dict(lamp) for lamp in self.lamps]


def args(**overrides):
    base = {
        "det_score_thr": 0.5,
        "det_low_score_thr": None,
        "det_nms_thr": 0.35,
        "tiles": False,
        "tile_overlap": 128,
        "min_box": 8.0,
        "crop_pad": 0.0,
        "drop_unknown": False,
        "classify_low_detections": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TlrAutolabelCandidateTest(unittest.TestCase):
    def test_low_threshold_keeps_low_candidates_out_of_signals(self):
        detector = FakeDetector()
        classifier = FakeClassifier()
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        signals, raw = process_image_with_candidates(
            img, detector, classifier, args(det_low_score_thr=0.2)
        )

        self.assertEqual(detector.thresholds, [0.2])
        self.assertEqual(len(classifier.calls), 1)
        self.assertEqual([s["detector_score"] for s in signals], [0.8])
        self.assertEqual([d["detector_score"] for d in raw], [0.8, 0.3])
        self.assertEqual([d["detection_level"] for d in raw], ["high", "low"])

    def test_low_candidates_can_be_classified_explicitly(self):
        detector = FakeDetector()
        classifier = FakeClassifier()
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        process_image_with_candidates(
            img, detector, classifier,
            args(det_low_score_thr=0.2, classify_low_detections=True),
        )

        self.assertEqual(len(classifier.calls), 2)

    def test_without_low_threshold_detector_runs_at_high_threshold(self):
        detector = FakeDetector()
        classifier = FakeClassifier()
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        signals, raw = process_image_with_candidates(img, detector, classifier, args())

        self.assertEqual(detector.thresholds, [0.5])
        self.assertEqual([s["detector_score"] for s in signals], [0.8])
        self.assertEqual([d["detector_score"] for d in raw], [0.8])

    def test_classifier_geometry_is_not_exposed_in_tier_a(self):
        detector = FakeDetector()
        classifier = FakeClassifier(lamps=[{
            "label": "red-circle",
            "color": "red",
            "shape": "circle",
            "arrow": None,
            "confidence": 0.9,
            "box_xyxy": [14, 15, 18, 19],
            "box_rel_xyxy": [0.2, 0.2, 0.4, 0.4],
            "box_source": "classifier_yolox",
        }])
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        signals, raw = process_image_with_candidates(img, detector, classifier, args())

        self.assertEqual(signals[0]["box_xyxy"], [10, 10, 30, 30])
        self.assertEqual(raw[0]["box_xyxy"], [10, 10, 30, 30])
        self.assertEqual(
            set(signals[0]["lamps"][0]),
            {"label", "color", "shape", "arrow", "confidence"},
        )
        self.assertEqual(signals[0]["lamps"], raw[0]["lamps"])


if __name__ == "__main__":
    unittest.main()
