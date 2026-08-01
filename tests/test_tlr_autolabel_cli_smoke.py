"""CLI smoke test for the Tier A tlr_autolabel/v1 per-frame JSON contract.

Runs tlr_autolabel.py's real main() end to end (arg parsing, image loop,
payload assembly, file writing) against a synthetic image with the
Detector/LampClassifier backends monkeypatched to fakes, so no real
model file or GPU is needed. See REFACTOR_PLAN.md section 2.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_tlr_autolabel_script():
    # tlr_autolabel.py (this CLI script) and the tlr_autolabel/ package (see
    # REFACTOR_PLAN.md) share a name; the package always wins a plain
    # `import tlr_autolabel`, so load the script by explicit path instead.
    spec = importlib.util.spec_from_file_location(
        "_tlr_autolabel_script_smoke", ROOT / "scripts" / "tlr_autolabel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDetector:
    def __init__(self, _model_path, _comlops_param):
        self.w = 64
        self.h = 64
        self.kind = "yolox"
        self.sess = None

    def detect(self, _img, score_thr):
        boxes = [{"prob": 0.9, "x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 30.0}]
        return [b for b in boxes if b["prob"] >= score_thr], 1.0


class FakeClassifier:
    def __init__(self, _model_path, _param_path, _args):
        self.width = 32
        self.height = 32
        self.backend = "fake"

    def classify(self, _img, _bbox):
        return [{"label": "red-circle", "color": "red", "shape": "circle",
                  "arrow": None, "confidence": 0.95}]


class TlrAutolabelCliSmokeTest(unittest.TestCase):
    def _run(self, tmp: Path, extra_args: list[str]) -> Path:
        image_dir = tmp / "images"
        image_dir.mkdir()
        out_dir = tmp / "out"
        img_path = image_dir / "00000.png"
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        module = _load_tlr_autolabel_script()
        import cv2
        cv2.imwrite(str(img_path), img)

        dummy_detector = tmp / "detector.onnx"
        dummy_classifier = tmp / "classifier.onnx"
        dummy_classifier_param = tmp / "classifier_param.yaml"
        for p in (dummy_detector, dummy_classifier, dummy_classifier_param):
            p.write_text("")

        module.Detector = FakeDetector
        module.LampClassifier = FakeClassifier

        argv = [
            "scripts/tlr_autolabel.py", str(image_dir),
            "--detector", str(dummy_detector),
            "--classifier", str(dummy_classifier),
            "--classifier-param", str(dummy_classifier_param),
            "--out-dir", str(out_dir),
            "--run-id", "synthetic-test-run",
        ] + extra_args
        old_argv = sys.argv
        sys.argv = argv
        try:
            module.main()
        finally:
            sys.argv = old_argv
        return out_dir / "00000.json"

    def test_required_metadata_and_signals_shape(self):
        with tempfile.TemporaryDirectory(prefix="tlr_autolabel_smoke_") as tmp:
            payload_path = self._run(Path(tmp), [])
            self.assertTrue(payload_path.exists())
            payload = json.loads(payload_path.read_text())

            self.assertEqual(payload["schema_version"], "tlr_autolabel/v1")
            for key in ("image", "sample_data_token", "channel", "frame_index",
                        "width", "height", "meta", "signals"):
                self.assertIn(key, payload)
            self.assertEqual(payload["meta"]["run_id"], "synthetic-test-run")
            self.assertIn("created_at", payload["meta"])

            self.assertEqual(len(payload["signals"]), 1)
            signal = payload["signals"][0]
            self.assertEqual(signal["signal_id"], "00000-00")
            for key in ("detector_score", "box_xyxy", "lamps", "state"):
                self.assertIn(key, signal)
            self.assertNotIn("raw_detections", payload)

    def test_raw_detections_present_only_with_low_score_thr(self):
        with tempfile.TemporaryDirectory(prefix="tlr_autolabel_smoke_") as tmp:
            payload_path = self._run(Path(tmp), ["--det-low-score-thr", "0.1"])
            payload = json.loads(payload_path.read_text())

            self.assertIn("raw_detections", payload)
            self.assertEqual(len(payload["raw_detections"]), 1)
            raw = payload["raw_detections"][0]
            self.assertEqual(raw["raw_detection_id"], "00000-raw-00")
            for key in ("detector_score", "box_xyxy", "lamps", "state", "detection_level"):
                self.assertIn(key, raw)


if __name__ == "__main__":
    unittest.main()
