"""Registry + wiring tests for the detector model plug-in
(docs/model_interface.md, phase A).

Phase A is a behavior-preserving seam: the yolox/comlops model objects must
delegate to the exact same preprocess/decode functions the Detector wrapper used
before, so their output is bit-identical by construction. These tests pin that
wiring and the registry resolution without needing a real ONNX model or GPU.
"""
import unittest
from pathlib import Path

import numpy as np

from tlr_autolabel.inference import models
from tlr_autolabel.inference.detector import comlops_decode, det_decode, det_preprocess
from tlr_autolabel.inference.models.comlops import ComlopsDetectorModel
from tlr_autolabel.inference.models.yolox import YoloxDetectorModel

REPO_ROOT = Path(__file__).resolve().parents[1]
COMLOPS_PARAM = REPO_ROOT / "configs" / "model_params" / "comlops_large_detector_ml.param.yaml"


class RegistryTest(unittest.TestCase):
    def test_builtins_registered(self):
        self.assertEqual(models.list_detectors(), ["comlops", "yolox"])

    def test_infer_detector_type_matches_legacy_mapping(self):
        self.assertEqual(models.infer_detector_type(1), "yolox")
        self.assertEqual(models.infer_detector_type(3), "comlops")

    def test_infer_detector_type_unknown_fails_loud(self):
        with self.assertRaises(SystemExit):
            models.infer_detector_type(2)

    def test_build_unknown_type_fails_loud(self):
        with self.assertRaises(SystemExit):
            models.build_detector_model("does-not-exist", "x.onnx", {})

    def test_model_metadata(self):
        self.assertEqual((YoloxDetectorModel.num_outputs, YoloxDetectorModel.supports_engine),
                         (1, True))
        self.assertEqual((ComlopsDetectorModel.num_outputs, ComlopsDetectorModel.supports_engine),
                         (3, False))


class YoloxWiringTest(unittest.TestCase):
    def test_preprocess_identical_to_legacy(self):
        img = np.random.randint(0, 255, (50, 40, 3), dtype=np.uint8)
        blob, scale = YoloxDetectorModel().preprocess(img, 64, 64)
        exp_blob, exp_scale = det_preprocess(img, 64, 64)  # BGR, norm=1.0
        np.testing.assert_array_equal(blob, exp_blob)
        self.assertEqual(scale, exp_scale)

    def test_decode_identical_to_legacy(self):
        # 64x64 net -> strides 8/16/32 -> 64+16+4 = 84 grid rows, 4box+1obj+1cls
        out = np.zeros((84, 6), dtype=np.float32)
        out[0, 4] = out[0, 5] = 1.0  # one confident detection
        got = YoloxDetectorModel().decode([out], 64, 64, 0.3)
        exp = det_decode(out, 64, 64, 0.3)
        self.assertEqual(got, exp)


class ComlopsWiringTest(unittest.TestCase):
    def setUp(self):
        if not COMLOPS_PARAM.exists():
            self.skipTest(f"missing {COMLOPS_PARAM}")
        self.model = ComlopsDetectorModel(params={"comlops_param_path": str(COMLOPS_PARAM)})

    def test_preprocess_uses_rgb_and_norm(self):
        img = np.random.randint(0, 255, (50, 40, 3), dtype=np.uint8)
        blob, _ = self.model.preprocess(img, 64, 64)
        exp, _ = det_preprocess(img, 64, 64, rgb=True, norm=1.0 / 255.0)
        np.testing.assert_array_equal(blob, exp)

    def test_set_keep_classes_maps_labels_to_ids(self):
        first = self.model.labels[0]
        self.model.set_keep_classes([first])
        self.assertEqual(self.model.keep_ids, {self.model.labels.index(first)})

    def test_decode_delegates_to_comlops_decode(self):
        self.model.set_keep_classes([self.model.labels[0]])
        mp = self.model.mp
        outs = [np.zeros((mp["num_anchors"] * mp["chans_per_anchor"], g, g), dtype=np.float32)
                for g in (8, 4, 2)]
        got = self.model.decode(outs, 64, 64, 0.3)
        exp = comlops_decode(outs, mp, 0.3, self.model.keep_ids)
        self.assertEqual(got, exp)


if __name__ == "__main__":
    unittest.main()
