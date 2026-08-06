"""Configuration resolution contract: defaults <- preset <- explicit overrides.

Pins the rules the CLI used to implement by mutating an argparse Namespace, so
that in-process callers (the comparison runner) get exactly the same behavior:
explicit values always beat the preset, unknown preset keys fail loud, and model
paths expand ${TLR_MODEL_ROOT} / ${AUTOWARE_MLMODELS}.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tlr_autolabel.inference.config import (
    CONFIG_FIELDS, InferenceConfig, config_from_args, expand_path,
    resolve_config, validate_config,
)


class ResolveConfigTest(unittest.TestCase):
    def test_defaults_match_the_documented_l1_defaults(self):
        cfg = resolve_config()
        self.assertEqual(cfg.det_score_thr, 0.35)
        self.assertEqual(cfg.det_nms_thr, 0.35)
        self.assertEqual(cfg.cls_score_thr, 0.2)
        self.assertEqual(cfg.cls_nms_thr, 0.2)
        self.assertEqual(cfg.min_box, 8.0)
        self.assertEqual(cfg.crop_pad, 0.0)
        self.assertFalse(cfg.tiles)
        self.assertEqual(cfg.tile_overlap, 128)
        self.assertIsNone(cfg.det_low_score_thr)
        self.assertTrue(cfg.classifier_enabled)

    def test_shipped_presets_resolve_and_declare_a_detector_type(self):
        from tlr_autolabel.inference.config import list_presets
        from tlr_autolabel.inference.models import list_detectors

        presets = list_presets()
        self.assertTrue(presets, "no detector presets found")
        for name in presets:
            cfg = resolve_config(name)
            self.assertTrue(cfg.detector, f"{name}: no detector path")
            self.assertIn(cfg.detector_type, list_detectors(),
                          f"{name}: detector_type not registered")

    def test_override_beats_preset(self):
        cfg = resolve_config("yolox-1920-int8", {"det_score_thr": 0.5, "tiles": False})
        self.assertEqual(cfg.det_score_thr, 0.5)
        self.assertFalse(cfg.tiles, "explicit tiles=False must beat the preset's true")

    def test_preset_value_used_when_not_overridden(self):
        cfg = resolve_config("yolox-1920-int8")
        self.assertTrue(cfg.tiles, "preset tiles: true should apply")
        self.assertEqual(cfg.detector_type, "yolox")

    def test_unknown_preset_key_fails_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            preset_dir = Path(tmp)
            (preset_dir / "bogus.yaml").write_text("detecotr: typo.onnx\n")
            with mock.patch("tlr_autolabel.inference.config.PRESET_DIR", str(preset_dir)):
                with self.assertRaises(SystemExit) as ctx:
                    resolve_config("bogus")
        self.assertIn("unknown key", str(ctx.exception))

    def test_unknown_preset_name_lists_available(self):
        with self.assertRaises(SystemExit) as ctx:
            resolve_config("no-such-preset")
        self.assertIn("unknown preset", str(ctx.exception))

    def test_classifier_none_means_detector_only(self):
        for value in ("none", "None", "off", ""):
            cfg = resolve_config(overrides={"classifier": value})
            self.assertIsNone(cfg.classifier, f"{value!r} should disable the classifier")
            self.assertFalse(cfg.classifier_enabled)

    def test_model_root_expansion(self):
        with mock.patch.dict(os.environ, {"TLR_MODEL_ROOT": "/models/root"}):
            self.assertEqual(expand_path("${TLR_MODEL_ROOT}/a.onnx"), "/models/root/a.onnx")
            cfg = resolve_config(overrides={"detector": "$TLR_MODEL_ROOT/b.engine"})
            self.assertEqual(cfg.detector, "/models/root/b.engine")
        with mock.patch.dict(os.environ, {"AUTOWARE_MLMODELS": "/mlm"}):
            self.assertEqual(expand_path("${AUTOWARE_MLMODELS}/c.onnx"), "/mlm/c.onnx")

    def test_config_is_hashable_so_runs_can_be_keyed_by_it(self):
        self.assertEqual(resolve_config(), resolve_config())
        self.assertEqual(len({resolve_config(), resolve_config()}), 1)


class ConfigFromArgsTest(unittest.TestCase):
    def _parser_and_args(self, argv):
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--preset", default=None)
        ap.add_argument("--detector", default=None)
        ap.add_argument("--det-score-thr", type=float, default=0.35)
        ap.add_argument("--tiles", action="store_true")
        ap.add_argument("--crop-pad", type=float, default=0.0)
        return ap, ap.parse_args(argv)

    def test_untyped_flags_leave_the_preset_in_charge(self):
        ap, args = self._parser_and_args(["--preset", "yolox-1920-int8"])
        cfg = config_from_args(ap, args)
        self.assertTrue(cfg.tiles)
        self.assertEqual(cfg.det_score_thr, 0.35)

    def test_typed_flag_overrides_the_preset(self):
        ap, args = self._parser_and_args(
            ["--preset", "yolox-1920-int8", "--det-score-thr", "0.2"])
        cfg = config_from_args(ap, args)
        self.assertEqual(cfg.det_score_thr, 0.2)

    def test_extra_overrides_win(self):
        ap, args = self._parser_and_args(["--preset", "yolox-1920-int8"])
        cfg = config_from_args(ap, args, extra_overrides={"tiles": False})
        self.assertFalse(cfg.tiles)


class ValidateConfigTest(unittest.TestCase):
    def test_missing_detector(self):
        with self.assertRaises(SystemExit) as ctx:
            validate_config(InferenceConfig())
        self.assertIn("choose a detector", str(ctx.exception))

    def test_low_threshold_above_high_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            det = Path(tmp) / "d.onnx"
            det.write_text("")
            with self.assertRaises(SystemExit) as ctx:
                validate_config(InferenceConfig(
                    detector=str(det), det_score_thr=0.3, det_low_score_thr=0.4))
        self.assertIn("--det-low-score-thr must be <=", str(ctx.exception))

    def test_missing_model_file_mentions_model_root(self):
        with self.assertRaises(SystemExit) as ctx:
            validate_config(InferenceConfig(detector="/nope/model.onnx"))
        self.assertIn("detector model not found", str(ctx.exception))
        self.assertIn("model root", str(ctx.exception))

    def test_drop_unknown_without_classifier_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            det = Path(tmp) / "d.onnx"
            det.write_text("")
            with self.assertRaises(SystemExit) as ctx:
                validate_config(InferenceConfig(
                    detector=str(det), classifier=None, drop_unknown=True))
        self.assertIn("--drop-unknown", str(ctx.exception))

    def test_every_field_is_documented_in_config_fields(self):
        # the comparison matrix validates its `overrides:` against this tuple
        self.assertIn("det_score_thr", CONFIG_FIELDS)
        self.assertIn("classifier_type", CONFIG_FIELDS)
        self.assertIn("record_timing", CONFIG_FIELDS)


if __name__ == "__main__":
    unittest.main()
