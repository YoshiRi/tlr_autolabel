"""Unit tests for tlr_autolabel/inference/lamp_recognizer.py (REFACTOR_PLAN.md
phase 6 extraction from tlr_autolabel.py).
"""
import importlib.util
import unittest
from pathlib import Path

from tlr_autolabel.inference.lamp_recognizer import lamp_label, normalize_lamps, signal_state

ROOT = Path(__file__).resolve().parents[1]


def _load_tlr_autolabel_script():
    spec = importlib.util.spec_from_file_location(
        "_tlr_autolabel_script_lamp_test", ROOT / "scripts" / "tlr_autolabel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LampLabelTest(unittest.TestCase):
    def test_circle_lamp_no_arrow_suffix(self):
        label = lamp_label({"color": 2, "shape": 0})  # red, circle
        self.assertEqual(label, "red-circle")

    def test_arrow_lamp_includes_direction(self):
        label = lamp_label({"color": 0, "shape": 1, "sin": 1.0, "cos": 0.0})  # green, arrow, up_right-ish
        self.assertTrue(label.startswith("green-arrow-"))


class SignalStateTest(unittest.TestCase):
    def test_empty_lamps_is_unknown(self):
        self.assertEqual(signal_state([]), "unknown")

    def test_multiple_lamps_sorted_and_joined(self):
        lamps = [{"label": "red-circle"}, {"label": "amber-circle"}]
        self.assertEqual(signal_state(lamps), "amber-circle,red-circle")


class NormalizeLampsTest(unittest.TestCase):
    def test_drops_internal_keys(self):
        lamps = [{"label": "red-circle", "color": "red", "shape": "circle",
                  "arrow": None, "confidence": 0.9, "sin": 0.1, "cos": 0.9}]
        normalized = normalize_lamps(lamps)
        self.assertEqual(set(normalized[0].keys()),
                         {"label", "color", "shape", "arrow", "confidence"})


class ReexportIdentityTest(unittest.TestCase):
    def test_tlr_autolabel_reexports_same_objects(self):
        module = _load_tlr_autolabel_script()
        from tlr_autolabel.inference import lamp_recognizer

        self.assertIs(module.LampClassifier, lamp_recognizer.LampClassifier)
        self.assertIs(module.normalize_lamps, lamp_recognizer.normalize_lamps)
        self.assertIs(module.signal_state, lamp_recognizer.signal_state)


if __name__ == "__main__":
    unittest.main()
