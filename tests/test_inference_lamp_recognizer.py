"""Unit tests for tlr_autolabel/inference/lamp_recognizer.py (REFACTOR_PLAN.md
phase 6 extraction from tlr_autolabel.py).
"""
import unittest

from tlr_autolabel.inference.lamp_recognizer import lamp_label, normalize_lamps, signal_state


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


if __name__ == "__main__":
    unittest.main()
