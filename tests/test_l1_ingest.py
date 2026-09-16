"""Contract tests for the L1 ingest layer (tlr_autolabel/core/l1_ingest.py).

The layer exists because `box_xyxy` used to be silently ambiguous: the native
producer puts the housing there, a CoMLOps-style lamp-level producer puts one
lit lamp there, and L3 compared them the same way. These tests pin that
box_level is resolved (with the housing default that every pre-box_level
sidecar implies), that lamp-level detections naming a parent housing fold into
one housing-level signal, and that ones with no parent stay lamp-level instead
of masquerading as housings.
"""
import unittest

from tlr_autolabel.core.l1_ingest import (
    BOX_LEVEL_HOUSING,
    BOX_LEVEL_LAMP,
    box_level,
    count_by_level,
    signals,
)


def lamp_signal(signal_id, box, state, **extra):
    return {"signal_id": signal_id, "box_xyxy": box, "state": state,
            "detector_score": 0.9, **extra}


class BoxLevelResolutionTest(unittest.TestCase):
    def test_absent_box_level_means_housing(self):
        self.assertEqual(box_level({}, {}), BOX_LEVEL_HOUSING)
        self.assertEqual(box_level({}), BOX_LEVEL_HOUSING)

    def test_signal_overrides_payload(self):
        self.assertEqual(box_level({"box_level": "housing"}, {"box_level": "lamp"}),
                         BOX_LEVEL_HOUSING)

    def test_payload_applies_to_every_signal(self):
        payload = {"box_level": "lamp",
                   "signals": [lamp_signal("a", [0, 0, 10, 10], "red-circle")]}
        self.assertEqual([s["box_level"] for s in signals(payload)], [BOX_LEVEL_LAMP])

    def test_unknown_box_level_is_rejected(self):
        with self.assertRaises(ValueError):
            box_level({"box_level": "bulb"})


class HousingPassthroughTest(unittest.TestCase):
    def test_housing_signals_pass_through_with_level_made_explicit(self):
        payload = {"signals": [{
            "signal_id": "00000-00",
            "box_xyxy": [100, 200, 260, 260],
            "lamps": [{"label": "red-circle", "color": "red", "shape": "circle",
                       "arrow": None, "confidence": 0.99}],
            "state": "red-circle",
            "detector_score": 0.93,
        }]}
        out = signals(payload)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["box_level"], BOX_LEVEL_HOUSING)
        self.assertEqual(out[0]["box_xyxy"], [100, 200, 260, 260])
        self.assertEqual(out[0]["state"], "red-circle")
        # untouched otherwise
        self.assertEqual(out[0]["lamps"], payload["signals"][0]["lamps"])


class LampLevelWithoutParentTest(unittest.TestCase):
    def test_lamp_signals_stay_separate_and_labelled(self):
        payload = {"box_level": "lamp", "signals": [
            lamp_signal("a", [100, 200, 130, 230], "red-circle"),
            lamp_signal("b", [400, 210, 424, 250], "green-ped"),
        ]}
        out = signals(payload)
        self.assertEqual([s["box_level"] for s in out], [BOX_LEVEL_LAMP] * 2)
        self.assertEqual([s["box_xyxy"] for s in out],
                         [[100, 200, 130, 230], [400, 210, 424, 250]])
        self.assertEqual(count_by_level(out), {BOX_LEVEL_HOUSING: 0, BOX_LEVEL_LAMP: 2})

    def test_comlops_category_names_become_canonical_lamps(self):
        # a lamp-level producer may only name the detector class
        payload = {"box_level": "lamp", "signals": [
            {"signal_id": "a", "box_xyxy": [1, 1, 9, 9], "detector_score": 0.8,
             "source_category": "green_pedestrian"},
        ]}
        out = signals(payload)
        self.assertEqual(out[0]["state"], "green-ped")
        self.assertEqual(out[0]["lamps"][0]["color"], "green")
        self.assertEqual(out[0]["lamps"][0]["shape"], "ped")


class FoldingTest(unittest.TestCase):
    def test_lamps_sharing_a_housing_id_fold_into_one_signal(self):
        payload = {"box_level": "lamp", "signals": [
            lamp_signal("a", [100, 200, 130, 230], "red-circle",
                        housing_id="h1", housing_box_xyxy=[90, 195, 250, 250],
                        housing_score=0.95),
            lamp_signal("b", [200, 200, 230, 230], "green-arrow-left",
                        housing_id="h1", housing_box_xyxy=[90, 195, 250, 250],
                        housing_score=0.95),
        ]}
        out = signals(payload)
        self.assertEqual(len(out), 1)
        folded = out[0]
        self.assertEqual(folded["box_level"], BOX_LEVEL_HOUSING)
        self.assertEqual(folded["box_xyxy"], [90.0, 195.0, 250.0, 250.0])
        self.assertEqual(folded["state"], "green-arrow-left,red-circle")
        self.assertEqual(folded["detector_score"], 0.95)
        self.assertEqual(folded["folded_from"], ["a", "b"])
        # each lamp keeps its own box, so nothing is lost by folding
        self.assertEqual([l["box_xyxy"] for l in folded["lamps"]],
                         [[100.0, 200.0, 130.0, 230.0], [200.0, 200.0, 230.0, 230.0]])
        self.assertNotIn("housing_box_source", folded)

    def test_housing_box_is_derived_from_the_lamps_when_not_reported(self):
        payload = {"box_level": "lamp", "signals": [
            lamp_signal("a", [100, 200, 130, 230], "red-circle", housing_id="h1"),
            lamp_signal("b", [200, 210, 230, 240], "green-circle", housing_id="h1"),
        ]}
        folded = signals(payload)[0]
        self.assertEqual(folded["box_xyxy"], [100.0, 200.0, 230.0, 240.0])
        self.assertEqual(folded["housing_box_source"], "lamp_union")
        self.assertEqual(folded["detector_score"], 0.9)

    def test_distinct_housings_do_not_merge(self):
        payload = {"box_level": "lamp", "signals": [
            lamp_signal("a", [100, 200, 130, 230], "red-circle", housing_id="h1"),
            lamp_signal("b", [500, 200, 530, 230], "green-circle", housing_id="h2"),
        ]}
        out = signals(payload)
        self.assertEqual(len(out), 2)
        self.assertEqual({s["state"] for s in out}, {"red-circle", "green-circle"})

    def test_mixed_payload_keeps_housing_rows_and_folds_lamp_rows(self):
        payload = {"signals": [
            {"signal_id": "h", "box_xyxy": [0, 0, 100, 40], "state": "red-circle"},
            lamp_signal("a", [200, 200, 230, 230], "red-circle",
                        box_level="lamp", housing_id="h1"),
            lamp_signal("b", [260, 200, 290, 230], "green-circle",
                        box_level="lamp", housing_id="h1"),
        ]}
        out = signals(payload)
        self.assertEqual(count_by_level(out), {BOX_LEVEL_HOUSING: 2, BOX_LEVEL_LAMP: 0})
        self.assertEqual(out[0]["signal_id"], "h")
        self.assertEqual(out[1]["state"], "green-circle,red-circle")


if __name__ == "__main__":
    unittest.main()
