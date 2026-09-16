"""Lamp-slot geometry inside a projected housing (map/lamp_geometry.py).

Pins the slot layout the lamp-level matching mode depends on: which axis the
lamps run along (taken from the projected box's own aspect, so vertical
snow-region housings need no extra map tag), the JP colour order along it, and
that anything unplaceable returns None so the caller falls back to the full
projection rather than guessing.
"""
import unittest

from tlr_autolabel.map.lamp_geometry import (
    expected_lamp_box,
    housing_axis,
    lamp_slots,
)


def candidate(bbox, subtype):
    return {"bbox": bbox, "subtype": subtype, "way_id": "1"}


def element(color, shape="circle", arrow=None):
    return {"color": color, "shape": shape, "arrow": arrow}


class AxisTest(unittest.TestCase):
    def test_wide_box_is_horizontal(self):
        self.assertEqual(housing_axis([0, 0, 120, 45]), "horizontal")

    def test_tall_box_is_vertical(self):
        self.assertEqual(housing_axis([0, 0, 35, 100]), "vertical")


class SlotOrderTest(unittest.TestCase):
    def test_horizontal_vehicle_is_green_amber_red_left_to_right(self):
        self.assertEqual(lamp_slots("red_yellow_green", [0, 0, 120, 45]),
                         ("green", "amber", "red"))

    def test_vertical_vehicle_is_red_amber_green_top_to_bottom(self):
        self.assertEqual(lamp_slots("red_yellow_green", [0, 0, 40, 120]),
                         ("red", "amber", "green"))

    def test_pedestrian_is_red_over_green(self):
        self.assertEqual(lamp_slots("red_green", [0, 0, 35, 80]), ("red", "green"))

    def test_unknown_subtype_has_no_slots(self):
        self.assertEqual(lamp_slots("", [0, 0, 120, 45]), ())
        self.assertEqual(lamp_slots("pedestrian_crossing", [0, 0, 120, 45]), ())


class ExpectedLampBoxTest(unittest.TestCase):
    def test_horizontal_vehicle_slots_split_the_width(self):
        cand = candidate([0, 0, 300, 100], "red_yellow_green")
        self.assertEqual(expected_lamp_box(cand, [element("green")]), [0, 0, 100, 100])
        self.assertEqual(expected_lamp_box(cand, [element("amber")]), [100, 0, 200, 100])
        self.assertEqual(expected_lamp_box(cand, [element("red")]), [200, 0, 300, 100])

    def test_vertical_pedestrian_slots_split_the_height(self):
        cand = candidate([0, 0, 40, 100], "red_green")
        self.assertEqual(expected_lamp_box(cand, [element("red", "ped")]), [0, 0, 40, 50])
        self.assertEqual(expected_lamp_box(cand, [element("green", "ped")]), [0, 50, 40, 100])

    def test_slot_centres_match_the_measured_lamp_offsets(self):
        # the offsets this model was validated against: green at -1/3 and red at
        # +1/3 of the housing width, red-ped at -1/4 and green-ped at +1/4 of
        # its height (measured against the same run's housing-level L1)
        cand = candidate([0, 0, 300, 100], "red_yellow_green")
        for color, expected in (("green", -1 / 3), ("amber", 0.0), ("red", 1 / 3)):
            box = expected_lamp_box(cand, [element(color)])
            offset = ((box[0] + box[2]) / 2 - 150) / 300
            self.assertAlmostEqual(offset, expected, places=6, msg=color)

        ped = candidate([0, 0, 40, 100], "red_green")
        for color, expected in (("red", -1 / 4), ("green", 1 / 4)):
            box = expected_lamp_box(ped, [element(color, "ped")])
            offset = ((box[1] + box[3]) / 2 - 50) / 100
            self.assertAlmostEqual(offset, expected, places=6, msg=color)

    def test_multiple_lamps_span_their_slots(self):
        cand = candidate([0, 0, 300, 100], "red_yellow_green")
        box = expected_lamp_box(cand, [element("green"), element("red")])
        self.assertEqual(box, [0, 0, 300, 100])

    def test_arrow_is_not_placed_because_its_panel_is_outside_the_housing(self):
        cand = candidate([0, 0, 300, 100], "red_yellow_green")
        self.assertIsNone(expected_lamp_box(cand, [element("green", "arrow", "left")]))

    def test_colour_the_housing_does_not_have_is_not_placed(self):
        ped = candidate([0, 0, 40, 100], "red_green")
        self.assertIsNone(expected_lamp_box(ped, [element("amber", "ped")]))

    def test_unknown_subtype_falls_back_to_the_full_projection(self):
        self.assertIsNone(expected_lamp_box(candidate([0, 0, 300, 100], ""),
                                            [element("red")]))

    def test_no_elements_is_not_placed(self):
        self.assertIsNone(expected_lamp_box(candidate([0, 0, 300, 100],
                                                      "red_yellow_green"), []))


if __name__ == "__main__":
    unittest.main()
