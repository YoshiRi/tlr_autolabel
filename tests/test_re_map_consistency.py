"""Unit tests for the map/image consistency check
(tlr_autolabel/review/re_map_consistency.py).
"""
import unittest

from tlr_autolabel.review.re_map_consistency import (
    associate,
    association_radius,
    box_center,
    box_longer_side,
    center_distance,
    find_warnings,
    is_readable,
    median,
    nearest_projection,
)


def projection(way="3281", box=(100, 100, 140, 140), facing="front", distance=30.0):
    return {
        "way_id": way,
        "bbox": list(box),
        "facing": facing,
        "distance_m": distance,
        "subtype": "",
    }


def detection(box=(100, 100, 140, 140), state="red-circle", score="0.9", kind="vehicle"):
    return {
        "box2d": list(box),
        "attributes": {
            "state": state,
            "detector_score": score,
            "signal_kind": kind,
        },
    }


def way_stats(front=100, oblique=0, paired_front=90, offset=10.0):
    return {
        "projected_readable_frames": front + oblique,
        "projected_front_frames": front,
        "projected_oblique_frames": oblique,
        "paired_frames": paired_front,
        "paired_front_frames": paired_front,
        "observation_rate": None,
        "front_observation_rate": round(paired_front / front, 3) if front else None,
        "median_offset_px": offset,
    }


def report(ways=None, unmapped=None):
    return {
        "frames": 100,
        "paired": 0,
        "map_only": 0,
        "image_only": 0,
        "per_channel": {},
        "ways": ways or {},
        "unmapped_by_channel": unmapped or {},
    }


DEFAULTS = dict(
    min_projected_frames=10,
    min_observation_rate=0.3,
    min_unmapped_boxes=20,
    offset_hint_px=80.0,
)


class GeometryHelpersTest(unittest.TestCase):
    def test_box_center(self):
        self.assertEqual(box_center([0, 0, 10, 20]), (5.0, 10.0))

    def test_box_longer_side_uses_the_larger_dimension(self):
        self.assertEqual(box_longer_side([0, 0, 10, 30]), 30)

    def test_center_distance_is_symmetric(self):
        a, b = [0, 0, 2, 2], [3, 4, 5, 6]
        self.assertAlmostEqual(center_distance(a, b), center_distance(b, a))
        self.assertAlmostEqual(center_distance(a, b), 5.0)

    def test_association_radius_respects_the_floor(self):
        self.assertEqual(association_radius([0, 0, 4, 4], min_px=60.0, scale=2.0), 60.0)

    def test_association_radius_scales_with_a_large_projection(self):
        self.assertEqual(association_radius([0, 0, 100, 100], min_px=60.0, scale=2.0), 200.0)


class MedianTest(unittest.TestCase):
    def test_odd_length(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_even_length_averages_the_middle(self):
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_empty_is_zero(self):
        self.assertEqual(median([]), 0.0)


class IsReadableTest(unittest.TestCase):
    def test_readable_state_above_score(self):
        self.assertTrue(is_readable(detection(), 0.3))

    def test_unknown_state_is_not_a_claim_about_the_map(self):
        self.assertFalse(is_readable(detection(state="unknown"), 0.3))

    def test_empty_state_is_not_readable(self):
        self.assertFalse(is_readable(detection(state=""), 0.3))

    def test_weak_detection_is_ignored(self):
        self.assertFalse(is_readable(detection(score="0.1"), 0.3))

    def test_missing_score_is_treated_as_zero(self):
        ann = detection()
        del ann["attributes"]["detector_score"]
        self.assertFalse(is_readable(ann, 0.3))


class AssociateTest(unittest.TestCase):
    def test_pairs_a_coincident_projection_and_detection(self):
        pairs, map_only, image_only = associate(
            [projection()], [detection()], 60.0, 2.0
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(map_only, [])
        self.assertEqual(image_only, [])

    def test_far_apart_yields_one_of_each(self):
        pairs, map_only, image_only = associate(
            [projection()], [detection(box=(2000, 2000, 2040, 2040))], 60.0, 2.0
        )
        self.assertEqual(pairs, [])
        self.assertEqual(len(map_only), 1)
        self.assertEqual(len(image_only), 1)

    def test_closest_pair_wins_when_two_compete(self):
        # Both detections are within radius of the projection; the nearer one
        # must take it, leaving the other as image_only.
        near = detection(box=(100, 100, 140, 140))
        far = detection(box=(150, 100, 190, 140))
        pairs, map_only, image_only = associate(
            [projection()], [far, near], 60.0, 2.0
        )
        self.assertEqual(len(pairs), 1)
        self.assertIs(pairs[0][1], near)
        self.assertEqual(image_only, [far])
        self.assertEqual(map_only, [])

    def test_one_detection_cannot_satisfy_two_projections(self):
        pairs, map_only, image_only = associate(
            [projection(way="a"), projection(way="b", box=(110, 100, 150, 140))],
            [detection()],
            60.0, 2.0,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(map_only), 1)
        self.assertEqual(image_only, [])

    def test_no_projections_makes_everything_image_only(self):
        pairs, map_only, image_only = associate([], [detection()], 60.0, 2.0)
        self.assertEqual((pairs, map_only), ([], []))
        self.assertEqual(len(image_only), 1)

    def test_no_detections_makes_everything_map_only(self):
        pairs, map_only, image_only = associate([projection()], [], 60.0, 2.0)
        self.assertEqual((pairs, image_only), ([], []))
        self.assertEqual(len(map_only), 1)


class NearestProjectionTest(unittest.TestCase):
    def test_returns_the_closest_regardless_of_radius(self):
        far = projection(way="far", box=(5000, 5000, 5040, 5040))
        near = projection(way="near", box=(300, 300, 340, 340))
        found = nearest_projection(detection(), [far, near])
        self.assertEqual(found[0]["way_id"], "near")

    def test_returns_none_without_projections(self):
        self.assertIsNone(nearest_projection(detection(), []))


class FindWarningsTest(unittest.TestCase):
    def test_healthy_way_raises_nothing(self):
        self.assertEqual(find_warnings(report({"3289": way_stats()}), **DEFAULTS), [])

    def test_never_observed_head_on_is_reported(self):
        found = find_warnings(
            report({"3281": way_stats(front=74, paired_front=0)}), **DEFAULTS
        )
        self.assertEqual([w["kind"] for w in found], ["signal_never_observed"])

    def test_a_way_only_ever_seen_obliquely_is_not_reported(self):
        # The projection's own docstring notes the matched rate collapses at a
        # steep incidence, so obliquely-seen ways were never a fair test.
        found = find_warnings(
            report({"3281": way_stats(front=0, oblique=74, paired_front=0)}),
            **DEFAULTS,
        )
        self.assertEqual(found, [])

    def test_rarely_observed_way_is_reported(self):
        found = find_warnings(
            report({"3591": way_stats(front=100, paired_front=10)}), **DEFAULTS
        )
        self.assertEqual([w["kind"] for w in found], ["low_observation_rate"])

    def test_way_below_the_observability_floor_is_ignored(self):
        found = find_warnings(
            report({"x": way_stats(front=5, paired_front=0)}), **DEFAULTS
        )
        self.assertEqual(found, [])

    def test_large_offset_is_reported_when_the_rate_is_fine(self):
        found = find_warnings(
            report({"3595": way_stats(offset=150.0)}), **DEFAULTS
        )
        self.assertEqual([w["kind"] for w in found], ["large_projection_offset"])

    def test_unmapped_detections_are_reported(self):
        found = find_warnings(
            report(unmapped={"CAM_A": {
                "boxes": 281, "states": {"red-ped": 200},
                "kinds": {"pedestrian": 278}, "nearest_way": {"3595": 100},
                "median_offset_px": 300.0,
            }}),
            **DEFAULTS,
        )
        self.assertEqual([w["kind"] for w in found], ["unmapped_signal"])
        self.assertEqual(found[0]["boxes"], 281)

    def test_a_handful_of_unmapped_detections_is_not_a_finding(self):
        found = find_warnings(
            report(unmapped={"CAM_A": {
                "boxes": 3, "states": {}, "kinds": {}, "nearest_way": {},
                "median_offset_px": None,
            }}),
            **DEFAULTS,
        )
        self.assertEqual(found, [])

    def test_missing_map_entries_are_ranked_above_per_way_findings(self):
        found = find_warnings(
            report(
                ways={"3281": way_stats(front=74, paired_front=0)},
                unmapped={"CAM_A": {
                    "boxes": 281, "states": {}, "kinds": {},
                    "nearest_way": {}, "median_offset_px": None,
                }},
            ),
            **DEFAULTS,
        )
        self.assertEqual(found[0]["kind"], "unmapped_signal")

    def test_each_way_raises_at_most_one_finding(self):
        found = find_warnings(
            report({"3281": way_stats(front=74, paired_front=0, offset=500.0)}),
            **DEFAULTS,
        )
        self.assertEqual(len(found), 1)


if __name__ == "__main__":
    unittest.main()
