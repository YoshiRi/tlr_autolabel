"""Unit tests for the top-down map view
(tlr_autolabel/review/re_map_view.py).
"""
import math
import unittest

import numpy as np

from tlr_autolabel.review.re_map_view import (
    build_steps,
    clip_polyline,
    light_positions,
    local_road_context,
    quaternion_yaw,
    relevant_lights,
    round_points,
    summarize,
    view_bounds,
)


def make_ann(
    token="t",
    channel="CAM_A",
    sd="sd1",
    timestamp=100,
    way="",
    cand="",
    state="red-circle",
    kind="vehicle",
    reason="",
    re_id="",
):
    return {
        "token": token,
        "channel": channel,
        "sample_data_token": sd,
        "timestamp": timestamp,
        "attributes": {
            "map_traffic_light_id": way,
            "regulatory_element_id": re_id,
            "map_candidate_id": cand,
            "state": state,
            "signal_kind": kind,
            "unmatched_reason": reason,
            "detector_score": "0.9",
        },
    }


def ego(x=0.0, y=0.0, yaw=0.0, t=100):
    return {"x": x, "y": y, "yaw": yaw, "t": t}


class QuaternionYawTest(unittest.TestCase):
    def test_identity_is_zero(self):
        self.assertAlmostEqual(quaternion_yaw([1, 0, 0, 0]), 0.0)

    def test_quarter_turn_about_z(self):
        half = math.sqrt(0.5)
        self.assertAlmostEqual(quaternion_yaw([half, 0, 0, half]), math.pi / 2, places=6)

    def test_half_turn_about_z(self):
        self.assertAlmostEqual(abs(quaternion_yaw([0, 0, 0, 1])), math.pi, places=6)

    def test_missing_or_short_rotation_is_zero(self):
        self.assertEqual(quaternion_yaw(None), 0.0)
        self.assertEqual(quaternion_yaw([1, 0]), 0.0)


class LightPositionsTest(unittest.TestCase):
    def test_centroid_and_facing_are_flattened(self):
        corners = np.array([
            [10.0, 20.0, 5.0], [12.0, 20.0, 5.0],
            [10.0, 20.0, 6.0], [12.0, 20.0, 6.0],
        ])
        lights = light_positions(
            {"3281": {"corners": corners, "subtype": "red_yellow_green",
                      "facing_axis": np.array([0.0, -1.0])}},
            {"3281": ["10149", "10150"]},
        )
        light = lights["3281"]
        self.assertEqual((light["x"], light["y"]), (11.0, 20.0))
        self.assertEqual(light["re"], ["10149", "10150"])
        self.assertEqual((light["fx"], light["fy"]), (0.0, -1.0))

    def test_missing_facing_axis_becomes_null(self):
        corners = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        light = light_positions(
            {"1": {"corners": corners, "subtype": "", "facing_axis": None}}, {}
        )["1"]
        self.assertIsNone(light["fx"])
        self.assertIsNone(light["fy"])

    def test_way_without_regulatory_elements_gets_empty_list(self):
        corners = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        light = light_positions(
            {"1": {"corners": corners, "subtype": "", "facing_axis": None}}, {}
        )["1"]
        self.assertEqual(light["re"], [])


class BuildStepsTest(unittest.TestCase):
    def test_groups_annotations_of_one_frame_and_attaches_ego(self):
        anns = [make_ann(token="a", way="3281"), make_ann(token="b", cand="3595")]
        steps = build_steps(anns, {"sd1": ego(x=5.0, y=6.0, yaw=1.0)}, None)
        self.assertEqual(len(steps), 1)
        self.assertEqual((steps[0]["x"], steps[0]["y"]), (5.0, 6.0))
        self.assertEqual(len(steps[0]["obs"]), 2)

    def test_annotation_without_ego_pose_is_dropped(self):
        steps = build_steps([make_ann(sd="missing")], {"sd1": ego()}, None)
        self.assertEqual(steps, [])

    def test_matched_flag_follows_way_or_regulatory_element(self):
        anns = [
            make_ann(token="a", sd="sd1", way="3281"),
            make_ann(token="b", sd="sd1", re_id="10149"),
            make_ann(token="c", sd="sd1", cand="3595"),
        ]
        steps = build_steps(anns, {"sd1": ego()}, None)
        self.assertEqual([o["matched"] for o in steps[0]["obs"]], [True, True, False])

    def test_matched_observations_sort_before_unmatched(self):
        anns = [
            make_ann(token="u", sd="sd1", cand="3595"),
            make_ann(token="m", sd="sd1", way="3281"),
        ]
        steps = build_steps(anns, {"sd1": ego()}, None)
        self.assertTrue(steps[0]["obs"][0]["matched"])
        self.assertFalse(steps[0]["obs"][1]["matched"])

    def test_channel_filter_applies(self):
        anns = [
            make_ann(channel="CAM_A", sd="sd1"),
            make_ann(channel="CAM_B", sd="sd2"),
        ]
        steps = build_steps(anns, {"sd1": ego(), "sd2": ego()}, {"CAM_A"})
        self.assertEqual([s["channel"] for s in steps], ["CAM_A"])

    def test_steps_are_ordered_by_channel_then_time_and_indexed(self):
        anns = [
            make_ann(channel="CAM_B", sd="sd3", timestamp=10),
            make_ann(channel="CAM_A", sd="sd2", timestamp=200),
            make_ann(channel="CAM_A", sd="sd1", timestamp=100),
        ]
        poses = {"sd1": ego(t=100), "sd2": ego(t=200), "sd3": ego(t=10)}
        steps = build_steps(anns, poses, None)
        self.assertEqual(
            [(s["channel"], s["t"]) for s in steps],
            [("CAM_A", 100), ("CAM_A", 200), ("CAM_B", 10)],
        )
        self.assertEqual([s["i"] for s in steps], [0, 1, 2])


class RelevantLightsTest(unittest.TestCase):
    def setUp(self):
        self.lights = {
            "near": {"way": "near", "x": 5.0, "y": 0.0},
            "far": {"way": "far", "x": 5000.0, "y": 0.0},
            "referenced_far": {"way": "referenced_far", "x": 9000.0, "y": 0.0},
        }
        self.steps = [{"x": 0.0, "y": 0.0, "obs": [
            {"way": "referenced_far", "cand": ""},
        ]}]

    def test_keeps_lights_within_the_context_radius(self):
        keep = relevant_lights(self.lights, self.steps, radius=50.0)
        self.assertIn("near", keep)
        self.assertFalse(keep["near"]["referenced"])

    def test_drops_distant_unreferenced_lights(self):
        keep = relevant_lights(self.lights, self.steps, radius=50.0)
        self.assertNotIn("far", keep)

    def test_keeps_referenced_lights_regardless_of_distance(self):
        keep = relevant_lights(self.lights, self.steps, radius=50.0)
        self.assertIn("referenced_far", keep)
        self.assertTrue(keep["referenced_far"]["referenced"])

    def test_candidate_ids_also_count_as_referenced(self):
        steps = [{"x": 0.0, "y": 0.0, "obs": [{"way": "", "cand": "far"}]}]
        keep = relevant_lights(self.lights, steps, radius=1.0)
        self.assertIn("far", keep)
        self.assertTrue(keep["far"]["referenced"])


class SummarizeTest(unittest.TestCase):
    def test_separates_matched_ways_from_rejected_candidates(self):
        anns = [
            make_ann(token="a", sd="sd1", way="3289", state="red-circle"),
            make_ann(token="b", sd="sd1", cand="3595", state="red-ped",
                     reason="beyond_gate"),
        ]
        steps = build_steps(anns, {"sd1": ego()}, None)
        stats = summarize(steps, {})
        self.assertEqual(stats["matched_ways"], {"3289": {"red-circle": 1}})
        self.assertEqual(stats["candidate_ways"], {"3595": {"red-ped": 1}})
        self.assertEqual(stats["unmatched_reasons"], {"beyond_gate": 1})
        self.assertEqual(stats["n_unmatched"], 1)

    def test_a_way_can_be_both_matched_and_rejected(self):
        anns = [
            make_ann(token="a", sd="sd1", way="3595", state="green-ped"),
            make_ann(token="b", sd="sd2", cand="3595", state="red-ped",
                     reason="beyond_gate"),
        ]
        steps = build_steps(anns, {"sd1": ego(t=1), "sd2": ego(t=2)}, None)
        stats = summarize(steps, {})
        self.assertIn("3595", stats["matched_ways"])
        self.assertIn("3595", stats["candidate_ways"])

    def test_counts_totals(self):
        stats = summarize([], {})
        self.assertEqual(stats["n_steps"], 0)
        self.assertEqual(stats["n_observations"], 0)
        self.assertEqual(stats["n_unmatched"], 0)


class ViewBoundsTest(unittest.TestCase):
    def test_covers_both_path_and_lights_with_margin(self):
        steps = [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 5.0}]
        lights = {"a": {"x": -4.0, "y": 20.0}}
        self.assertEqual(view_bounds(steps, lights, 2.0), (-6.0, -2.0, 12.0, 22.0))

    def test_margin_widens_a_degenerate_box(self):
        bounds = view_bounds([{"x": 3.0, "y": 3.0}], {}, 5.0)
        self.assertEqual(bounds, (-2.0, -2.0, 8.0, 8.0))


class ClipPolylineTest(unittest.TestCase):
    BOUNDS = (0.0, 0.0, 10.0, 10.0)

    def test_keeps_polyline_with_a_vertex_inside(self):
        self.assertTrue(clip_polyline([(-5, -5), (5, 5)], self.BOUNDS))

    def test_drops_polyline_entirely_outside(self):
        self.assertFalse(clip_polyline([(50, 50), (60, 60)], self.BOUNDS))

    def test_boundary_vertex_counts_as_inside(self):
        self.assertTrue(clip_polyline([(0, 0)], self.BOUNDS))

    def test_empty_polyline_is_dropped(self):
        self.assertFalse(clip_polyline([], self.BOUNDS))


class RoundPointsTest(unittest.TestCase):
    def test_rounds_and_converts_to_lists(self):
        self.assertEqual(round_points([(1.23456, 2.7)], 2), [[1.23, 2.7]])


class LocalRoadContextTest(unittest.TestCase):
    BOUNDS = (0.0, 0.0, 10.0, 10.0)

    def setUp(self):
        self.lanelets = [
            {"id": "near", "subtype": "road",
             "left": [(1.0, 1.0), (2.0, 2.0)], "right": [(1.0, 0.5), (2.0, 1.5)]},
            {"id": "far", "subtype": "road",
             "left": [(500.0, 500.0)], "right": [(500.0, 501.0)]},
        ]
        self.ways = [
            {"id": "near_way", "type": "stop_line", "points": [(3.0, 3.0), (4.0, 4.0)]},
            {"id": "far_way", "type": "stop_line", "points": [(900.0, 900.0)]},
        ]

    def test_keeps_only_geometry_intersecting_the_bounds(self):
        lanelets, ways = local_road_context(self.lanelets, self.ways, self.BOUNDS)
        self.assertEqual([l["id"] for l in lanelets], ["near"])
        self.assertEqual([w["id"] for w in ways], ["near_way"])

    def test_kept_geometry_is_rounded_to_plain_lists(self):
        lanelets, _ = local_road_context(self.lanelets, self.ways, self.BOUNDS)
        self.assertEqual(lanelets[0]["left"], [[1.0, 1.0], [2.0, 2.0]])

    def test_lanelet_kept_when_only_one_bound_is_inside(self):
        lanelets, _ = local_road_context(
            [{"id": "half", "subtype": "road",
              "left": [(5.0, 5.0)], "right": [(900.0, 900.0)]}],
            [], self.BOUNDS,
        )
        self.assertEqual([l["id"] for l in lanelets], ["half"])

    def test_subtype_is_preserved(self):
        lanelets, _ = local_road_context(self.lanelets, self.ways, self.BOUNDS)
        self.assertEqual(lanelets[0]["subtype"], "road")


if __name__ == "__main__":
    unittest.main()
