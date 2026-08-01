"""Unit tests for tlr_autolabel/t4/object_ann.py and traffic_light.py
(REFACTOR_PLAN.md phase 4 extraction from to_object_ann.py).
"""
import unittest

from tlr_autolabel.t4.object_ann import assign_instances, box_iou
from tlr_autolabel.t4.traffic_light import build_traffic_light_row


class BoxIouTest(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)


class AssignInstancesTest(unittest.TestCase):
    def test_overlapping_boxes_across_frames_merge_into_one_instance(self):
        records = [
            {"sd": "sd-0", "box": [10.0, 10.0, 30.0, 30.0], "uid": "a", "cat": "red", "map_id": None},
            {"sd": "sd-1", "box": [11.0, 11.0, 31.0, 31.0], "uid": "b", "cat": "red", "map_id": None},
        ]
        sd_meta = {"sd-0": ("CAM_FRONT", 0), "sd-1": ("CAM_FRONT", 1)}
        inst_of, instances = assign_instances(records, sd_meta)
        self.assertEqual(len(instances), 1)
        self.assertEqual(inst_of[0], inst_of[1])
        self.assertEqual(instances[0]["nbr_annotations"], 0)  # finalized later by the caller

    def test_disjoint_boxes_produce_separate_instances(self):
        records = [
            {"sd": "sd-0", "box": [10.0, 10.0, 30.0, 30.0], "uid": "a", "cat": "red", "map_id": None},
            {"sd": "sd-0", "box": [100.0, 100.0, 130.0, 130.0], "uid": "b", "cat": "red", "map_id": None},
        ]
        sd_meta = {"sd-0": ("CAM_FRONT", 0)}
        inst_of, instances = assign_instances(records, sd_meta)
        self.assertEqual(len(instances), 2)
        self.assertNotEqual(inst_of[0], inst_of[1])

    def test_incompatible_map_ids_prevent_merge(self):
        records = [
            {"sd": "sd-0", "box": [10.0, 10.0, 30.0, 30.0], "uid": "a", "cat": "red", "map_id": "101"},
            {"sd": "sd-1", "box": [11.0, 11.0, 31.0, 31.0], "uid": "b", "cat": "red", "map_id": "202"},
        ]
        sd_meta = {"sd-0": ("CAM_FRONT", 0), "sd-1": ("CAM_FRONT", 1)}
        inst_of, instances = assign_instances(records, sd_meta)
        self.assertEqual(len(instances), 2)
        self.assertNotEqual(inst_of[0], inst_of[1])

    def test_different_channels_never_merge(self):
        records = [
            {"sd": "sd-0", "box": [10.0, 10.0, 30.0, 30.0], "uid": "a", "cat": "red", "map_id": None},
            {"sd": "sd-1", "box": [10.0, 10.0, 30.0, 30.0], "uid": "b", "cat": "red", "map_id": None},
        ]
        sd_meta = {"sd-0": ("CAM_FRONT", 0), "sd-1": ("CAM_BACK", 0)}
        inst_of, instances = assign_instances(records, sd_meta)
        self.assertEqual(len(instances), 2)
        self.assertNotEqual(inst_of[0], inst_of[1])


class BuildTrafficLightRowTest(unittest.TestCase):
    def test_row_shape(self):
        row = build_traffic_light_row("inst-1", "501")
        self.assertEqual(set(row.keys()), {"token", "instance_token", "primitive_id"})
        self.assertEqual(row["instance_token"], "inst-1")
        self.assertEqual(row["primitive_id"], "501")

    def test_token_deterministic_and_pair_sensitive(self):
        self.assertEqual(build_traffic_light_row("inst-1", "501")["token"],
                          build_traffic_light_row("inst-1", "501")["token"])
        self.assertNotEqual(build_traffic_light_row("inst-1", "501")["token"],
                             build_traffic_light_row("inst-1", "502")["token"])


if __name__ == "__main__":
    unittest.main()
