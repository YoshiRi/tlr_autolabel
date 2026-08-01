"""Unit tests for tlr_autolabel/map/association.py.

The matching helpers are package-level reusable functions; the CLI entrypoint
under scripts/ imports them.
"""
import unittest

from tlr_autolabel.map.association import center, iou, match_boxes_legacy, unmatched_reason


class IouCenterTest(unittest.TestCase):
    def test_iou_identical_boxes(self):
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_iou_disjoint_boxes(self):
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_center_midpoint(self):
        c = center([0, 0, 10, 20])
        self.assertEqual(list(c), [5.0, 10.0])


class MatchBoxesLegacyTest(unittest.TestCase):
    def test_overlapping_pair_matches(self):
        detections = [{"box_xyxy": [10, 10, 30, 30]}]
        candidates = [{"bbox": [11, 11, 31, 31]}]
        matches, sources = match_boxes_legacy(detections, candidates)
        self.assertEqual(matches, {0: 0})
        self.assertEqual(sources, {0: "legacy"})

    def test_far_apart_pair_stays_unmatched(self):
        detections = [{"box_xyxy": [10, 10, 30, 30]}]
        candidates = [{"bbox": [1000, 1000, 1030, 1030]}]
        matches, _ = match_boxes_legacy(detections, candidates)
        self.assertEqual(matches, {})

    def test_empty_inputs(self):
        self.assertEqual(match_boxes_legacy([], []), ({}, {}))


class PackageExportTest(unittest.TestCase):
    def test_package_exposes_matching_helpers(self):
        from tlr_autolabel.map import association

        self.assertIs(iou, association.iou)
        self.assertIs(center, association.center)
        self.assertIs(match_boxes_legacy, association.match_boxes_legacy)
        self.assertIs(unmatched_reason, association.unmatched_reason)


if __name__ == "__main__":
    unittest.main()
