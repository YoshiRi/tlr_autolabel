"""Unit tests for tlr_autolabel/inference/detector.py (REFACTOR_PLAN.md phase 6
extraction from the L1 CLI).
"""
import unittest

from tlr_autolabel.inference.detector import clipped_detection_box, det_nms, tile_origins


class DetNmsTest(unittest.TestCase):
    def test_higher_score_survives_iou_overlap(self):
        dets = [
            {"prob": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
            {"prob": 0.5, "x1": 1, "y1": 1, "x2": 11, "y2": 11},
        ]
        kept = det_nms(dets, iou_thr=0.3)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["prob"], 0.9)

    def test_disjoint_boxes_both_kept(self):
        dets = [
            {"prob": 0.9, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
            {"prob": 0.5, "x1": 100, "y1": 100, "x2": 110, "y2": 110},
        ]
        kept = det_nms(dets, iou_thr=0.3)
        self.assertEqual(len(kept), 2)


class TileOriginsTest(unittest.TestCase):
    def test_fits_without_tiling(self):
        self.assertEqual(tile_origins(500, 960), [0])

    def test_covers_larger_size_with_overlap(self):
        origins = tile_origins(2000, 960, min_overlap=128)
        self.assertEqual(origins[0], 0)
        self.assertEqual(origins[-1], 2000 - 960)

    def test_overlap_at_or_above_net_still_covers(self):
        # overlap >= net leaves no tile step; it must be clamped, not divide by zero
        for overlap in (64, 128, 1000):
            origins = tile_origins(200, 64, min_overlap=overlap)
            self.assertEqual(origins[0], 0)
            self.assertEqual(origins[-1], 200 - 64)
            self.assertGreater(len(origins), 1)
            steps = [b - a for a, b in zip(origins, origins[1:])]
            self.assertTrue(all(0 < s <= 64 for s in steps), steps)


class ClippedDetectionBoxTest(unittest.TestCase):
    def test_clips_to_image_bounds(self):
        box = clipped_detection_box({"x1": -5, "y1": -5, "x2": 20, "y2": 20}, 15, 15, min_box=1.0)
        self.assertEqual(box, (0, 0, 15, 15))

    def test_returns_none_when_below_min_box(self):
        box = clipped_detection_box({"x1": 0, "y1": 0, "x2": 2, "y2": 2}, 100, 100, min_box=8.0)
        self.assertIsNone(box)


if __name__ == "__main__":
    unittest.main()
