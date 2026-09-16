"""Per-frame projection offset removal (tlr_autolabel/map/registration.py).

The offset that motivated this is a translation of a few tens of pixels that
varies frame to frame, so the tests pin the two properties the matcher relies
on: the estimate is a robust median over anchors (a single wrong anchor must
not move it), and only detections whose state was read are eligible anchors --
an unreadable one is usually a housing's back face, whose nearest map way
belongs to a different signal entirely.

Stdlib only:
    python3 -m pytest tests/test_projection_registration.py
"""
from __future__ import annotations

import unittest

from tlr_autolabel.map.registration import (
    SHIFT_KEY,
    UNSHIFTED_KEY,
    apply_shift,
    estimate_frame_shift,
)


def cand(way_id, box, **kw):
    return {"way_id": way_id, "bbox": list(box), "subtype": "red_yellow_green",
            "facing": "front", "distance_m": 60.0, **kw}


def moved(box, dx, dy):
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


class EstimateTest(unittest.TestCase):
    def test_recovers_a_pure_translation(self):
        boxes = [[1000, 500, 1054, 522], [1600, 540, 1660, 564]]
        cands = [cand("a", moved(boxes[0], 30, -12)),
                 cand("b", moved(boxes[1], 30, -12))]
        dx, dy, n = estimate_frame_shift(boxes, cands)
        self.assertAlmostEqual(dx, -30.0, places=6)
        self.assertAlmostEqual(dy, 12.0, places=6)
        self.assertEqual(n, 2)

    def test_zero_shift_when_already_aligned(self):
        boxes = [[100, 100, 140, 118]]
        dx, dy, _ = estimate_frame_shift(boxes, [cand("a", boxes[0])])
        self.assertAlmostEqual(dx, 0.0)
        self.assertAlmostEqual(dy, 0.0)

    def test_one_wrong_anchor_does_not_move_the_median(self):
        # three anchors agree on (-40, +15); a fourth pairs with a way 250 px
        # off -- close enough to pass the gate, so only the median rejects it.
        # This is the back-face case that dragged a real frame's estimate out
        # by 218 px before anchors were restricted to stated detections.
        base = [[900, 480, 950, 502], [1200, 500, 1250, 522], [1500, 520, 1550, 542]]
        boxes = base + [[2000, 800, 2050, 822]]
        cands = [cand(str(i), moved(b, 40, -15)) for i, b in enumerate(base)]
        cands.append(cand("bad", [2250, 820, 2300, 842]))  # inside the gate, wrong way
        dx, dy, n = estimate_frame_shift(boxes, cands)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(dx, -40.0, places=6)
        self.assertAlmostEqual(dy, 15.0, places=6)

    def test_none_when_nothing_to_register_against(self):
        self.assertIsNone(estimate_frame_shift([], [cand("a", [0, 0, 10, 10])]))
        self.assertIsNone(estimate_frame_shift([[0, 0, 10, 10]], []))
        self.assertIsNone(estimate_frame_shift(None, None))

    def test_none_when_no_anchor_finds_a_candidate_in_gate(self):
        # the only candidate is far beyond the gate, so this frame contributes
        # no estimate and the caller carries the previous one
        boxes = [[100, 100, 140, 118]]
        self.assertIsNone(estimate_frame_shift(boxes, [cand("a", [3000, 100, 3040, 118])]))

    def test_min_pairs_rejects_a_single_anchor_frame(self):
        boxes = [[100, 100, 140, 118]]
        cands = [cand("a", moved(boxes[0], 10, 10))]
        self.assertIsNotNone(estimate_frame_shift(boxes, cands, min_pairs=1))
        self.assertIsNone(estimate_frame_shift(boxes, cands, min_pairs=2))

    def test_size_ratio_rejects_a_wildly_mismatched_candidate(self):
        boxes = [[1000, 500, 1020, 510]]              # 20 px wide
        cands = [cand("a", [1000, 500, 1400, 700])]   # 400 px wide
        self.assertIsNone(estimate_frame_shift(boxes, cands, size_ratio=4.0))

    def test_gate_scales_with_the_larger_box(self):
        # a 200 px housing tolerates a 300 px offset at gate 6; a 20 px one does not
        big = [[1000, 500, 1200, 590]]
        self.assertIsNotNone(estimate_frame_shift(big, [cand("a", moved(big[0], 300, 0))]))
        small = [[1000, 500, 1020, 509]]
        self.assertIsNone(estimate_frame_shift(small, [cand("a", moved(small[0], 300, 0))]))

    def test_each_candidate_is_claimed_once(self):
        boxes = [[1000, 500, 1060, 524], [1010, 502, 1070, 526]]
        cands = [cand("a", [1030, 490, 1090, 514])]
        _, _, n = estimate_frame_shift(boxes, cands)
        self.assertEqual(n, 1)

    def test_degenerate_boxes_are_skipped(self):
        boxes = [[100, 100, 100, 100], [200, 200, 240, 218]]
        cands = [cand("a", moved([200, 200, 240, 218], 5, 5))]
        dx, dy, n = estimate_frame_shift(boxes, cands)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(dx, -5.0)


class ApplyTest(unittest.TestCase):
    def test_moves_the_box_and_keeps_the_raw_projection(self):
        cands = [cand("a", [100, 200, 160, 224], distance_m=71.0)]
        out = apply_shift(cands, -30.0, 12.0)
        self.assertEqual(out[0]["bbox"], [70.0, 212.0, 130.0, 236.0])
        self.assertEqual(out[0][UNSHIFTED_KEY], [100, 200, 160, 224])
        self.assertEqual(out[0][SHIFT_KEY], [-30.0, 12.0])
        self.assertEqual(out[0]["distance_m"], 71.0)
        self.assertEqual(out[0]["way_id"], "a")

    def test_does_not_mutate_the_input(self):
        cands = [cand("a", [100, 200, 160, 224])]
        apply_shift(cands, 10.0, 10.0)
        self.assertEqual(cands[0]["bbox"], [100, 200, 160, 224])
        self.assertNotIn(UNSHIFTED_KEY, cands[0])

    def test_zero_shift_is_a_no_op_without_annotating(self):
        cands = [cand("a", [100, 200, 160, 224])]
        out = apply_shift(cands, 0.0, 0.0)
        self.assertIs(out, cands)
        self.assertNotIn(SHIFT_KEY, out[0])

    def test_empty_input(self):
        self.assertEqual(apply_shift([], 5.0, 5.0), [])


class RoundTripTest(unittest.TestCase):
    def test_estimate_then_apply_lands_on_the_detections(self):
        boxes = [[900, 480, 960, 504], [1400, 500, 1470, 528], [1900, 520, 1950, 542]]
        cands = [cand(str(i), moved(b, 37, -19)) for i, b in enumerate(boxes)]
        dx, dy, _ = estimate_frame_shift(boxes, cands)
        for box, moved_cand in zip(boxes, apply_shift(cands, dx, dy)):
            for a, b in zip(box, moved_cand["bbox"]):
                self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
