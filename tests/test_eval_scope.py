"""Evaluation scope: a detection with no readable state is not a false positive.

GT does not annotate the back of a housing -- a face with no lamp has no state
to evaluate -- so an unmatched detection that read no state has nothing to be
scored against. On 2821adc7 that was 202 of 236 unmatched detections and it
held precision at 0.660 while the model and the annotator actually agreed.

These pin that the exclusion is a *reclassification*, not a discount: the rows
still leave the evaluator under `pred_not_evaluated`, TP/FN never move, and the
old number stays reproducible.

    python3 -m pytest tests/test_eval_scope.py -p no:anyio
"""
from __future__ import annotations

import unittest

from tlr_autolabel.eval.l1_vs_t4 import evaluate


def gt(box, state="red-circle", category="red"):
    return {"box": list(box), "state": state, "category": category,
            "signal_kind": "vehicle"}


def pred(box, state="red-circle", score=0.9, kind="vehicle"):
    return {"box": list(box), "state": state, "detector_score": score,
            "signal_kind": kind, "filename": "data/CAM/00000.jpg"}


HIT = [100, 100, 160, 124]
ELSEWHERE = [2000, 800, 2060, 824]
ELSEWHERE2 = [2400, 900, 2460, 924]


class ScopeTest(unittest.TestCase):
    def report(self, gts, preds, **kw):
        return evaluate({"t": gts}, {"t": preds}, 0.3, **kw)["detection"]["overall"]

    def test_stateless_unmatched_detection_is_not_a_false_positive(self):
        o = self.report([gt(HIT)], [pred(HIT), pred(ELSEWHERE, state="unknown")])
        self.assertEqual((o["tp"], o["fp"], o["fn"]), (1, 0, 0))
        self.assertEqual(o["not_evaluated"], 1)
        self.assertEqual(o["precision"], 1.0)

    def test_a_stated_unmatched_detection_is_still_a_false_positive(self):
        o = self.report([gt(HIT)], [pred(HIT), pred(ELSEWHERE, state="green-circle")])
        self.assertEqual((o["tp"], o["fp"], o["fn"]), (1, 1, 0))
        self.assertEqual(o["not_evaluated"], 0)

    def test_the_old_behaviour_stays_reproducible(self):
        preds = [pred(HIT), pred(ELSEWHERE, state="unknown")]
        o = self.report([gt(HIT)], preds, score_stateless_as_fp=True)
        self.assertEqual((o["tp"], o["fp"], o["fn"]), (1, 1, 0))
        self.assertEqual(o["not_evaluated"], 0)
        self.assertEqual(o["precision"], 0.5)

    def test_excluded_rows_are_reported_not_discarded(self):
        report = evaluate({"t": [gt(HIT)]},
                          {"t": [pred(HIT), pred(ELSEWHERE, state="unknown")]}, 0.3)
        self.assertEqual(len(report["pred_not_evaluated"]), 1)
        self.assertEqual(report["pred_not_evaluated"][0]["pred_box"], ELSEWHERE)
        self.assertEqual(report["pred_false_positives"], [])

    def test_both_precisions_are_reported_side_by_side(self):
        o = self.report([gt(HIT)], [pred(HIT), pred(ELSEWHERE, state="unknown")])
        self.assertEqual(o["precision"], 1.0)
        self.assertEqual(o["precision_counting_stateless_as_fp"], 0.5)

    def test_recall_and_misses_are_untouched_by_the_scope(self):
        # the excluded detection must not rescue a missed GT box
        for flag in (False, True):
            with self.subTest(score_stateless_as_fp=flag):
                o = self.report([gt(HIT)], [pred(ELSEWHERE, state="unknown")],
                                score_stateless_as_fp=flag)
                self.assertEqual((o["tp"], o["fn"]), (0, 1))
                self.assertEqual(o["recall"], 0.0)

    def test_a_matched_detection_with_no_state_is_still_a_true_positive(self):
        # only *unmatched* stateless detections leave scope; a stateless
        # detection sitting on an annotated box is a real localization hit
        o = self.report([gt(HIT)], [pred(HIT, state="unknown")])
        self.assertEqual((o["tp"], o["fp"], o["fn"]), (1, 0, 0))
        self.assertEqual(o["not_evaluated"], 0)

    def test_f1_uses_the_scoped_false_positives(self):
        o = self.report([gt(HIT)], [pred(HIT), pred(ELSEWHERE, state="unknown")])
        self.assertEqual(o["f1"], 1.0)

    def test_several_excluded_rows_accumulate(self):
        o = self.report([gt(HIT)], [pred(HIT),
                                    pred(ELSEWHERE, state="unknown"),
                                    pred(ELSEWHERE2, state="unknown")])
        self.assertEqual(o["not_evaluated"], 2)
        self.assertEqual(o["fp"], 0)

    def test_per_kind_false_positives_follow_the_same_scope(self):
        report = evaluate({"t": [gt(HIT)]},
                          {"t": [pred(HIT),
                                 pred(ELSEWHERE, state="unknown", kind="pedestrian")]}, 0.3)
        by_kind = report["detection"]["by_signal_kind"]
        self.assertNotIn("pedestrian", by_kind)


if __name__ == "__main__":
    unittest.main()
