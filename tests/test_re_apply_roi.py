"""Unit tests for per-frame ROI (box2d) application
(tlr_autolabel/review/re_apply.py).
"""
import unittest

from tlr_autolabel.review.re_apply import (
    apply_review,
    normalize_box2d,
    normalize_roi_decisions,
    roi_decision_matches_annotation,
)


def make_sidecar(annotations):
    return {"schema_version": "traffic_signal_2d/v2", "annotations": annotations}


def make_ann(token, channel="CAM_A", box2d=(10.0, 10.0, 20.0, 20.0)):
    return {
        "token": token,
        "channel": channel,
        "timestamp": 100,
        "sample_token": f"sample-{token}",
        "box2d": list(box2d),
        "attributes": {"map_traffic_light_id": "101", "state": "red-circle"},
    }


class NormalizeBox2dTest(unittest.TestCase):
    def test_valid_box_normalizes(self):
        self.assertEqual(normalize_box2d([1, 2, 3, 4]), [1.0, 2.0, 3.0, 4.0])

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            normalize_box2d([1, 2, 3])

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            normalize_box2d([1, 2, "x", 4])

    def test_degenerate_box_raises(self):
        with self.assertRaises(ValueError):
            normalize_box2d([5, 5, 5, 10])


class NormalizeRoiDecisionsTest(unittest.TestCase):
    def test_valid_decision_normalizes(self):
        review = {
            "groups": [{
                "signal_group_id": "ways:101",
                "roi_decisions": [{
                    "annotation_token": "a",
                    "channel": "CAM_A",
                    "box2d": [1, 2, 3, 4],
                    "review_status": "fixed",
                }],
            }],
        }
        decisions = normalize_roi_decisions(review)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["annotation_token"], "a")
        self.assertEqual(decisions[0]["box2d"], [1.0, 2.0, 3.0, 4.0])

    def test_missing_token_fails_loud(self):
        review = {
            "groups": [{
                "signal_group_id": "ways:101",
                "roi_decisions": [{"box2d": [1, 2, 3, 4], "review_status": "fixed"}],
            }],
        }
        with self.assertRaises(SystemExit):
            normalize_roi_decisions(review)

    def test_invalid_box2d_fails_loud(self):
        review = {
            "groups": [{
                "signal_group_id": "ways:101",
                "roi_decisions": [{
                    "annotation_token": "a", "box2d": [1, 2], "review_status": "fixed",
                }],
            }],
        }
        with self.assertRaises(SystemExit):
            normalize_roi_decisions(review)

    def test_invalid_review_status_fails_loud(self):
        review = {
            "groups": [{
                "signal_group_id": "ways:101",
                "roi_decisions": [{
                    "annotation_token": "a", "box2d": [1, 2, 3, 4], "review_status": "bogus",
                }],
            }],
        }
        with self.assertRaises(SystemExit):
            normalize_roi_decisions(review)

    def test_group_without_roi_decisions_is_skipped(self):
        review = {"groups": [{"signal_group_id": "ways:101"}]}
        self.assertEqual(normalize_roi_decisions(review), [])


class RoiDecisionMatchesAnnotationTest(unittest.TestCase):
    def test_matches_exact_token(self):
        decision = {"annotation_token": "a"}
        self.assertTrue(roi_decision_matches_annotation(decision, make_ann("a")))

    def test_different_token_does_not_match(self):
        decision = {"annotation_token": "a"}
        self.assertFalse(roi_decision_matches_annotation(decision, make_ann("b")))


class ApplyReviewRoiTest(unittest.TestCase):
    def test_fixed_roi_overrides_box2d_for_matching_token_only(self):
        sidecar = make_sidecar([
            make_ann("a", box2d=(0, 0, 10, 10)),
            make_ann("b", box2d=(0, 0, 10, 10)),
        ])
        roi_decisions = [{
            "annotation_token": "a", "channel": "CAM_A",
            "box2d": [1.0, 2.0, 3.0, 4.0], "review_status": "fixed",
        }]
        reviewed, summary = apply_review(sidecar, [], [], roi_decisions)
        by_token = {a["token"]: a for a in reviewed["annotations"]}
        self.assertEqual(by_token["a"]["box2d"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(by_token["b"]["box2d"], [0, 0, 10, 10])
        self.assertEqual(summary["applied_roi_annotations"], 1)

    def test_accepted_roi_does_not_change_box2d(self):
        sidecar = make_sidecar([make_ann("a", box2d=(0, 0, 10, 10))])
        roi_decisions = [{
            "annotation_token": "a", "channel": "CAM_A",
            "box2d": [1.0, 2.0, 3.0, 4.0], "review_status": "accepted",
        }]
        reviewed, summary = apply_review(sidecar, [], [], roi_decisions)
        self.assertEqual(reviewed["annotations"][0]["box2d"], [0, 0, 10, 10])
        self.assertEqual(summary["applied_roi_annotations"], 0)

    def test_unchecked_roi_does_not_apply(self):
        sidecar = make_sidecar([make_ann("a", box2d=(0, 0, 10, 10))])
        roi_decisions = [{
            "annotation_token": "a", "channel": "CAM_A",
            "box2d": [1.0, 2.0, 3.0, 4.0], "review_status": "unchecked",
        }]
        reviewed, summary = apply_review(sidecar, [], [], roi_decisions)
        self.assertEqual(reviewed["annotations"][0]["box2d"], [0, 0, 10, 10])
        self.assertEqual(summary["applied_roi_annotations"], 0)

    def test_state_visibility_and_roi_apply_independently(self):
        sidecar = make_sidecar([make_ann("a", box2d=(0, 0, 10, 10))])
        state_decisions = [{
            "member_ways": {"101"}, "regulatory_element_ids": set(),
            "start_timestamp": 0, "end_timestamp": 200,
            "state": "green-circle", "review_status": "accepted",
        }]
        visibility_decisions = [{
            "channel": "CAM_A", "member_ways": {"101"}, "regulatory_element_ids": set(),
            "start_timestamp": 0, "end_timestamp": 200,
            "visibility": "occluded", "review_status": "fixed",
        }]
        roi_decisions = [{
            "annotation_token": "a", "channel": "CAM_A",
            "box2d": [1.0, 2.0, 3.0, 4.0], "review_status": "fixed",
        }]
        reviewed, _ = apply_review(sidecar, state_decisions, visibility_decisions, roi_decisions)
        ann = reviewed["annotations"][0]
        self.assertEqual(ann["attributes"]["state"], "green-circle")
        self.assertEqual(ann["attributes"]["visibility"], "occluded")
        self.assertEqual(ann["box2d"], [1.0, 2.0, 3.0, 4.0])

    def test_no_roi_decisions_preserves_existing_box2d(self):
        sidecar = make_sidecar([make_ann("a", box2d=(0, 0, 10, 10))])
        reviewed, _ = apply_review(sidecar, [], [], [])
        self.assertEqual(reviewed["annotations"][0]["box2d"], [0, 0, 10, 10])


if __name__ == "__main__":
    unittest.main()
