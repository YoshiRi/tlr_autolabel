"""Unit tests for per-camera visibility application
(tlr_autolabel/review/re_apply.py).
"""
import unittest

from tlr_autolabel.review.re_apply import (
    ReviewValidationError,
    apply_review,
    normalize_visibility_decisions,
    visibility_decision_matches_annotation,
)


def make_sidecar(annotations):
    return {"schema_version": "traffic_signal_2d/v2", "annotations": annotations}


def make_ann(token, channel, timestamp, map_id="101", visibility="full"):
    return {
        "token": token, "channel": channel, "timestamp": timestamp,
        "sample_token": f"sample-{token}",
        "attributes": {"map_traffic_light_id": map_id, "visibility": visibility, "state": "red-circle"},
    }


class NormalizeVisibilityDecisionsTest(unittest.TestCase):
    def test_valid_decision_normalizes(self):
        review = {
            "groups": [{
                "signal_group_id": "ways:101",
                "member_ways": ["101"],
                "regulatory_element_ids": [],
                "visibility_decisions": {
                    "CAM_A": [{
                        "start_timestamp": 100, "end_timestamp": 200,
                        "visibility": "occluded", "review_status": "fixed",
                    }],
                },
            }],
        }
        decisions = normalize_visibility_decisions(review, {})
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["channel"], "CAM_A")
        self.assertEqual(decisions[0]["visibility"], "occluded")

    def test_invalid_visibility_value_fails_loud(self):
        review = {
            "groups": [{
                "signal_group_id": "ways:101",
                "member_ways": ["101"],
                "visibility_decisions": {
                    "CAM_A": [{"start_timestamp": 100, "end_timestamp": 200,
                               "visibility": "bogus", "review_status": "fixed"}],
                },
            }],
        }
        with self.assertRaises(ReviewValidationError):
            normalize_visibility_decisions(review, {})

    def test_group_without_visibility_decisions_is_skipped(self):
        review = {"groups": [{"signal_group_id": "ways:101", "member_ways": ["101"]}]}
        self.assertEqual(normalize_visibility_decisions(review, {}), [])


class VisibilityDecisionMatchesAnnotationTest(unittest.TestCase):
    def test_matches_same_channel_and_way_and_time(self):
        decision = {"channel": "CAM_A", "member_ways": {"101"}, "regulatory_element_ids": set(),
                    "start_timestamp": 100, "end_timestamp": 200}
        ann = make_ann("a", "CAM_A", 150)
        self.assertTrue(visibility_decision_matches_annotation(decision, ann))

    def test_different_channel_does_not_match(self):
        decision = {"channel": "CAM_A", "member_ways": {"101"}, "regulatory_element_ids": set(),
                    "start_timestamp": 100, "end_timestamp": 200}
        ann = make_ann("a", "CAM_B", 150)
        self.assertFalse(visibility_decision_matches_annotation(decision, ann))


class ApplyReviewVisibilityTest(unittest.TestCase):
    def test_accepted_visibility_overrides_matching_channel_only(self):
        sidecar = make_sidecar([
            make_ann("a", "CAM_A", 150, visibility="full"),
            make_ann("b", "CAM_B", 150, visibility="full"),
        ])
        visibility_decisions = [{
            "channel": "CAM_A", "member_ways": {"101"}, "regulatory_element_ids": set(),
            "start_timestamp": 100, "end_timestamp": 200,
            "visibility": "occluded", "review_status": "fixed",
        }]
        reviewed, summary = apply_review(sidecar, [], visibility_decisions)
        by_token = {a["token"]: a for a in reviewed["annotations"]}
        self.assertEqual(by_token["a"]["attributes"]["visibility"], "occluded")
        self.assertEqual(by_token["b"]["attributes"]["visibility"], "full")
        self.assertEqual(summary["applied_visibility_annotations"], 1)

    def test_unchecked_visibility_decision_does_not_apply(self):
        sidecar = make_sidecar([make_ann("a", "CAM_A", 150, visibility="full")])
        visibility_decisions = [{
            "channel": "CAM_A", "member_ways": {"101"}, "regulatory_element_ids": set(),
            "start_timestamp": 100, "end_timestamp": 200,
            "visibility": "occluded", "review_status": "unchecked",
        }]
        reviewed, summary = apply_review(sidecar, [], visibility_decisions)
        self.assertEqual(reviewed["annotations"][0]["attributes"]["visibility"], "full")
        self.assertEqual(summary["applied_visibility_annotations"], 0)

    def test_state_and_visibility_apply_independently(self):
        sidecar = make_sidecar([make_ann("a", "CAM_A", 150, visibility="full")])
        state_decisions = [{
            "member_ways": {"101"}, "regulatory_element_ids": set(),
            "start_timestamp": 100, "end_timestamp": 200,
            "state": "green-circle", "review_status": "accepted",
        }]
        visibility_decisions = [{
            "channel": "CAM_A", "member_ways": {"101"}, "regulatory_element_ids": set(),
            "start_timestamp": 100, "end_timestamp": 200,
            "visibility": "occluded", "review_status": "fixed",
        }]
        reviewed, _ = apply_review(sidecar, state_decisions, visibility_decisions)
        attrs = reviewed["annotations"][0]["attributes"]
        self.assertEqual(attrs["state"], "green-circle")
        self.assertEqual(attrs["visibility"], "occluded")

    def test_no_visibility_decisions_preserves_existing_visibility(self):
        sidecar = make_sidecar([make_ann("a", "CAM_A", 150, visibility="partial")])
        reviewed, _ = apply_review(sidecar, [], [])
        self.assertEqual(reviewed["annotations"][0]["attributes"]["visibility"], "partial")


if __name__ == "__main__":
    unittest.main()
