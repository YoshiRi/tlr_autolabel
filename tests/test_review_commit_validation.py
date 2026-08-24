"""Unit tests for validation at the commit boundary.

Covers the review feedback that `validate_review_payload()` only checked the
document's outer shape, so a review with an invalid state token or a malformed
`box2d` could be committed atomically and only blow up later during apply.
"""
import unittest

from tlr_autolabel.review.re_apply import (
    ReviewValidationError,
    normalize_decisions,
    validate_review_semantics,
)
from tlr_autolabel.review.re_review_timeline import validate_review_payload

SAMPLE_TS = {"s1": 1000, "s2": 2000}


def review(decisions=None, visibility=None, roi=None):
    return {
        "schema_version": "traffic_signal_re_review/v1",
        "groups": [{
            "signal_group_id": "ways:3281",
            "member_ways": ["3281"],
            "regulatory_element_ids": ["10149"],
            "decisions": decisions if decisions is not None else [],
            "visibility_decisions": visibility or {},
            "roi_decisions": roi or [],
        }],
    }


def decision(state="green-circle", status="fixed"):
    return {
        "start_sample_token": "s1",
        "end_sample_token": "s2",
        "state": state,
        "review_status": status,
    }


class ValidationErrorTypeTest(unittest.TestCase):
    """The post-commit view refresh catches `Exception`; a `SystemExit` from
    the normalizers would have escaped it and killed the response mid-flight.
    """

    def test_is_catchable_as_a_plain_exception(self):
        with self.assertRaises(Exception):
            normalize_decisions(review([decision(state="not-a-state")]), SAMPLE_TS)

    def test_is_a_value_error_not_a_system_exit(self):
        self.assertTrue(issubclass(ReviewValidationError, ValueError))
        self.assertFalse(issubclass(ReviewValidationError, SystemExit))

    def test_normalizers_raise_the_catchable_type(self):
        with self.assertRaises(ReviewValidationError):
            normalize_decisions(review([decision(state="not-a-state")]), SAMPLE_TS)


class ShapeValidationTest(unittest.TestCase):
    """`validate_review_payload` stays a cheap shape check; it is not expected
    to catch semantic problems, which is why the semantic pass exists."""

    def test_accepts_a_well_formed_document(self):
        self.assertIsNone(validate_review_payload(review([decision()])))

    def test_still_rejects_a_wrong_schema_version(self):
        payload = review()
        payload["schema_version"] = "traffic_signal_re_review/v2"
        self.assertIsNotNone(validate_review_payload(payload))

    def test_does_not_see_an_invalid_state(self):
        self.assertIsNone(
            validate_review_payload(review([decision(state="not-a-state")]))
        )


class SemanticValidationTest(unittest.TestCase):
    def test_valid_document_reports_no_problem(self):
        self.assertIsNone(validate_review_semantics(review([decision()]), SAMPLE_TS))

    def test_empty_document_reports_no_problem(self):
        self.assertIsNone(validate_review_semantics(review(), SAMPLE_TS))

    def test_rejects_an_invalid_state_token(self):
        problem = validate_review_semantics(
            review([decision(state="not-a-state")]), SAMPLE_TS
        )
        self.assertIsNotNone(problem)
        self.assertIn("not-a-state", problem)

    def test_rejects_an_invalid_review_status(self):
        problem = validate_review_semantics(
            review([decision(status="probably")]), SAMPLE_TS
        )
        self.assertIsNotNone(problem)
        self.assertIn("review_status", problem)

    def test_rejects_a_malformed_box2d(self):
        problem = validate_review_semantics(
            review(roi=[{"annotation_token": "t", "box2d": [1, 2],
                         "review_status": "fixed"}]),
            SAMPLE_TS,
        )
        self.assertIsNotNone(problem)
        self.assertIn("box2d", problem)

    def test_rejects_a_roi_decision_without_an_annotation_token(self):
        problem = validate_review_semantics(
            review(roi=[{"box2d": [1, 2, 3, 4], "review_status": "fixed"}]),
            SAMPLE_TS,
        )
        self.assertIsNotNone(problem)
        self.assertIn("annotation_token", problem)

    def test_rejects_an_invalid_visibility_value(self):
        problem = validate_review_semantics(
            review(visibility={"CAM_A": [{
                "start_sample_token": "s1", "end_sample_token": "s2",
                "visibility": "mostly", "review_status": "fixed",
            }]}),
            SAMPLE_TS,
        )
        self.assertIsNotNone(problem)
        self.assertIn("visibility", problem)

    def test_rejects_a_group_without_any_identity(self):
        payload = review([decision()])
        payload["groups"][0]["signal_group_id"] = ""
        payload["groups"][0]["member_ways"] = []
        payload["groups"][0]["regulatory_element_ids"] = []
        self.assertIsNotNone(validate_review_semantics(payload, SAMPLE_TS))

    def test_reports_every_problem_not_just_the_first(self):
        problem = validate_review_semantics(
            review([decision(state="not-a-state"), decision(status="probably")]),
            SAMPLE_TS,
        )
        self.assertIn("2 problem(s)", problem)


if __name__ == "__main__":
    unittest.main()
