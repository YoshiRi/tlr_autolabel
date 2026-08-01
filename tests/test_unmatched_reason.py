"""Unit tests for the unmatched_reason() classifier.

Prerequisite coverage REFACTOR_PLAN.md calls for before attempting the
Phase 5 L3 map-enrichment split ("duplicate prevention, and map-missing
behavior"): this pins the review-triage classification for a detection
with no map candidate in view (map-missing) and one whose nearest
candidate was already claimed by another detection (duplicate
prevention), alongside the other two branches, without touching
the CLI orchestration itself.
"""
import unittest

from tlr_autolabel.map.association import unmatched_reason


def det(box):
    return {"box_xyxy": box}


def candidate(box):
    return {"bbox": box}


class UnmatchedReasonTest(unittest.TestCase):
    def test_unknown_state_is_backside(self):
        reason = unmatched_reason(det([0, 0, 10, 10]), "unknown",
                                   [candidate([0, 0, 10, 10])], {})
        self.assertEqual(reason, "state_unknown_backside")

    def test_no_candidates_in_view_is_map_missing(self):
        reason = unmatched_reason(det([0, 0, 10, 10]), "red-circle", [], {})
        self.assertEqual(reason, "no_map_candidate_in_view")

    def test_nearest_candidate_already_matched_is_candidate_taken(self):
        candidates = [candidate([0, 0, 10, 10])]
        # candidate index 0 already assigned to some other detection
        matches = {5: 0}
        reason = unmatched_reason(det([1, 1, 11, 11]), "red-circle", candidates, matches)
        self.assertEqual(reason, "candidate_taken")

    def test_far_nearest_candidate_is_beyond_gate(self):
        candidates = [candidate([1000, 1000, 1010, 1010])]
        reason = unmatched_reason(det([0, 0, 10, 10]), "red-circle", candidates, {})
        self.assertEqual(reason, "beyond_gate")

    def test_close_unmatched_candidate_is_geometry_mismatch(self):
        candidates = [candidate([0, 0, 10, 10])]
        reason = unmatched_reason(det([1, 1, 11, 11]), "red-circle", candidates, {})
        self.assertEqual(reason, "geometry_mismatch")


if __name__ == "__main__":
    unittest.main()
