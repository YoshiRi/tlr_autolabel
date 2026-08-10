"""Unit tests for the --serve save path
(tlr_autolabel/review/re_review_timeline.py).
"""
import json
import tempfile
import unittest
from pathlib import Path

from tlr_autolabel.review.re_review_timeline import (
    diff_reviews,
    draft_path_for,
    index_review,
    validate_review_payload,
    write_review_file,
)


def make_payload(groups=None):
    return {
        "schema_version": "traffic_signal_re_review/v1",
        "groups": groups if groups is not None else [],
    }


def make_group(gid="ways:1", decisions=None, visibility=None, roi=None):
    return {
        "signal_group_id": gid,
        "decisions": decisions or [],
        "visibility_decisions": visibility or {},
        "roi_decisions": roi or [],
    }


def state_decision(start="s1", end="s2", state="green-circle", status="fixed", note=""):
    return {
        "start_sample_token": start,
        "end_sample_token": end,
        "state": state,
        "review_status": status,
        "note": note,
    }


class ValidateReviewPayloadTest(unittest.TestCase):
    def test_accepts_well_formed_payload(self):
        self.assertIsNone(validate_review_payload(make_payload()))

    def test_rejects_non_object(self):
        for payload in ([], "x", 3, None):
            self.assertIsNotNone(validate_review_payload(payload))

    def test_rejects_wrong_schema_version(self):
        payload = make_payload()
        payload["schema_version"] = "traffic_signal_re_review/v2"
        self.assertIn("schema_version", validate_review_payload(payload))

    def test_rejects_missing_schema_version(self):
        self.assertIsNotNone(validate_review_payload({"groups": []}))

    def test_rejects_missing_groups(self):
        self.assertIn("groups", validate_review_payload(
            {"schema_version": "traffic_signal_re_review/v1"}
        ))

    def test_rejects_non_list_groups(self):
        payload = make_payload()
        payload["groups"] = {}
        self.assertIn("groups", validate_review_payload(payload))


class WriteReviewFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.target = self.root / "annotation" / "traffic_signal_re_review.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_creates_parent_directories(self):
        write_review_file(self.target, make_payload())
        self.assertTrue(self.target.exists())
        self.assertEqual(json.loads(self.target.read_text()), make_payload())

    def test_first_write_leaves_no_backup(self):
        write_review_file(self.target, make_payload())
        self.assertFalse(self.target.with_suffix(".json.bak").exists())

    def test_second_write_backs_up_previous_content(self):
        first = make_payload([{"signal_group_id": "ways:1"}])
        second = make_payload([{"signal_group_id": "ways:2"}])
        write_review_file(self.target, first)
        write_review_file(self.target, second)
        self.assertEqual(json.loads(self.target.read_text()), second)
        backup = self.target.with_suffix(".json.bak")
        self.assertEqual(json.loads(backup.read_text()), first)

    def test_leaves_no_temp_file_behind(self):
        write_review_file(self.target, make_payload())
        leftovers = [p.name for p in self.target.parent.iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftovers, [])

    def test_writes_trailing_newline(self):
        write_review_file(self.target, make_payload())
        self.assertTrue(self.target.read_text().endswith("\n"))

    def test_backup_can_be_skipped(self):
        write_review_file(self.target, make_payload([{"signal_group_id": "a"}]))
        write_review_file(self.target, make_payload(), backup=False)
        self.assertFalse(self.target.with_suffix(".json.bak").exists())


class DraftPathTest(unittest.TestCase):
    def test_sits_next_to_the_committed_file(self):
        target = Path("/ds/annotation/traffic_signal_re_review.json")
        draft = draft_path_for(target)
        self.assertEqual(draft.parent, target.parent)
        self.assertEqual(draft.name, "traffic_signal_re_review.draft.json")

    def test_draft_is_distinct_from_target(self):
        target = Path("/ds/annotation/traffic_signal_re_review.json")
        self.assertNotEqual(draft_path_for(target), target)


class IndexReviewTest(unittest.TestCase):
    def test_keys_each_decision_kind_separately(self):
        payload = make_payload([make_group(
            decisions=[state_decision()],
            visibility={"CAM_A": [{"start_sample_token": "s1",
                                   "end_sample_token": "s2",
                                   "visibility": "occluded"}]},
            roi=[{"annotation_token": "tok", "box2d": [1, 2, 3, 4]}],
        )])
        index = index_review(payload)
        self.assertEqual(list(index["state"]), [("ways:1", "s1", "s2")])
        self.assertEqual(list(index["visibility"]), [("ways:1", "CAM_A", "s1", "s2")])
        self.assertEqual(list(index["roi"]), [("ways:1", "tok")])

    def test_tolerates_missing_optional_sections(self):
        index = index_review(make_payload([{"signal_group_id": "ways:1"}]))
        self.assertEqual(index["state"], {})
        self.assertEqual(index["visibility"], {})
        self.assertEqual(index["roi"], {})

    def test_same_interval_in_different_groups_stays_distinct(self):
        payload = make_payload([
            make_group("ways:1", decisions=[state_decision()]),
            make_group("ways:2", decisions=[state_decision()]),
        ])
        self.assertEqual(len(index_review(payload)["state"]), 2)


class DiffReviewsTest(unittest.TestCase):
    def test_no_differences_against_itself(self):
        payload = make_payload([make_group(decisions=[state_decision()])])
        diff = diff_reviews(payload, payload)
        self.assertEqual(diff["total"], 0)

    def test_detects_added_decision_against_empty_file(self):
        new = make_payload([make_group(decisions=[state_decision()])])
        diff = diff_reviews({}, new)
        self.assertEqual(diff["total"], 1)
        self.assertEqual(len(diff["state"]["added"]), 1)
        self.assertEqual(diff["state"]["added"][0]["after"]["state"], "green-circle")

    def test_detects_changed_state_on_same_interval(self):
        old = make_payload([make_group(decisions=[state_decision(state="green-circle")])])
        new = make_payload([make_group(decisions=[state_decision(state="red-circle")])])
        diff = diff_reviews(old, new)
        self.assertEqual(len(diff["state"]["changed"]), 1)
        entry = diff["state"]["changed"][0]
        self.assertEqual(entry["before"]["state"], "green-circle")
        self.assertEqual(entry["after"]["state"], "red-circle")
        self.assertEqual(diff["state"]["added"], [])

    def test_detects_removed_decision(self):
        old = make_payload([make_group(decisions=[state_decision()])])
        diff = diff_reviews(old, make_payload())
        self.assertEqual(len(diff["state"]["removed"]), 1)

    def test_note_only_edit_counts_as_a_change(self):
        old = make_payload([make_group(decisions=[state_decision(note="")])])
        new = make_payload([make_group(decisions=[state_decision(note="checked")])])
        self.assertEqual(diff_reviews(old, new)["total"], 1)

    def test_derived_bookkeeping_fields_are_not_changes(self):
        old_d = state_decision()
        old_d.update({"source": "manual_timeline_review", "n_frames": 3})
        new_d = state_decision()
        new_d.update({"source": "manual_timeline_review", "n_frames": 9})
        old = make_payload([make_group(decisions=[old_d])])
        new = make_payload([make_group(decisions=[new_d])])
        self.assertEqual(diff_reviews(old, new)["total"], 0)

    def test_detects_roi_box_change(self):
        old = make_payload([make_group(
            roi=[{"annotation_token": "t", "box2d": [1, 2, 3, 4], "review_status": "fixed"}]
        )])
        new = make_payload([make_group(
            roi=[{"annotation_token": "t", "box2d": [5, 6, 7, 8], "review_status": "fixed"}]
        )])
        diff = diff_reviews(old, new)
        self.assertEqual(len(diff["roi"]["changed"]), 1)
        self.assertEqual(diff["roi"]["changed"][0]["after"]["box2d"], [5, 6, 7, 8])

    def test_visibility_change_is_scoped_per_channel(self):
        def payload(channel):
            return make_payload([make_group(visibility={channel: [
                {"start_sample_token": "s1", "end_sample_token": "s2",
                 "visibility": "occluded", "review_status": "fixed"}
            ]})])
        diff = diff_reviews(payload("CAM_A"), payload("CAM_B"))
        self.assertEqual(len(diff["visibility"]["added"]), 1)
        self.assertEqual(len(diff["visibility"]["removed"]), 1)
        self.assertEqual(len(diff["visibility"]["changed"]), 0)

    def test_total_sums_every_kind_and_bucket(self):
        old = make_payload([make_group(decisions=[state_decision(start="keep")])])
        new = make_payload([make_group(
            decisions=[state_decision(start="new")],
            roi=[{"annotation_token": "t", "box2d": [1, 2, 3, 4]}],
        )])
        diff = diff_reviews(old, new)
        # one state added, one state removed, one roi added
        self.assertEqual(diff["total"], 3)


if __name__ == "__main__":
    unittest.main()
