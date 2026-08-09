"""Unit tests for the unified review launcher
(tlr_autolabel/review/re_review_all.py) and reviewed-sidecar loading.
"""
import json
import tempfile
import unittest
from pathlib import Path

from tlr_autolabel.review.re_apply import load_reviewed_sidecar
from tlr_autolabel.review.re_review_all import (
    parse_args,
    timeline_argv,
    view_argv,
)


class ViewArgvTest(unittest.TestCase):
    def test_passes_shared_flags_through(self):
        args = parse_args(["--dataset-root", "/ds", "--crop-channels", "CAM_A"])
        argv = view_argv(args, None)
        self.assertIn("--dataset-root", argv)
        self.assertIn("/ds", argv)
        self.assertEqual(argv[argv.index("--crop-channels") + 1], "CAM_A")

    def test_omits_review_when_none(self):
        self.assertNotIn("--review", view_argv(parse_args([]), None))

    def test_includes_review_when_given(self):
        argv = view_argv(parse_args([]), Path("/ds/annotation/r.json"))
        self.assertEqual(argv[argv.index("--review") + 1], "/ds/annotation/r.json")


class TimelineArgvTest(unittest.TestCase):
    def test_serves_by_default(self):
        self.assertIn("--serve", timeline_argv(parse_args([])))

    def test_no_serve_drops_the_serve_flag(self):
        self.assertNotIn("--serve", timeline_argv(parse_args(["--no-serve"])))

    def test_forwards_port_and_review_out(self):
        args = parse_args(["--port", "9100", "--review-out", "annotation/x.json"])
        argv = timeline_argv(args)
        self.assertEqual(argv[argv.index("--port") + 1], "9100")
        self.assertEqual(argv[argv.index("--review-out") + 1], "annotation/x.json")

    def test_passes_companion_view_paths_so_links_resolve(self):
        argv = timeline_argv(parse_args([]))
        self.assertIn("--frame-view", argv)
        self.assertIn("--map-view", argv)

    def test_show_empty_crop_segments_is_forwarded_only_when_set(self):
        self.assertNotIn("--show-empty-crop-segments", timeline_argv(parse_args([])))
        self.assertIn(
            "--show-empty-crop-segments",
            timeline_argv(parse_args(["--show-empty-crop-segments"])),
        )


def make_sidecar(state="unknown"):
    return {
        "annotations": [{
            "token": "a1",
            "sample_token": "s1",
            "timestamp": 1000,
            "channel": "CAM_A",
            "box2d": [0, 0, 10, 10],
            "attributes": {
                "state": state,
                "map_traffic_light_id": "3281",
                "regulatory_element_id": "10149",
                "review_status": "unchecked",
            },
        }]
    }


def make_review(state="green-circle"):
    return {
        "schema_version": "traffic_signal_re_review/v1",
        "groups": [{
            "signal_group_id": "ways:3281",
            "member_ways": ["3281"],
            "regulatory_element_ids": ["10149"],
            "decisions": [{
                "start_sample_token": "s1",
                "end_sample_token": "s1",
                "start_timestamp": 1000,
                "end_timestamp": 1000,
                "state": state,
                "review_status": "fixed",
            }],
            "visibility_decisions": {},
            "roi_decisions": [],
        }],
    }


class LoadReviewedSidecarTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "annotation").mkdir()
        self.sidecar = self.root / "annotation/sidecar.json"
        self.sidecar.write_text(json.dumps(make_sidecar()))
        self.review = self.root / "annotation/review.json"
        self.review.write_text(json.dumps(make_review()))

    def tearDown(self):
        self._tmp.cleanup()

    def test_without_review_returns_sidecar_unchanged(self):
        sidecar, summary = load_reviewed_sidecar(self.sidecar, self.root, None)
        self.assertIsNone(summary)
        self.assertEqual(
            sidecar["annotations"][0]["attributes"]["state"], "unknown"
        )

    def test_applies_the_review_when_given(self):
        sidecar, summary = load_reviewed_sidecar(self.sidecar, self.root, self.review)
        self.assertEqual(
            sidecar["annotations"][0]["attributes"]["state"], "green-circle"
        )
        self.assertEqual(summary["applied_annotations"], 1)

    def test_does_not_mutate_the_file_on_disk(self):
        load_reviewed_sidecar(self.sidecar, self.root, self.review)
        on_disk = json.loads(self.sidecar.read_text())
        self.assertEqual(on_disk["annotations"][0]["attributes"]["state"], "unknown")

    def test_missing_optional_review_is_tolerated(self):
        sidecar, summary = load_reviewed_sidecar(
            self.sidecar, self.root, self.root / "annotation/absent.json"
        )
        self.assertIsNone(summary)
        self.assertEqual(
            sidecar["annotations"][0]["attributes"]["state"], "unknown"
        )

    def test_missing_required_review_raises(self):
        with self.assertRaises(SystemExit):
            load_reviewed_sidecar(
                self.sidecar,
                self.root,
                self.root / "annotation/absent.json",
                required=True,
            )


if __name__ == "__main__":
    unittest.main()
