"""Unit tests for the per-frame review view
(tlr_autolabel/review/re_frame_view.py).
"""
import unittest
from pathlib import Path

from tlr_autolabel.review.re_frame_view import (
    annotation_view,
    build_frames,
    summarize,
)

ROOT = Path("/ds")
OUT_DIR = Path("/ds/build/tl_match")


def make_ann(
    token="t",
    channel="CAM_A",
    filename="data/CAM_A/00000.jpg",
    timestamp=100,
    box=(10, 20, 30, 40),
    state="red-circle",
    kind="vehicle",
    way="",
    re_id="",
    cand="",
    reason="",
    sample_token="s1",
):
    return {
        "token": token,
        "channel": channel,
        "filename": filename,
        "timestamp": timestamp,
        "sample_token": sample_token,
        "box2d": list(box),
        "attributes": {
            "state": state,
            "signal_kind": kind,
            "visibility": "full",
            "review_status": "unchecked",
            "detector_score": "0.9",
            "source_type": "auto",
            "map_traffic_light_id": way,
            "regulatory_element_id": re_id,
            "map_candidate_id": cand,
            "unmatched_reason": reason,
            "raw_state": state,
        },
    }


class AnnotationViewTest(unittest.TestCase):
    def test_matched_when_way_id_present(self):
        self.assertTrue(annotation_view(make_ann(way="3281"))["matched"])

    def test_matched_when_only_regulatory_element_present(self):
        self.assertTrue(annotation_view(make_ann(re_id="10149"))["matched"])

    def test_unmatched_when_both_ids_empty(self):
        view = annotation_view(make_ann(way="", re_id="", reason="geometry_mismatch"))
        self.assertFalse(view["matched"])
        self.assertEqual(view["reason"], "geometry_mismatch")

    def test_carries_candidate_id_for_unmatched(self):
        self.assertEqual(annotation_view(make_ann(cand="3595"))["cand"], "3595")

    def test_missing_attributes_do_not_raise(self):
        view = annotation_view({"token": "t", "box2d": [0, 0, 1, 1]})
        self.assertEqual(view["state"], "unknown")
        self.assertFalse(view["matched"])

    def test_score_is_numeric(self):
        self.assertIsInstance(annotation_view(make_ann())["score"], float)


class BuildFramesTest(unittest.TestCase):
    def test_groups_annotations_of_one_image_into_one_frame(self):
        anns = [
            make_ann(token="a", box=(10, 0, 20, 10)),
            make_ann(token="b", box=(50, 0, 60, 10)),
        ]
        frames = build_frames(anns, ROOT, OUT_DIR, None)
        self.assertEqual(len(frames), 1)
        self.assertEqual(len(frames[0]["anns"]), 2)

    def test_separate_channels_stay_separate_frames(self):
        anns = [
            make_ann(channel="CAM_A", filename="data/CAM_A/0.jpg"),
            make_ann(channel="CAM_B", filename="data/CAM_B/0.jpg"),
        ]
        self.assertEqual(len(build_frames(anns, ROOT, OUT_DIR, None)), 2)

    def test_orders_boxes_left_to_right_within_a_frame(self):
        anns = [
            make_ann(token="right", box=(90, 0, 99, 10)),
            make_ann(token="left", box=(1, 0, 9, 10)),
        ]
        frames = build_frames(anns, ROOT, OUT_DIR, None)
        self.assertEqual([a["token"] for a in frames[0]["anns"]], ["left", "right"])

    def test_orders_frames_by_channel_then_timestamp(self):
        anns = [
            make_ann(channel="CAM_B", filename="data/CAM_B/1.jpg", timestamp=50),
            make_ann(channel="CAM_A", filename="data/CAM_A/2.jpg", timestamp=200),
            make_ann(channel="CAM_A", filename="data/CAM_A/1.jpg", timestamp=100),
        ]
        frames = build_frames(anns, ROOT, OUT_DIR, None)
        self.assertEqual(
            [(f["channel"], f["timestamp"]) for f in frames],
            [("CAM_A", 100), ("CAM_A", 200), ("CAM_B", 50)],
        )

    def test_assigns_contiguous_indices(self):
        anns = [
            make_ann(filename="data/CAM_A/1.jpg", timestamp=1),
            make_ann(filename="data/CAM_A/2.jpg", timestamp=2),
        ]
        frames = build_frames(anns, ROOT, OUT_DIR, None)
        self.assertEqual([f["i"] for f in frames], [0, 1])

    def test_channel_filter_drops_other_channels(self):
        anns = [
            make_ann(channel="CAM_A", filename="data/CAM_A/0.jpg"),
            make_ann(channel="CAM_B", filename="data/CAM_B/0.jpg"),
        ]
        frames = build_frames(anns, ROOT, OUT_DIR, {"CAM_A"})
        self.assertEqual([f["channel"] for f in frames], ["CAM_A"])

    def test_none_filter_keeps_every_channel(self):
        anns = [
            make_ann(channel="CAM_A", filename="data/CAM_A/0.jpg"),
            make_ann(channel="CAM_B", filename="data/CAM_B/0.jpg"),
        ]
        self.assertEqual(len(build_frames(anns, ROOT, OUT_DIR, None)), 2)

    def test_annotations_without_filename_are_skipped(self):
        anns = [make_ann(filename=""), make_ann(filename="data/CAM_A/0.jpg")]
        frames = build_frames(anns, ROOT, OUT_DIR, None)
        self.assertEqual(len(frames), 1)

    def test_image_src_is_relative_to_the_output_directory(self):
        frames = build_frames([make_ann()], ROOT, OUT_DIR, None)
        self.assertEqual(frames[0]["src"], "../../data/CAM_A/00000.jpg")


class SummarizeTest(unittest.TestCase):
    def test_counts_unmatched_separately(self):
        anns = [
            make_ann(token="a", box=(0, 0, 5, 5), way="3281"),
            make_ann(token="b", box=(10, 0, 15, 5), state="red-ped",
                     kind="pedestrian", reason="geometry_mismatch"),
        ]
        stats = summarize(build_frames(anns, ROOT, OUT_DIR, None))
        self.assertEqual(stats["n_annotations"], 2)
        self.assertEqual(stats["n_unmatched"], 1)
        self.assertEqual(stats["unmatched_kinds"], {"pedestrian": 1})
        self.assertEqual(stats["unmatched_reasons"], {"geometry_mismatch": 1})

    def test_counts_states_across_all_frames(self):
        anns = [
            make_ann(token="a", filename="data/CAM_A/1.jpg", timestamp=1),
            make_ann(token="b", filename="data/CAM_A/2.jpg", timestamp=2),
        ]
        stats = summarize(build_frames(anns, ROOT, OUT_DIR, None))
        self.assertEqual(stats["states"], {"red-circle": 2})
        self.assertEqual(stats["n_frames"], 2)

    def test_empty_input_summarizes_to_zeros(self):
        stats = summarize([])
        self.assertEqual(stats["n_frames"], 0)
        self.assertEqual(stats["n_annotations"], 0)
        self.assertEqual(stats["n_unmatched"], 0)


if __name__ == "__main__":
    unittest.main()
