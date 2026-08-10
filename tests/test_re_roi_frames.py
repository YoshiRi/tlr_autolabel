"""Unit tests for per-frame ROI candidate listing
(tlr_autolabel/review/re_review_timeline.py).
"""
import unittest
from pathlib import Path

from tlr_autolabel.review.re_review_timeline import (
    annotations_in_segment,
    build_roi_frames,
)


def make_segment(start=0, end=300, sample_tokens=None):
    seg = {"start_timestamp": start, "end_timestamp": end, "n_frames": 3}
    if sample_tokens is not None:
        seg["sample_tokens"] = sample_tokens
    return seg


def make_ann(token, channel, timestamp, sample_token, box2d=(0, 0, 10, 10), filename="img.jpg"):
    return {
        "token": token,
        "channel": channel,
        "timestamp": timestamp,
        "sample_token": sample_token,
        "box2d": list(box2d),
        "filename": filename,
    }


class AnnotationsInSegmentTest(unittest.TestCase):
    def test_filters_by_sample_tokens_when_present(self):
        seg = make_segment(sample_tokens=["s1", "s2"])
        annotations = [
            make_ann("a", "CAM_A", 100, "s1"),
            make_ann("b", "CAM_A", 200, "s3"),
        ]
        result = annotations_in_segment(annotations, seg)
        self.assertEqual([a["token"] for a in result], ["a"])

    def test_falls_back_to_timestamp_range(self):
        seg = make_segment(start=100, end=200)
        annotations = [
            make_ann("a", "CAM_A", 150, "s1"),
            make_ann("b", "CAM_A", 900, "s2"),
        ]
        result = annotations_in_segment(annotations, seg)
        self.assertEqual([a["token"] for a in result], ["a"])


class BuildRoiFramesTest(unittest.TestCase):
    def test_returns_every_frame_per_channel_not_just_top_candidates(self):
        seg = make_segment(sample_tokens=["s1", "s2", "s3"])
        annotations = [
            make_ann("a", "CAM_A", 100, "s1"),
            make_ann("b", "CAM_A", 150, "s2"),
            make_ann("c", "CAM_A", 200, "s3"),
            make_ann("d", "CAM_B", 100, "s1"),
        ]
        frames = build_roi_frames(annotations, seg, Path("/out"), Path("/root"))
        self.assertEqual(len(frames["CAM_A"]), 3)
        self.assertEqual(len(frames["CAM_B"]), 1)

    def test_sorted_by_timestamp(self):
        seg = make_segment(sample_tokens=["s1", "s2"])
        annotations = [
            make_ann("b", "CAM_A", 200, "s2"),
            make_ann("a", "CAM_A", 100, "s1"),
        ]
        frames = build_roi_frames(annotations, seg, Path("/out"), Path("/root"))
        self.assertEqual([f["token"] for f in frames["CAM_A"]], ["a", "b"])

    def test_skips_annotations_without_box2d_or_filename(self):
        seg = make_segment(sample_tokens=["s1"])
        ann = make_ann("a", "CAM_A", 100, "s1")
        del ann["box2d"]
        frames = build_roi_frames([ann], seg, Path("/out"), Path("/root"))
        self.assertEqual(frames, {})

    def test_frame_carries_box2d_and_full_image_href(self):
        seg = make_segment(sample_tokens=["s1"])
        annotations = [make_ann("a", "CAM_A", 100, "s1", box2d=(1.23, 4.0, 9.0, 12.0),
                                 filename="CAM_A/000.jpg")]
        frames = build_roi_frames(annotations, seg, Path("/root/build/tl_match"), Path("/root"))
        frame = frames["CAM_A"][0]
        self.assertEqual(frame["box2d"], [1.2, 4.0, 9.0, 12.0])
        self.assertEqual(frame["full_image"], "../../CAM_A/000.jpg")
        self.assertEqual(frame["token"], "a")
        self.assertEqual(frame["sample_token"], "s1")


if __name__ == "__main__":
    unittest.main()
