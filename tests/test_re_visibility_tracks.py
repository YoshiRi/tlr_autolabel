"""Unit tests for per-camera visibility segmentation
(tlr_autolabel/review/re_review_timeline.py).
"""
import unittest

from tlr_autolabel.review.re_review_timeline import (
    build_visibility_tracks,
    segment_visibility,
    visibility_observations_by_channel,
)


class SegmentVisibilityTest(unittest.TestCase):
    def test_contiguous_same_visibility_merges(self):
        obs = [
            {"timestamp": 1, "sample_token": "s1", "visibility": "full"},
            {"timestamp": 2, "sample_token": "s2", "visibility": "full"},
            {"timestamp": 3, "sample_token": "s3", "visibility": "full"},
        ]
        segs = segment_visibility(obs)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["n_frames"], 3)
        self.assertEqual(segs[0]["start_sample_token"], "s1")
        self.assertEqual(segs[0]["end_sample_token"], "s3")

    def test_visibility_change_splits_segments(self):
        obs = [
            {"timestamp": 1, "sample_token": "s1", "visibility": "full"},
            {"timestamp": 2, "sample_token": "s2", "visibility": "occluded"},
            {"timestamp": 3, "sample_token": "s3", "visibility": "occluded"},
            {"timestamp": 4, "sample_token": "s4", "visibility": "full"},
        ]
        segs = segment_visibility(obs)
        self.assertEqual([s["visibility"] for s in segs], ["full", "occluded", "full"])
        self.assertEqual(segs[1]["n_frames"], 2)

    def test_empty_observations(self):
        self.assertEqual(segment_visibility([]), [])


class VisibilityObservationsByChannelTest(unittest.TestCase):
    def test_groups_by_channel_and_matches_group(self):
        row = {"member_ways": ["101"], "regulatory_element_ids": []}
        annotations = [
            {"channel": "CAM_A", "timestamp": 1, "sample_token": "s1",
             "attributes": {"map_traffic_light_id": "101", "visibility": "full"}},
            {"channel": "CAM_B", "timestamp": 1, "sample_token": "s1",
             "attributes": {"map_traffic_light_id": "101", "visibility": "occluded"}},
            {"channel": "CAM_A", "timestamp": 2, "sample_token": "s2",
             "attributes": {"map_traffic_light_id": "999", "visibility": "full"}},
        ]
        by_channel = visibility_observations_by_channel(annotations, row)
        self.assertEqual(set(by_channel.keys()), {"CAM_A", "CAM_B"})
        self.assertEqual(len(by_channel["CAM_A"]), 1)  # the way-999 row is excluded
        self.assertEqual(by_channel["CAM_B"][0]["visibility"], "occluded")

    def test_missing_visibility_defaults_to_unknown(self):
        row = {"member_ways": ["101"], "regulatory_element_ids": []}
        annotations = [
            {"channel": "CAM_A", "timestamp": 1, "sample_token": "s1",
             "attributes": {"map_traffic_light_id": "101"}},
        ]
        by_channel = visibility_observations_by_channel(annotations, row)
        self.assertEqual(by_channel["CAM_A"][0]["visibility"], "unknown")


class BuildVisibilityTracksTest(unittest.TestCase):
    def test_restricts_to_crop_channels(self):
        rows = [{"signal_group_id": "ways:101", "member_ways": ["101"], "regulatory_element_ids": []}]
        annotations = [
            {"channel": "CAM_KEEP", "timestamp": 1, "sample_token": "s1",
             "attributes": {"map_traffic_light_id": "101", "visibility": "full"}},
            {"channel": "CAM_DROP", "timestamp": 1, "sample_token": "s1",
             "attributes": {"map_traffic_light_id": "101", "visibility": "occluded"}},
        ]
        build_visibility_tracks(annotations, rows, crop_channels={"CAM_KEEP"})
        self.assertEqual(set(rows[0]["visibility_tracks"].keys()), {"CAM_KEEP"})

    def test_none_crop_channels_keeps_all(self):
        rows = [{"signal_group_id": "ways:101", "member_ways": ["101"], "regulatory_element_ids": []}]
        annotations = [
            {"channel": "CAM_A", "timestamp": 1, "sample_token": "s1",
             "attributes": {"map_traffic_light_id": "101", "visibility": "full"}},
            {"channel": "CAM_B", "timestamp": 1, "sample_token": "s1",
             "attributes": {"map_traffic_light_id": "101", "visibility": "full"}},
        ]
        build_visibility_tracks(annotations, rows, crop_channels=None)
        self.assertEqual(set(rows[0]["visibility_tracks"].keys()), {"CAM_A", "CAM_B"})


if __name__ == "__main__":
    unittest.main()
