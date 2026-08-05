"""Crop-channel resolution for the RE review timeline.

t4devkit allows any CAM_* channel name, so the usable channel set is discovered
from the data instead of being hard-coded. A TLR dataset typically ships
CAM_TRAFFIC_LIGHT_NEAR/FAR, which the old CAM_FRONT* default silently filtered
out (producing an empty timeline). Only explicitly rear-facing cameras are
dropped.
"""
import unittest

from tlr_autolabel.review.re_review_timeline import (
    auto_crop_channels,
    is_rear_channel,
    resolve_crop_channels,
)


class RearChannelTest(unittest.TestCase):
    def test_rear_names_detected(self):
        for ch in ("CAM_BACK", "CAM_REAR", "CAM_BACK_LEFT", "CAM_REAR_RIGHT",
                   "CAM_BACKWARD"):
            self.assertTrue(is_rear_channel(ch), ch)

    def test_forward_names_not_flagged(self):
        for ch in ("CAM_FRONT", "CAM_TRAFFIC_LIGHT_NEAR", "CAM_TRAFFIC_LIGHT_FAR",
                   "CAM_FRONT_LEFT"):
            self.assertFalse(is_rear_channel(ch), ch)

    def test_substring_does_not_false_positive(self):
        # token-based match: "BACKUP" must not count as "BACK"
        self.assertFalse(is_rear_channel("CAM_BACKUP_FRONT"))


class AutoCropChannelsTest(unittest.TestCase):
    def test_discovers_tlr_channels(self):
        got = auto_crop_channels(
            {"CAM_TRAFFIC_LIGHT_NEAR", "CAM_TRAFFIC_LIGHT_FAR", "CAM_BACK"})
        self.assertEqual(got, {"CAM_TRAFFIC_LIGHT_NEAR", "CAM_TRAFFIC_LIGHT_FAR"})

    def test_ignores_non_camera_and_empty(self):
        got = auto_crop_channels({"LIDAR_TOP", "", None, "CAM_FRONT"})
        self.assertEqual(got, {"CAM_FRONT"})


class ResolveCropChannelsTest(unittest.TestCase):
    def setUp(self):
        self.available = {"CAM_TRAFFIC_LIGHT_NEAR", "CAM_TRAFFIC_LIGHT_FAR", "CAM_BACK"}

    def test_auto_uses_available_minus_rear(self):
        self.assertEqual(
            resolve_crop_channels("auto", self.available),
            {"CAM_TRAFFIC_LIGHT_NEAR", "CAM_TRAFFIC_LIGHT_FAR"})

    def test_all_disables_filtering(self):
        self.assertIsNone(resolve_crop_channels("all", self.available))
        self.assertIsNone(resolve_crop_channels("*", self.available))

    def test_explicit_list_is_honored(self):
        self.assertEqual(
            resolve_crop_channels("CAM_TRAFFIC_LIGHT_FAR", self.available),
            {"CAM_TRAFFIC_LIGHT_FAR"})

    def test_explicit_list_may_name_absent_channels(self):
        # explicit means explicit: no intersection with what the data has
        self.assertEqual(resolve_crop_channels("CAM_FRONT", self.available), {"CAM_FRONT"})


if __name__ == "__main__":
    unittest.main()
