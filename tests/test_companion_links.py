"""Unit tests for cross-view link resolution
(tlr_autolabel/review/re_review_timeline.companion_links).
"""
import unittest
from pathlib import Path

from tlr_autolabel.review.re_review_timeline import companion_links

ROOT = Path("/ds")


class CompanionLinksTest(unittest.TestCase):
    def test_sibling_page_links_by_basename(self):
        links = companion_links(
            ROOT, Path("/ds/build/tl_match"),
            frame_view=Path("build/tl_match/re_frame_view.html"),
        )
        self.assertEqual(links["frame_view"], "re_frame_view.html")

    def test_page_in_another_directory_gets_a_relative_path(self):
        links = companion_links(
            ROOT, Path("/ds/build/tl_match"),
            map_view=Path("other/re_map_view.html"),
        )
        self.assertEqual(links["map_view"], "../../other/re_map_view.html")

    def test_absolute_target_is_not_rejoined_to_root(self):
        links = companion_links(
            ROOT, Path("/ds/build/tl_match"),
            map_view=Path("/elsewhere/re_map_view.html"),
        )
        self.assertEqual(links["map_view"], "../../../elsewhere/re_map_view.html")

    def test_every_requested_name_is_present(self):
        links = companion_links(
            ROOT, Path("/ds/build/tl_match"),
            frame_view=Path("build/tl_match/a.html"),
            map_view=Path("build/tl_match/b.html"),
        )
        self.assertEqual(sorted(links), ["frame_view", "map_view"])

    def test_hrefs_use_posix_separators(self):
        links = companion_links(
            ROOT, Path("/ds"), frame_view=Path("build/tl_match/re_frame_view.html")
        )
        self.assertNotIn("\\", links["frame_view"])
        self.assertEqual(links["frame_view"], "build/tl_match/re_frame_view.html")

    def test_no_targets_yields_empty_mapping(self):
        self.assertEqual(companion_links(ROOT, Path("/ds/build")), {})


if __name__ == "__main__":
    unittest.main()
