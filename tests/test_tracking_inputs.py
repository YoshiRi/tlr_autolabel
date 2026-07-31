import unittest

from match_traffic_lights import collect_low_tracking_candidates


class TrackingInputTest(unittest.TestCase):
    def test_collect_low_candidates_from_signals_and_raw_detections(self):
        payload = {
            "signals": [
                {"detector_score": 0.55, "box_xyxy": [0, 0, 10, 10]},
                {"detector_score": 0.35, "box_xyxy": [20, 20, 30, 30]},
            ],
            "raw_detections": [
                {"detector_score": 0.24, "box_xyxy": [40, 40, 50, 50]},
                {"detector_score": 0.24, "box_xyxy": [0, 0, 10, 10]},
                {"detector_score": 0.19, "box_xyxy": [60, 60, 70, 70]},
            ],
        }

        lows = collect_low_tracking_candidates(payload, high_threshold=0.5, low_threshold=0.2)

        self.assertEqual([d["box_xyxy"] for d in lows], [[20, 20, 30, 30], [40, 40, 50, 50]])


if __name__ == "__main__":
    unittest.main()
