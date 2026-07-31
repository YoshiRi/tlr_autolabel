import unittest

from evaluate_signals import build_detection_records


class EvaluateSignalRecordsTest(unittest.TestCase):
    def test_detection_records_keep_temporal_tracking_fields(self):
        sidecar = {
            "annotations": [
                {
                    "token": "ann-1",
                    "sample_token": "sample-1",
                    "sample_data_token": "sd-1",
                    "channel": "CAM_FRONT",
                    "timestamp": 123,
                    "box2d": [0.0, 0.0, 10.0, 12.0],
                    "attributes": {
                        "state": "unknown",
                        "signal_kind": "unknown",
                        "review_status": "unchecked",
                        "map_traffic_light_id": "101",
                        "regulatory_element_id": "201",
                        "detector_score": "",
                        "source_type": "propagated",
                        "temporal_source": "propagated",
                        "track_id": "tltrk-000001",
                        "tracking_status": "lost",
                        "tracking_lost_frames": "2",
                    },
                }
            ]
        }

        record = build_detection_records(sidecar, {})[0]

        self.assertEqual(record["source_type"], "propagated")
        self.assertEqual(record["temporal_source"], "propagated")
        self.assertEqual(record["track_id"], "tltrk-000001")
        self.assertEqual(record["tracking_status"], "lost")
        self.assertEqual(record["tracking_lost_frames"], 2)


if __name__ == "__main__":
    unittest.main()
