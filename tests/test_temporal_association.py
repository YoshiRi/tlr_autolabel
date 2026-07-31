import unittest

from temporal_association import TemporalAssociator, TemporalTrackingConfig


class TemporalAssociatorTest(unittest.TestCase):
    def test_low_detection_updates_existing_track_only(self):
        cfg = TemporalTrackingConfig(enabled=True, low_score=0.2)
        associator = TemporalAssociator(cfg)
        candidates = [{"way_id": "101", "bbox": [10.0, 10.0, 20.0, 20.0]}]
        low = [{"detector_score": 0.25, "box_xyxy": [11.0, 11.0, 21.0, 21.0]}]

        self.assertEqual(
            associator.match_low_detections("CAM_FRONT", low, candidates, set()),
            {},
        )

        track = associator.update_observed(
            "CAM_FRONT",
            "101",
            1,
            [10.0, 10.0, 20.0, 20.0],
            [10.0, 10.0, 20.0, 20.0],
            "red-circle",
            "red-circle",
            0.9,
        )
        self.assertEqual(track.track_id, "tltrk-000001")
        matches = associator.match_low_detections("CAM_FRONT", low, candidates, set())
        self.assertEqual(set(matches), {0})
        self.assertEqual(matches[0].candidate_index, 0)
        self.assertEqual(matches[0].candidate["way_id"], "101")
        self.assertEqual(matches[0].association_source, "current_map_projection")

    def test_low_detection_can_match_previous_bbox_without_current_projection(self):
        cfg = TemporalTrackingConfig(enabled=True, low_score=0.2)
        associator = TemporalAssociator(cfg)
        associator.update_observed(
            "CAM_FRONT",
            "101",
            1,
            [10.0, 10.0, 20.0, 20.0],
            [10.0, 10.0, 20.0, 20.0],
            "red-circle",
            "red-circle",
            0.9,
        )

        matches = associator.match_low_detections(
            "CAM_FRONT",
            [{"detector_score": 0.25, "box_xyxy": [11.0, 11.0, 21.0, 21.0]}],
            [],
            set(),
        )

        self.assertEqual(set(matches), {0})
        self.assertIsNone(matches[0].candidate_index)
        self.assertEqual(matches[0].candidate["way_id"], "101")
        self.assertEqual(matches[0].association_source, "last_map_projection")

    def test_lost_track_propagates_until_ttl(self):
        cfg = TemporalTrackingConfig(enabled=True, max_lost_frames=2)
        associator = TemporalAssociator(cfg)
        associator.update_observed(
            "CAM_FRONT",
            "101",
            1,
            [10.0, 10.0, 20.0, 20.0],
            [10.0, 10.0, 20.0, 20.0],
            "red-circle",
            "red-circle",
            0.9,
        )
        candidates = {"101": {"way_id": "101", "bbox": [12.0, 10.0, 22.0, 20.0]}}

        self.assertEqual(len(associator.propagate_missing("CAM_FRONT", 2, candidates, set())), 1)
        self.assertEqual(len(associator.propagate_missing("CAM_FRONT", 3, candidates, set())), 1)
        self.assertEqual(associator.propagate_missing("CAM_FRONT", 4, candidates, set()), [])

    def test_reobserved_map_way_reuses_track_id_after_ttl(self):
        cfg = TemporalTrackingConfig(enabled=True, max_lost_frames=1)
        associator = TemporalAssociator(cfg)
        first = associator.update_observed(
            "CAM_FRONT",
            "101",
            1,
            [10.0, 10.0, 20.0, 20.0],
            [10.0, 10.0, 20.0, 20.0],
            "red-circle",
            "red-circle",
            0.9,
        )
        associator.propagate_missing("CAM_FRONT", 2, {}, set())
        associator.propagate_missing("CAM_FRONT", 3, {}, set())

        second = associator.update_observed(
            "CAM_FRONT",
            "101",
            4,
            [12.0, 10.0, 22.0, 20.0],
            [12.0, 10.0, 22.0, 20.0],
            "green-circle",
            "green-circle",
            0.8,
        )

        self.assertEqual(second.track_id, first.track_id)
        self.assertEqual(second.status, "tracked")
        self.assertEqual(second.lost_frames, 0)

    def test_update_returns_tracking_result(self):
        cfg = TemporalTrackingConfig(enabled=True, low_score=0.2, max_lost_frames=2)
        associator = TemporalAssociator(cfg)
        candidates = [{"way_id": "101", "bbox": [10.0, 10.0, 20.0, 20.0]}]
        high = [{"detector_score": 0.9, "box_xyxy": [10.0, 10.0, 20.0, 20.0], "state": "red-circle"}]

        first = associator.update("CAM_FRONT", 1, high, {0: 0}, [], candidates)

        self.assertEqual(len(first.observed_tracks), 1)
        self.assertEqual(first.observed_tracks[0].source_type, "auto")
        self.assertEqual(first.observed_candidate_states, {0: "red-circle"})
        self.assertEqual(first.propagated_tracks, [])

        low = [{"detector_score": 0.25, "box_xyxy": [11.0, 11.0, 21.0, 21.0], "state": "green-circle"}]
        second = associator.update("CAM_FRONT", 2, [], {}, low, candidates)

        self.assertEqual(len(second.observed_tracks), 1)
        self.assertEqual(second.observed_tracks[0].source_type, "tracked")
        self.assertEqual(second.observed_tracks[0].association_source, "current_map_projection")
        self.assertEqual(second.propagated_tracks, [])

        third = associator.update("CAM_FRONT", 3, [], {}, [], candidates)

        self.assertEqual(third.observed_tracks, [])
        self.assertEqual(len(third.propagated_tracks), 1)
        self.assertEqual(third.propagated_tracks[0].track.track_id, first.observed_tracks[0].track.track_id)

    def test_low_unknown_detection_keeps_previous_track_state(self):
        cfg = TemporalTrackingConfig(enabled=True, low_score=0.2)
        associator = TemporalAssociator(cfg)
        candidates = [{"way_id": "101", "bbox": [10.0, 10.0, 20.0, 20.0]}]
        high = [{"detector_score": 0.9, "box_xyxy": [10.0, 10.0, 20.0, 20.0], "state": "red-circle"}]
        low = [{"detector_score": 0.25, "box_xyxy": [11.0, 11.0, 21.0, 21.0], "state": "unknown"}]

        first = associator.update("CAM_FRONT", 1, high, {0: 0}, [], candidates)
        second = associator.update("CAM_FRONT", 2, [], {}, low, candidates)

        self.assertEqual(first.observed_tracks[0].track.last_state, "red-circle")
        self.assertEqual(second.observed_tracks[0].track.last_state, "red-circle")
        self.assertEqual(second.observed_tracks[0].source_type, "tracked")

    def test_high_unknown_detection_does_not_overwrite_previous_track_state(self):
        cfg = TemporalTrackingConfig(enabled=True)
        associator = TemporalAssociator(cfg)
        candidates = [{"way_id": "101", "bbox": [10.0, 10.0, 20.0, 20.0]}]
        high = [{"detector_score": 0.9, "box_xyxy": [10.0, 10.0, 20.0, 20.0], "state": "red-circle"}]
        high_unknown = [{"detector_score": 0.8, "box_xyxy": [11.0, 10.0, 21.0, 20.0], "state": "unknown"}]

        first = associator.update("CAM_FRONT", 1, high, {0: 0}, [], candidates)
        second = associator.update("CAM_FRONT", 2, high_unknown, {0: 0}, [], candidates)

        self.assertEqual(first.observed_tracks[0].track.last_state, "red-circle")
        self.assertEqual(second.observed_tracks[0].track.last_state, "red-circle")


if __name__ == "__main__":
    unittest.main()
