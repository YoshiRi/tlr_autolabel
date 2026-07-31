#!/usr/bin/env python3
"""Small temporal associator for traffic-light detections.

This module intentionally knows nothing about ONNX, image loading, T4 files, or
lanelet2 parsing. Callers provide per-frame detections and already-projected map
candidates; the associator maintains map-way keyed tracks and decides whether
low-confidence detections can update an existing track.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from typing import Any


BBox = list[float]


@dataclass(frozen=True)
class TemporalTrackingConfig:
    enabled: bool = False
    low_score: float = 0.2
    max_lost_frames: int = 3
    min_iou: float = 0.01
    center_gate_factor: float = 2.0
    projection_gate_factor: float = 2.0
    max_size_ratio: float = 8.0
    propagate: bool = True
    propagate_state: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "TemporalTrackingConfig":
        cfg = cls()
        if not data:
            return cfg
        aliases = {
            "max-lost-frames": "max_lost_frames",
            "low-score": "low_score",
            "min-iou": "min_iou",
            "center-gate-factor": "center_gate_factor",
            "projection-gate-factor": "projection_gate_factor",
            "max-size-ratio": "max_size_ratio",
            "propagate-state": "propagate_state",
        }
        values = {}
        for key, value in data.items():
            dest = aliases.get(key, key.replace("-", "_"))
            if hasattr(cfg, dest):
                values[dest] = value
        return replace(cfg, **values)


@dataclass
class TemporalTrack:
    track_id: str
    channel: str
    map_traffic_light_id: str
    last_bbox: BBox
    last_projected_bbox: BBox | None
    last_state: str
    last_raw_state: str
    last_detector_score: float | None
    last_observed_frame: int
    lost_frames: int = 0
    status: str = "tracked"


@dataclass
class PropagatedTrack:
    track: TemporalTrack
    bbox: BBox
    candidate: dict[str, Any]


@dataclass(frozen=True)
class LowDetectionMatch:
    det_index: int
    track: TemporalTrack
    candidate: dict[str, Any]
    candidate_index: int | None
    association_source: str


@dataclass(frozen=True)
class ObservedTrack:
    det_index: int
    track: TemporalTrack
    candidate: dict[str, Any]
    candidate_index: int | None
    source_type: str
    association_source: str


@dataclass(frozen=True)
class TrackingResult:
    observed_tracks: list[ObservedTrack]
    propagated_tracks: list[PropagatedTrack]
    terminated_tracks: list[TemporalTrack]
    low_matches: dict[int, LowDetectionMatch]
    updated_way_ids: set[str]
    observed_candidate_states: dict[int, str]


def iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 0.0)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 0.0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def diagonal(box: list[float]) -> float:
    return hypot(box[2] - box[0], box[3] - box[1])


def center_distance(a: list[float], b: list[float]) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return hypot(ax - bx, ay - by)


def diag_size_ratio(a: list[float], b: list[float]) -> float:
    da = diagonal(a)
    db = diagonal(b)
    return max(da, db) / max(min(da, db), 1e-6)


def min_side(box: list[float]) -> float:
    return min(abs(box[2] - box[0]), abs(box[3] - box[1]))


def _candidate_cost(
    det_box: list[float],
    projected_box: list[float],
    track_box: list[float],
    cfg: TemporalTrackingConfig,
) -> float | None:
    if diag_size_ratio(det_box, projected_box) > cfg.max_size_ratio:
        return None
    if diag_size_ratio(det_box, track_box) > cfg.max_size_ratio:
        return None

    proj_iou = iou(det_box, projected_box)
    track_iou = iou(det_box, track_box)
    proj_diag = max(diagonal(det_box), diagonal(projected_box), 1e-6)
    track_diag = max(diagonal(det_box), diagonal(track_box), 1e-6)
    proj_dist = center_distance(det_box, projected_box)
    track_dist = center_distance(det_box, track_box)

    projection_ok = (
        proj_iou >= cfg.min_iou
        or proj_dist <= cfg.projection_gate_factor * proj_diag
    )
    track_ok = (
        track_iou >= cfg.min_iou
        or track_dist <= cfg.center_gate_factor * track_diag
    )
    if not (projection_ok or track_ok):
        return None

    overlap_cost = 1.0 - max(proj_iou, track_iou)
    distance_cost = min(
        proj_dist / max(cfg.projection_gate_factor * proj_diag, 1e-6),
        track_dist / max(cfg.center_gate_factor * track_diag, 1e-6),
    )
    return overlap_cost + min(distance_cost, 1.0)


class TemporalAssociator:
    def __init__(self, cfg: TemporalTrackingConfig):
        self.cfg = cfg
        self._tracks: dict[tuple[str, str], TemporalTrack] = {}
        self._next_track_id = 1

    def tracks_for_channel(self, channel: str) -> dict[str, TemporalTrack]:
        return {
            way_id: track
            for (track_channel, way_id), track in self._tracks.items()
            if track_channel == channel and track.status != "terminated"
        }

    def update_observed(
        self,
        channel: str,
        map_traffic_light_id: str,
        frame_number: int,
        bbox: list[float],
        projected_bbox: list[float] | None,
        state: str,
        raw_state: str,
        detector_score: float | None,
    ) -> TemporalTrack:
        key = (channel, map_traffic_light_id)
        track = self._tracks.get(key)
        if track is None:
            track = TemporalTrack(
                track_id=f"tltrk-{self._next_track_id:06d}",
                channel=channel,
                map_traffic_light_id=map_traffic_light_id,
                last_bbox=[float(v) for v in bbox],
                last_projected_bbox=(
                    [float(v) for v in projected_bbox] if projected_bbox else None
                ),
                last_state=state or "unknown",
                last_raw_state=raw_state or "",
                last_detector_score=detector_score,
                last_observed_frame=frame_number,
            )
            self._next_track_id += 1
        else:
            next_state = state or "unknown"
            next_raw_state = raw_state or ""
            if next_state == "unknown" and track.last_state != "unknown":
                next_state = track.last_state
                next_raw_state = track.last_raw_state or next_raw_state
            track.last_bbox = [float(v) for v in bbox]
            track.last_projected_bbox = (
                [float(v) for v in projected_bbox] if projected_bbox else None
            )
            track.last_state = next_state
            track.last_raw_state = next_raw_state
            track.last_detector_score = detector_score
            track.last_observed_frame = frame_number
            track.lost_frames = 0
            track.status = "tracked"
        self._tracks[key] = track
        return track

    def match_low_detections(
        self,
        channel: str,
        detections: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        updated_way_ids: set[str],
    ) -> dict[int, LowDetectionMatch]:
        tracks = self.tracks_for_channel(channel)
        current_candidates = {cand["way_id"]: (cand_j, cand) for cand_j, cand in enumerate(candidates)}
        proposals: list[tuple[float, float, int, str, LowDetectionMatch]] = []
        for det_i, det in enumerate(detections):
            score = float(det.get("detector_score") or 0.0)
            if score < self.cfg.low_score:
                continue
            det_box = det["box_xyxy"]
            for way_id, track in tracks.items():
                if way_id in updated_way_ids:
                    continue
                cand_j, cand, association_source = self._candidate_for_track(
                    way_id, track, current_candidates
                )
                cost = _candidate_cost(det_box, cand["bbox"], track.last_bbox, self.cfg)
                if cost is not None:
                    proposals.append((
                        cost, -score, det_i, way_id,
                        LowDetectionMatch(
                            det_index=det_i,
                            track=track,
                            candidate=cand,
                            candidate_index=cand_j,
                            association_source=association_source,
                        ),
                    ))

        matches: dict[int, LowDetectionMatch] = {}
        used_dets: set[int] = set()
        used_way_ids: set[str] = set()
        for _cost, _score, det_i, way_id, match in sorted(
                proposals, key=lambda row: (row[0], row[1], row[2], row[3])):
            if det_i in used_dets or way_id in used_way_ids:
                continue
            matches[det_i] = match
            used_dets.add(det_i)
            used_way_ids.add(way_id)
        return matches

    def update(
        self,
        channel: str,
        frame_number: int,
        high_detections: list[dict[str, Any]],
        high_matches: dict[int, int],
        low_detections: list[dict[str, Any]],
        map_projections: list[dict[str, Any]],
        *,
        state_fn=None,
        raw_state_fn=None,
    ) -> TrackingResult:
        """Update tracks for one frame.

        High detections are already associated to current map projections by the
        caller's frame-level matcher. Low detections are matched here only
        against existing map-way tracks. `state_fn` and `raw_state_fn` keep this
        module independent of the traffic-light state vocabulary parser.
        """
        state_of = state_fn or (lambda det: det.get("state") or "unknown")
        raw_state_of = raw_state_fn or (lambda det: det.get("state") or "unknown")
        observed_tracks: list[ObservedTrack] = []
        updated_way_ids: set[str] = set()
        observed_candidate_states: dict[int, str] = {}

        for det_i, cand_i in high_matches.items():
            det = high_detections[det_i]
            cand = map_projections[cand_i]
            state = state_of(det)
            track = self.update_observed(
                channel,
                cand["way_id"],
                frame_number,
                det["box_xyxy"],
                cand["bbox"],
                state,
                raw_state_of(det),
                det.get("detector_score"),
            )
            updated_way_ids.add(cand["way_id"])
            observed_candidate_states[cand_i] = state
            observed_tracks.append(
                ObservedTrack(
                    det_index=det_i,
                    track=track,
                    candidate=cand,
                    candidate_index=cand_i,
                    source_type="auto",
                    association_source="high",
                )
            )

        low_matches = self.match_low_detections(
            channel, low_detections, map_projections, updated_way_ids
        )
        for det_i, low_match in low_matches.items():
            det = low_detections[det_i]
            cand = low_match.candidate
            projected_bbox = (
                cand["bbox"] if low_match.association_source != "last_bbox"
                else low_match.track.last_projected_bbox
            )
            state = state_of(det)
            raw_state = raw_state_of(det)
            if state == "unknown" and low_match.track.last_state != "unknown":
                state = low_match.track.last_state
                raw_state = low_match.track.last_raw_state or raw_state
            track = self.update_observed(
                channel,
                cand["way_id"],
                frame_number,
                det["box_xyxy"],
                projected_bbox,
                state,
                raw_state,
                det.get("detector_score"),
            )
            updated_way_ids.add(cand["way_id"])
            if low_match.candidate_index is not None:
                observed_candidate_states[low_match.candidate_index] = state
            observed_tracks.append(
                ObservedTrack(
                    det_index=det_i,
                    track=track,
                    candidate=cand,
                    candidate_index=low_match.candidate_index,
                    source_type="tracked",
                    association_source=low_match.association_source,
                )
            )

        candidates_by_way = {c["way_id"]: c for c in map_projections}
        propagated, terminated = self._propagate_missing_with_terminated(
            channel, frame_number, candidates_by_way, updated_way_ids
        )
        return TrackingResult(
            observed_tracks=observed_tracks,
            propagated_tracks=propagated,
            terminated_tracks=terminated,
            low_matches=low_matches,
            updated_way_ids=updated_way_ids,
            observed_candidate_states=observed_candidate_states,
        )

    def _candidate_for_track(
        self,
        way_id: str,
        track: TemporalTrack,
        current_candidates: dict[str, tuple[int, dict[str, Any]]],
    ) -> tuple[int | None, dict[str, Any], str]:
        if way_id in current_candidates:
            cand_j, cand = current_candidates[way_id]
            return cand_j, cand, "current_map_projection"
        if track.last_projected_bbox is not None:
            return None, {
                "way_id": way_id,
                "subtype": "",
                "bbox": track.last_projected_bbox,
                "distance_m": None,
                "facing": "",
                "facing_deg": None,
                "proj_min_side_px": round(min_side(track.last_projected_bbox), 1),
            }, "last_map_projection"
        return None, {
            "way_id": way_id,
            "subtype": "",
            "bbox": track.last_bbox,
            "distance_m": None,
            "facing": "",
            "facing_deg": None,
            "proj_min_side_px": round(min_side(track.last_bbox), 1),
        }, "last_bbox"

    def propagate_missing(
        self,
        channel: str,
        frame_number: int,
        candidates_by_way: dict[str, dict[str, Any]],
        updated_way_ids: set[str],
    ) -> list[PropagatedTrack]:
        propagated, _terminated = self._propagate_missing_with_terminated(
            channel, frame_number, candidates_by_way, updated_way_ids
        )
        return propagated

    def _propagate_missing_with_terminated(
        self,
        channel: str,
        frame_number: int,
        candidates_by_way: dict[str, dict[str, Any]],
        updated_way_ids: set[str],
    ) -> tuple[list[PropagatedTrack], list[TemporalTrack]]:
        propagated = []
        terminated = []
        for way_id, track in list(self.tracks_for_channel(channel).items()):
            if way_id in updated_way_ids:
                continue
            track.lost_frames += 1
            if track.lost_frames > self.cfg.max_lost_frames:
                track.status = "terminated"
                terminated.append(track)
                continue
            track.status = "lost"
            cand = candidates_by_way.get(way_id)
            if self.cfg.propagate and cand is not None:
                propagated.append(
                    PropagatedTrack(
                        track=track,
                        bbox=[float(v) for v in cand["bbox"]],
                        candidate=cand,
                    )
                )
        return propagated, terminated
