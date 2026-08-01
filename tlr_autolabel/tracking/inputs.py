"""Helpers for adapting Tier A detections into temporal-tracking inputs."""

from __future__ import annotations


def detector_score(signal_entry: dict, default: float = 1.0) -> float:
    score = signal_entry.get("detector_score", default)
    if score is None or score == "":
        return default
    return float(score)


def box_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = map(float, a)
    bx0, by0, bx1, by1 = map(float, b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def collect_low_tracking_candidates(
    payload: dict,
    high_threshold: float,
    low_threshold: float,
) -> list[dict]:
    """Return low-confidence candidates without duplicating high detections.

    Newer L1 JSONs may carry raw_detections down to --det-low-score-thr. Older
    runs only have signals; those can still serve as low candidates when L1 was
    run below match_traffic_lights.py --min-score.
    """
    low = []
    high_boxes = []
    for det in payload.get("signals", []):
        score = detector_score(det)
        if score >= high_threshold:
            high_boxes.append(det["box_xyxy"])
        elif score >= low_threshold:
            low.append(det)

    for det in payload.get("raw_detections", []):
        score = detector_score(det)
        if score < low_threshold or score >= high_threshold:
            continue
        box = det["box_xyxy"]
        if any(box_iou(box, existing) >= 0.98 for existing in high_boxes):
            continue
        if any(box_iou(box, existing["box_xyxy"]) >= 0.98 for existing in low):
            continue
        low.append(det)
    return low
