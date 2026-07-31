"""Detection-to-map-projection association (REFACTOR_PLAN.md phase 5).

Extracted from match_traffic_lights.py.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def center(box):
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])


def _solve_assignment(cost, n_cand, unmatch_cost):
    rows, cols = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(rows, cols) if j < n_cand and cost[i, j] < unmatch_cost}


def match_boxes_legacy(detections, candidates, gate_factor=1.5, min_iou=0.05, unmatch_cost=1.5):
    """One-to-one Hungarian matching. Returns {det_index: cand_index}.

    Costs: overlapping pairs cost 1 - IoU (0..1); non-overlapping pairs within
    the center-distance gate cost 1 + dist/gate (1..2). Dummy "unmatched"
    columns cost `unmatch_cost`, so a poor pairing is only chosen when it beats
    leaving both boxes unmatched -- this stops the assignment from maximizing
    match count at the expense of quality.
    """
    if not detections or not candidates:
        return {}, {}
    big = 1e6
    n_det, n_cand = len(detections), len(candidates)
    cost = np.full((n_det, n_cand + n_det), big)
    cost[:, n_cand:] = unmatch_cost
    for i, det in enumerate(detections):
        dbox = det["box_xyxy"]
        ddiag = np.hypot(dbox[2] - dbox[0], dbox[3] - dbox[1])
        for j, cand in enumerate(candidates):
            cbox = cand["bbox"]
            cdiag = np.hypot(cbox[2] - cbox[0], cbox[3] - cbox[1])
            overlap = iou(dbox, cbox)
            dist = float(np.linalg.norm(center(dbox) - center(cbox)))
            gate = gate_factor * max(cdiag, ddiag)
            size_ratio = max(ddiag, cdiag) / max(min(ddiag, cdiag), 1e-6)
            if overlap >= min_iou:
                cost[i, j] = 1.0 - overlap
            elif dist <= gate and size_ratio <= 2.5:
                # offset fallback: sizes must agree, since projection error
                # shifts boxes but barely changes their scale
                cost[i, j] = 1.0 + dist / max(gate, 1e-6)
    matches = _solve_assignment(cost, n_cand, unmatch_cost)
    return matches, {i: "legacy" for i in matches}


def _build_cost(detections, candidates, det_indices, cand_indices, rules, unmatch_cost):
    big = 1e6
    cost = np.full((len(det_indices), len(cand_indices) + len(det_indices)), big)
    cost[:, len(cand_indices):] = unmatch_cost
    stages = {}
    for ri, det_i in enumerate(det_indices):
        dbox = detections[det_i]["box_xyxy"]
        ddiag = np.hypot(dbox[2] - dbox[0], dbox[3] - dbox[1])
        for cj, cand_j in enumerate(cand_indices):
            cbox = candidates[cand_j]["bbox"]
            cdiag = np.hypot(cbox[2] - cbox[0], cbox[3] - cbox[1])
            overlap = iou(dbox, cbox)
            dist = float(np.linalg.norm(center(dbox) - center(cbox)))
            size_ratio = max(ddiag, cdiag) / max(min(ddiag, cdiag), 1e-6)
            for rule in rules:
                if rule["kind"] == "iou":
                    if overlap < rule["min_iou"]:
                        continue
                    cost[ri, cj] = 1.0 - overlap
                    stages[(ri, cj)] = rule["stage"]
                    break
                if rule["kind"] == "distance":
                    gate = rule["gate_factor"] * max(cdiag, ddiag)
                    if dist > gate or size_ratio > rule["max_size_ratio"]:
                        continue
                    cost[ri, cj] = 1.0 + dist / max(gate, 1e-6)
                    stages[(ri, cj)] = rule["stage"]
                    break
    return cost, stages


def match_boxes_staged(detections, candidates, gate_factor=1.5, strict_min_iou=0.05,
                       relaxed_min_iou=0.01, relaxed_size_ratio=8.0,
                       relaxed_unmatch_cost=2.25):
    """Two-pass one-to-one matching.

    Pass 1 uses only IoU, so high-quality overlaps are claimed first. Pass 2
    runs on the remaining detections/candidates and allows either a weaker IoU
    or a center-distance gate with a looser size-ratio bound. This is intended
    for map QA / low-quality map geometry cases where projection boxes are
    visibly near the detection but too small or offset for the legacy cost.
    """
    if not detections or not candidates:
        return {}, {}

    det_indices = list(range(len(detections)))
    cand_indices = list(range(len(candidates)))
    strict_cost, strict_stages = _build_cost(
        detections,
        candidates,
        det_indices,
        cand_indices,
        [{"kind": "iou", "min_iou": strict_min_iou, "stage": "strict_iou"}],
        unmatch_cost=1.5,
    )
    strict_local = _solve_assignment(strict_cost, len(cand_indices), 1.5)
    matches = {det_indices[i]: cand_indices[j] for i, j in strict_local.items()}
    stages = {
        det_indices[i]: strict_stages.get((i, j), "strict_iou")
        for i, j in strict_local.items()
    }

    rem_det = [i for i in det_indices if i not in matches]
    used_cands = set(matches.values())
    rem_cand = [j for j in cand_indices if j not in used_cands]
    if not rem_det or not rem_cand:
        return matches, stages

    relaxed_cost, relaxed_stages = _build_cost(
        detections,
        candidates,
        rem_det,
        rem_cand,
        [
            {"kind": "iou", "min_iou": relaxed_min_iou, "stage": "relaxed_iou"},
            {
                "kind": "distance",
                "gate_factor": gate_factor,
                "max_size_ratio": relaxed_size_ratio,
                "stage": "relaxed_distance",
            },
        ],
        unmatch_cost=relaxed_unmatch_cost,
    )
    relaxed_local = _solve_assignment(relaxed_cost, len(rem_cand), relaxed_unmatch_cost)
    for i, j in relaxed_local.items():
        det_i = rem_det[i]
        cand_j = rem_cand[j]
        matches[det_i] = cand_j
        stages[det_i] = relaxed_stages.get((i, j), "relaxed")
    return matches, stages


def match_boxes(detections, candidates, mode="legacy", gate_factor=1.5,
                strict_min_iou=0.05, relaxed_min_iou=0.01,
                relaxed_size_ratio=8.0, relaxed_unmatch_cost=2.25):
    if mode == "legacy":
        return match_boxes_legacy(detections, candidates, gate_factor=gate_factor,
                                  min_iou=strict_min_iou)
    return match_boxes_staged(
        detections,
        candidates,
        gate_factor=gate_factor,
        strict_min_iou=strict_min_iou,
        relaxed_min_iou=relaxed_min_iou,
        relaxed_size_ratio=relaxed_size_ratio,
        relaxed_unmatch_cost=relaxed_unmatch_cost,
    )


def unmatched_reason(det, det_state, candidates, matches, gate_factor=1.5):
    """Classify why a detection stayed unmatched (review triage; report-only).
    Categories, checked in order:
      state_unknown_backside  — classifier saw no lamps (likely back/side view)
      no_map_candidate_in_view — no projected map TL in this frame at all
      candidate_taken         — nearest candidate was assigned to another detection
      beyond_gate             — nearest candidate exists but is too far
      geometry_mismatch       — within gate yet rejected (size ratio / cost)
    """
    if det_state == "unknown":
        return "state_unknown_backside"
    if not candidates:
        return "no_map_candidate_in_view"
    dbox = det["box_xyxy"]
    ddiag = np.hypot(dbox[2] - dbox[0], dbox[3] - dbox[1])
    dists = [float(np.linalg.norm(center(dbox) - center(c["bbox"]))) for c in candidates]
    j = int(np.argmin(dists))
    if j in set(matches.values()):
        return "candidate_taken"
    cbox = candidates[j]["bbox"]
    cdiag = np.hypot(cbox[2] - cbox[0], cbox[3] - cbox[1])
    if dists[j] > gate_factor * max(cdiag, ddiag):
        return "beyond_gate"
    return "geometry_mismatch"
