"""Compensate the map projection's per-frame 2D offset before association.

The projected housing box does not land on the detected one. Measured on three
recordings sharing one map, the horizontal offset's median was -0.07 / -0.44 /
-0.28 of the box width -- it varies by recording *and* by frame, so it is not a
calibration constant that could be baked into the extrinsic. The root cause is
under separate investigation; this module does not explain it, it removes it.

What it is, empirically, is almost entirely a **translation**. On the 167 GT
frames of `2821adc7`, subtracting one (dx, dy) per frame moved the association's
IoU median from 0.362 to 0.707 and the centre residual from 32.0 px to 3.7 px.
A residual of 3.7 px on boxes whose median width is 44 px leaves nothing worth
modelling with more parameters.

The shift has to be estimated without GT, from the detections themselves, and
the one thing that makes that work is *which* detections are allowed to be
anchors:

  * a detection whose state was read is looking at a lamp, so it is the front
    of a housing that the map also places near it;
  * a detection with no readable state is very often the *back* of a housing
    (202 of 236 unmatched detections on that run, confirmed by eye), whose
    nearest map way is the front face of some other signal entirely. Those
    anchors drag the per-frame median off by up to 218 px.

Filtering anchors to stated detections alone reaches the GT-derived ceiling
(IoU median 0.707, residual 6.0 px) with no map facing test and no extra pass.
Temporal smoothing was measured and made it worse (residual 6.0 -> 11.5 px):
the offset really does move frame to frame, so a per-frame estimate beats a
smoothed one.
"""
from __future__ import annotations

import statistics as st

SHIFT_KEY = "projection_shift"
UNSHIFTED_KEY = "bbox_unshifted"


def _wh(box):
    return box[2] - box[0], box[3] - box[1]


def _centre(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _pairs(anchors, candidates, gate_factor, size_ratio):
    """Greedy nearest-centre anchor->candidate pairs, largest anchor first.

    Deliberately loose: this association only has to be right often enough for
    a median to be robust, and it runs *before* the offset is known, so the
    gate has to admit the offset itself.
    """
    used = set()
    out = []
    for box in sorted(anchors, key=lambda b: -_wh(b)[0]):
        bw, _ = _wh(box)
        if bw <= 0:
            continue
        bx, by = _centre(box)
        best, best_dist = None, None
        for index, cand in enumerate(candidates):
            if index in used:
                continue
            cw, _ = _wh(cand["bbox"])
            if cw <= 0 or max(bw / cw, cw / bw) > size_ratio:
                continue
            cx, cy = _centre(cand["bbox"])
            dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
            if dist <= gate_factor * max(bw, cw) and (best_dist is None or dist < best_dist):
                best, best_dist = index, dist
        if best is not None:
            used.add(best)
            cx, cy = _centre(candidates[best]["bbox"])
            out.append((bx - cx, by - cy))
    return out


def estimate_frame_shift(anchors, candidates, *, gate_factor: float = 6.0,
                         size_ratio: float = 4.0, min_pairs: int = 1):
    """(dx, dy, n_pairs) to add to every projected box, or None.

    None means this frame had nothing to register against -- the caller should
    carry the previous frame's shift rather than resetting to zero, since the
    offset is continuous in ego motion while anchor availability is not.
    """
    if not anchors or not candidates:
        return None
    deltas = _pairs(anchors, candidates, gate_factor, size_ratio)
    if len(deltas) < max(1, min_pairs):
        return None
    return (st.median(d[0] for d in deltas),
            st.median(d[1] for d in deltas),
            len(deltas))


def apply_shift(candidates, dx: float, dy: float):
    """Candidates with the shift added, keeping the raw projection.

    The unshifted box is preserved under `bbox_unshifted` so a reviewer can see
    where the map actually put it, and so the shift stays auditable rather than
    silently rewriting the geometry.
    """
    if not dx and not dy:
        return candidates
    out = []
    for cand in candidates:
        box = cand["bbox"]
        moved = dict(cand)
        moved[UNSHIFTED_KEY] = list(box)
        moved["bbox"] = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]
        moved[SHIFT_KEY] = [round(dx, 2), round(dy, 2)]
        out.append(moved)
    return out
