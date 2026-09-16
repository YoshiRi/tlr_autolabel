"""Where one lamp sits inside a projected traffic-light housing.

The map projects the whole housing (the lanelet2 `traffic_light` way is the
housing's bottom edge extruded by its `height` tag), and a housing-level
detector boxes the same thing, so IoU compares like with like. A lamp-level
detector (CoMLOps, CoMET `--only-tlr`) boxes only the lit lamp, which is ~1/4
of the housing diagonal and sits off-centre by up to a third of it. IoU against
the housing projection then collapses even when the association is correct
(measured on 668 frames: median matched-pair IoU 0.108 lamp-level vs 0.338
housing-level, 21.9% vs 52.5% matched).

This module narrows the projection to the sub-box where the detected lamp is
expected, so a lamp-level detection can be matched with the same cost function.

Slot layout: lamps are evenly spaced along the housing's long axis, so slot i
of n spans [i/n, (i+1)/n] of that axis and the full extent across it. The
projected box's own aspect decides which axis is long, which is why both
horizontal and vertical (snow-region) vehicle housings work with no extra map
tag. Order along the axis follows the JP convention:

    horizontal vehicle : green, amber, red  (left to right)
    vertical vehicle   : red, amber, green  (top to bottom)
    pedestrian         : red, green         (top to bottom)

Verified against the same run's housing-level L1 (1287 detections, unambiguous
1-detection/1-candidate frames): green-circle sat at -1/3 of the housing width
and red-circle at +1/3 (predicted -1/3, +1/3), red-ped at -0.21 of the height
and green-ped at +0.27 (predicted -1/4, +1/4).
"""
from __future__ import annotations

# lanelet2 subtype -> lamp colors in slot order along the long axis
SUBTYPE_LAMP_ORDER = {
    ("red_yellow_green", "horizontal"): ("green", "amber", "red"),
    ("red_yellow_green", "vertical"): ("red", "amber", "green"),
    ("red_green", "horizontal"): ("red", "green"),
    ("red_green", "vertical"): ("red", "green"),
}

# Shapes that live inside the housing box. An arrow panel is mounted below the
# housing and is not covered by the map way, so an arrow element cannot be
# placed in a slot -- callers fall back to the full projection for those.
IN_HOUSING_SHAPES = {"circle", "ped"}


def housing_axis(bbox) -> str:
    """Which way the housing's lamps are laid out, from the projected box."""
    return "horizontal" if (bbox[2] - bbox[0]) >= (bbox[3] - bbox[1]) else "vertical"


def lamp_slots(subtype: str, bbox) -> tuple[str, ...]:
    """Lamp colors in slot order for this housing, () when subtype is unknown."""
    return SUBTYPE_LAMP_ORDER.get((subtype or "", housing_axis(bbox)), ())


def _slot_box(bbox, axis: str, indices: set[int], n: int) -> list[float]:
    """The sub-box of `bbox` spanning the given slot indices along `axis`."""
    lo, hi = min(indices), max(indices)
    if axis == "horizontal":
        span = bbox[2] - bbox[0]
        return [bbox[0] + span * lo / n, bbox[1],
                bbox[0] + span * (hi + 1) / n, bbox[3]]
    span = bbox[3] - bbox[1]
    return [bbox[0], bbox[1] + span * lo / n,
            bbox[2], bbox[1] + span * (hi + 1) / n]


def expected_lamp_box(candidate, elements) -> list[float] | None:
    """Sub-box of a map candidate's projection where `elements` should appear.

    Returns None when the lamp cannot be placed -- unknown subtype, no lamp
    element that lives inside the housing, or a color the housing does not
    have. The caller then falls back to the full projected box, which is the
    pre-existing behavior.
    """
    bbox = candidate["bbox"]
    slots = lamp_slots(candidate.get("subtype", ""), bbox)
    if not slots:
        return None
    indices = {slots.index(e["color"]) for e in elements
               if e.get("shape") in IN_HOUSING_SHAPES and e.get("color") in slots}
    if not indices:
        return None
    return _slot_box(bbox, housing_axis(bbox), indices, len(slots))
