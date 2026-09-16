"""Normalize a `tlr_autolabel/v1` payload before L3 consumes it.

Autoware represents a traffic light as one housing carrying a list of
{color, shape, status} lamps, and the official T4 dataset follows that: one
box per housing, the state as its lamp list. `tlr_autolabel/v1` already has
that shape (`box_xyxy` + `lamps[]` + canonical `state`), and the native
pipeline fills it that way -- the detector finds the housing
(`det_classes: TRAFFIC_LIGHT`), the lamp recognizer only assigns state.

Lamp-level producers do not. CoMLOps-style detectors emit one detection per
lit lamp, with no housing box, so a naive adapter puts a lamp box in
`box_xyxy` and L3 silently compares a lamp against a housing projection.
Nothing in the schema said which one `box_xyxy` was, so the mismatch was
invisible.

This module makes it explicit and reconciles both flavors:

  * `box_level` -- "housing" (default, back-compatible) or "lamp", readable
    per signal or once per payload.
  * folding -- lamp-level signals that name their parent housing
    (`housing_id`, and/or carry `housing_box_xyxy`) are folded into one
    housing-level signal whose `lamps[]` is the union of theirs. Each lamp
    keeps its own box under `lamps[].box_xyxy`, so nothing is lost.
  * lamp-level signals with no housing information stay lamp-level and are
    labelled as such, so L3 can match them by expected lamp position instead
    of by IoU (see tlr_autolabel.map.lamp_geometry) rather than quietly
    scoring them near zero.

Read `signals()` instead of `payload["signals"]` anywhere L3 geometry is
involved.
"""
from __future__ import annotations

from collections import OrderedDict

from tlr_autolabel.core.state_tokens import elements_key, parse_state

BOX_LEVEL_HOUSING = "housing"
BOX_LEVEL_LAMP = "lamp"
BOX_LEVELS = (BOX_LEVEL_HOUSING, BOX_LEVEL_LAMP)


def state_score(signal: dict):
    """How confident the *state* is, as distinct from `detector_score`.

    Two different confidences arrive from a two-stage producer and answer
    different questions: `detector_score` says "there is a signal here"
    (CoMET's front-stage detector, threshold 0.1), while the lamp confidences
    say "and its state is this" (the TLR subnet, threshold 0.5). Folding them
    into one number hid a large class of rows: on one 2400-frame run, 11368 of
    18772 housings had no readable lamp yet carried a median detector_score of
    0.950, so they passed a 0.5 detector-score gate as confident detections
    while their state was pure `unknown`.

    Explicit `state_score` wins; otherwise it is the strongest lamp. None when
    no lamp was read -- which is the signal worth acting on, not a zero.
    """
    explicit = signal.get("state_score")
    if isinstance(explicit, (int, float)):
        return float(explicit)
    scores = [lamp.get("confidence") for lamp in (signal.get("lamps") or [])]
    scores = [float(s) for s in scores if isinstance(s, (int, float))]
    return max(scores) if scores else None


def box_level(signal: dict, payload: dict | None = None) -> str:
    """Resolved box semantics for one signal: per-signal, then payload, then
    the housing default that every pre-`box_level` sidecar implies."""
    for source in (signal, payload or {}):
        value = source.get("box_level")
        if value:
            if value not in BOX_LEVELS:
                raise ValueError(
                    f"unknown box_level {value!r}; expected one of {BOX_LEVELS}")
            return value
    return BOX_LEVEL_HOUSING


def _union(boxes: list[list[float]]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _lamp_elements(signal: dict) -> list[dict]:
    """Lamp elements of a signal, from `lamps[]` or parsed from `state`."""
    elements = []
    for lamp in signal.get("lamps") or []:
        if lamp.get("color") and lamp.get("shape"):
            elements.append({"color": lamp["color"], "shape": lamp["shape"],
                             "arrow": lamp.get("arrow")})
        elif lamp.get("label"):
            elements.extend(parse_state(lamp["label"]))
    if not elements:
        # a lamp-level producer may only name the class, e.g. CoMET's
        # object_ann category `red_circle`
        elements = parse_state(signal.get("state") or signal.get("source_category") or "")
    return elements


def _lamp_entry(signal: dict, element: dict) -> dict:
    """One lamps[] entry that keeps the lamp's own box and score."""
    entry = {
        "label": elements_key([element]),
        "color": element["color"],
        "shape": element["shape"],
        "arrow": element.get("arrow"),
        "confidence": signal.get("detector_score"),
    }
    if signal.get("box_xyxy"):
        entry["box_xyxy"] = [float(v) for v in signal["box_xyxy"]]
    return entry


def _housing_group_key(signal: dict):
    """What identifies the parent housing, or None when nothing does."""
    for key in ("housing_id", "housing_token", "parent_id"):
        if signal.get(key) not in (None, ""):
            return (key, signal[key])
    box = signal.get("housing_box_xyxy")
    if box:
        return ("housing_box_xyxy", tuple(round(float(v), 3) for v in box))
    return None


def _fold_group(members: list[dict]) -> dict:
    """One housing-level signal from the lamp-level detections inside it."""
    first = members[0]
    lamps, elements = [], []
    for member in members:
        for element in _lamp_elements(member):
            lamps.append(_lamp_entry(member, element))
            elements.append(element)

    housing_box = first.get("housing_box_xyxy")
    if housing_box:
        box = [float(v) for v in housing_box]
    else:
        # no housing box was reported: the lamps' extent is the best available
        # stand-in, and it is marked so the reviewer knows it is derived
        box = _union([m["box_xyxy"] for m in members if m.get("box_xyxy")])

    scores = [m["detector_score"] for m in members
              if isinstance(m.get("detector_score"), (int, float))]
    folded = {
        **{k: v for k, v in first.items()
           if k not in ("lamps", "state", "box_xyxy", "box_level",
                        "housing_box_xyxy", "detector_score")},
        "box_xyxy": box,
        "box_level": BOX_LEVEL_HOUSING,
        "lamps": lamps,
        "state": elements_key(elements) or "unknown",
        "detector_score": (first.get("housing_score") if
                           isinstance(first.get("housing_score"), (int, float))
                           else (max(scores) if scores else None)),
        # the housing's own score localizes it; the lamps say how sure the state is
        "state_score": max(scores) if scores else None,
        "folded_from": [m.get("signal_id") for m in members if m.get("signal_id")],
    }
    if not housing_box:
        folded["housing_box_source"] = "lamp_union"
    return folded


def signals(payload: dict) -> list[dict]:
    """Signals normalized to the housing-level contract where possible.

    Housing-level signals pass through with `box_level` made explicit. Groups
    of lamp-level signals that name a parent housing are folded into one.
    Lamp-level signals with no parent stay separate and keep
    `box_level="lamp"`.
    """
    out: list[dict] = []
    groups: OrderedDict = OrderedDict()
    for signal in payload.get("signals") or []:
        level = box_level(signal, payload)
        if level == BOX_LEVEL_HOUSING:
            out.append({**signal, "box_level": BOX_LEVEL_HOUSING})
            continue
        key = _housing_group_key(signal)
        if key is None:
            elements = _lamp_elements(signal)
            out.append({
                **signal,
                "box_level": BOX_LEVEL_LAMP,
                "lamps": signal.get("lamps") or [_lamp_entry(signal, e) for e in elements],
                "state": signal.get("state") or elements_key(elements) or "unknown",
            })
            continue
        groups.setdefault(key, []).append(signal)

    out.extend(_fold_group(members) for members in groups.values())
    return out


def count_by_level(normalized: list[dict]) -> dict[str, int]:
    """Per-level counts, for run diagnostics."""
    counts = {level: 0 for level in BOX_LEVELS}
    for signal in normalized:
        counts[signal.get("box_level", BOX_LEVEL_HOUSING)] += 1
    return counts
