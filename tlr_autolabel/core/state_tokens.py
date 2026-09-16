"""Canonical signal-state token parsing, shared by the enrichment tools.

Canonical tokens (tlr_autolabel/v1 state spec, see README):
    {color}-{shape}[-{direction}]   e.g. green-arrow-up, red-circle, red-ped
    colors: green | amber | red
Legacy tokens (pre-v1 sidecars): green-arrow(up_left), yellow-circle style.
CoMLOps tokens (lamp-level detectors, e.g. CoMET --only-tlr writes these as
object_ann category names): underscore-joined, green_pedestrian style.
All three parse into the same element dicts; colors normalize to canonical
(amber), and `pedestrian` normalizes to the canonical `ped`.

The lanelet2 map's light_bulbs nodes use "yellow" — translate with
MAP_BULB_COLOR when comparing against the map, never inside our own data.
"""
import re

CANON_RE = re.compile(
    r"^(?P<color>green|amber|red)-(?P<shape>circle|arrow|u_turn|ped|number|cross)"
    r"(?:-(?P<arrow>up_right|up_left|down_right|down_left|up|down|left|right|unknown))?$")
LEGACY_RE = re.compile(
    r"^(?P<color>red|yellow|green)-(?P<shape>circle|ped|arrow)"
    r"(?:\((?P<arrow>[a-z_]+)\))?$")
# Underscore is both the separator and part of the direction (up_left), so the
# direction alternatives are matched longest-first and `shape` cannot swallow it.
COMLOPS_RE = re.compile(
    r"^(?P<color>red|yellow|amber|green)_"
    r"(?P<shape>circle|pedestrian|ped|arrow|number|cross|u_turn)"
    r"(?:_(?P<arrow>up_right|up_left|down_right|down_left|up|down|left|right|unknown))?$")

SHAPE_ALIASES = {"pedestrian": "ped"}

MAP_BULB_COLOR = {"amber": "yellow"}  # canonical -> lanelet2 bulb color tag


def parse_state(state: str) -> list[dict]:
    """Parse a state string (canonical, legacy or CoMLOps) into element dicts.
    'unknown' and unparsable tokens carry no state and are dropped."""
    elements, seen = [], set()
    for token in filter(None, (t.strip() for t in (state or "").split(","))):
        m = CANON_RE.match(token) or LEGACY_RE.match(token)
        is_comlops = False
        if not m:
            m = COMLOPS_RE.match(token)
            if not m:
                continue
            is_comlops = True
        color = m.group("color")
        color = "amber" if color == "yellow" else color
        shape = SHAPE_ALIASES.get(m.group("shape"), m.group("shape"))
        arrow = m.group("arrow")
        # CoMLOps names its directionless arrow class plain `arrow`; spell that
        # `unknown` so it round-trips through elements_key(). Canonical and
        # legacy tokens keep arrow=None -- existing sidecars depend on
        # `green-arrow` staying `green-arrow`.
        if is_comlops and shape == "arrow" and not arrow:
            arrow = "unknown"
        key = (color, shape, arrow)
        if key in seen:
            continue
        seen.add(key)
        elements.append({"color": color, "shape": shape, "arrow": arrow})
    return elements


def elements_key(elements: list[dict]) -> str:
    """Canonical order-independent state string; '' when no elements."""
    parts = []
    for e in elements:
        p = f"{e['color']}-{e['shape']}"
        if e.get("arrow"):
            p += f"-{e['arrow']}"
        parts.append(p)
    return ",".join(sorted(parts))


def bulb_color(canonical_color: str) -> str:
    """Canonical color as the lanelet2 light_bulbs color tag."""
    return MAP_BULB_COLOR.get(canonical_color, canonical_color)
