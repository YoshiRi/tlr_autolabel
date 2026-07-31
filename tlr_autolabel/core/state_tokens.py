"""Canonical signal-state token parsing, shared by the enrichment tools.

Canonical tokens (tlr_autolabel/v1 state spec, see README):
    {color}-{shape}[-{direction}]   e.g. green-arrow-up, red-circle, red-ped
    colors: green | amber | red
Legacy tokens (pre-v1 sidecars): green-arrow(up_left), yellow-circle style.
Both parse into the same element dicts; colors normalize to canonical (amber).

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

MAP_BULB_COLOR = {"amber": "yellow"}  # canonical -> lanelet2 bulb color tag


def parse_state(state: str) -> list[dict]:
    """Parse a state string (canonical or legacy) into element dicts.
    'unknown' and unparsable tokens carry no state and are dropped."""
    elements, seen = [], set()
    for token in filter(None, (t.strip() for t in (state or "").split(","))):
        m = CANON_RE.match(token)
        if m:
            color = m.group("color")
        else:
            m = LEGACY_RE.match(token)
            if not m:
                continue
            color = "amber" if m.group("color") == "yellow" else m.group("color")
        key = (color, m.group("shape"), m.group("arrow"))
        if key in seen:
            continue
        seen.add(key)
        elements.append({"color": color, "shape": m.group("shape"),
                         "arrow": m.group("arrow")})
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
