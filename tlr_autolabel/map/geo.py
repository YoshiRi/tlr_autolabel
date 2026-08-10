"""MGRS local map coordinates -> WGS84 latitude/longitude.

Autoware maps here are `projector_type: MGRS` with an `mgrs_grid` like
"53SPA", and node coordinates are metres inside that 100 km square. Nothing
in the pipeline needed geographic coordinates until the review views wanted
to answer "where on earth is this", so the conversion lives here rather than
pulling in pyproj: it is not in requirements.txt, and one inverse UTM
projection is not worth a new pinned dependency.

Accuracy is validated against pyproj in tests/test_map_geo.py.
"""

from __future__ import annotations

import math

# WGS84
A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)
K0 = 0.9996
FALSE_EASTING = 500000.0

# MGRS 100 km square identifiers. I and O are never used.
COLUMN_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")
ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"
ROW_CYCLE_M = 2_000_000.0
# Latitude bands, 8 degrees each starting at -80. X is 12 degrees but the
# extra height never matters for picking a northing cycle.
BAND_LETTERS = "CDEFGHJKLMNPQRSTUVWX"


class MgrsGridError(ValueError):
    """The mgrs_grid string is not something we can interpret."""


def parse_mgrs_grid(grid: str) -> tuple[int, str, str, str]:
    """"53SPA" -> (53, "S", "P", "A")."""
    text = (grid or "").strip().upper().replace(" ", "")
    digits = ""
    index = 0
    while index < len(text) and text[index].isdigit():
        digits += text[index]
        index += 1
    letters = text[index:]
    if not digits or len(letters) != 3:
        raise MgrsGridError(f"expected a grid like '53SPA', got {grid!r}")
    zone = int(digits)
    if not 1 <= zone <= 60:
        raise MgrsGridError(f"UTM zone out of range in {grid!r}")
    band, column, row = letters[0], letters[1], letters[2]
    if band not in BAND_LETTERS:
        raise MgrsGridError(f"unknown latitude band {band!r} in {grid!r}")
    if column not in COLUMN_SETS[(zone - 1) % 3]:
        raise MgrsGridError(f"column {column!r} is not valid for zone {zone}")
    if row not in ROW_LETTERS:
        raise MgrsGridError(f"unknown row letter {row!r} in {grid!r}")
    return zone, band, column, row


def square_origin(zone: int, column: str, row: str) -> tuple[float, float]:
    """South-west corner of the 100 km square, as (easting, northing).

    The northing is only known modulo 2000 km; `local_to_latlon` resolves the
    cycle using the latitude band.
    """
    easting = (COLUMN_SETS[(zone - 1) % 3].index(column) + 1) * 100_000.0
    row_index = ROW_LETTERS.index(row)
    # Even zones start their row lettering five places along.
    if zone % 2 == 0:
        row_index = (row_index - 5) % len(ROW_LETTERS)
    return easting, row_index * 100_000.0


def band_latitudes(band: str) -> tuple[float, float]:
    index = BAND_LETTERS.index(band)
    low = -80.0 + 8.0 * index
    return low, (84.0 if band == "X" else low + 8.0)


def utm_to_latlon(easting: float, northing: float, zone: int, northern: bool = True):
    """Inverse transverse Mercator (Karney-style series, ample for metres)."""
    x = easting - FALSE_EASTING
    y = northing if northern else northing - 10_000_000.0

    m = y / K0
    mu = m / (A * (1 - E2 / 4 - 3 * E2**2 / 64 - 5 * E2**3 / 256))
    e1 = (1 - math.sqrt(1 - E2)) / (1 + math.sqrt(1 - E2))
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )
    sin_phi1, cos_phi1, tan_phi1 = math.sin(phi1), math.cos(phi1), math.tan(phi1)
    ep2 = E2 / (1 - E2)
    c1 = ep2 * cos_phi1**2
    t1 = tan_phi1**2
    n1 = A / math.sqrt(1 - E2 * sin_phi1**2)
    r1 = A * (1 - E2) / (1 - E2 * sin_phi1**2) ** 1.5
    d = x / (n1 * K0)

    lat = phi1 - (n1 * tan_phi1 / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ep2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ep2 - 3 * c1**2) * d**6 / 720
    )
    lon = (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ep2 + 24 * t1**2) * d**5 / 120
    ) / cos_phi1
    central = math.radians((zone - 1) * 6 - 180 + 3)
    return math.degrees(lat), math.degrees(central + lon)


def local_to_latlon(x: float, y: float, grid: str) -> tuple[float, float]:
    """Map-frame metres inside `grid` -> (latitude, longitude) in WGS84."""
    zone, band, column, row = parse_mgrs_grid(grid)
    easting_base, northing_base = square_origin(zone, column, row)
    easting = easting_base + x
    low, high = band_latitudes(band)
    northern = low >= 0

    # The row letter repeats every 2000 km, so try each cycle and keep the one
    # that lands inside the declared latitude band.
    best = None
    for cycle in range(11):  # 0..20000 km covers both hemispheres
        northing = northing_base + y + cycle * ROW_CYCLE_M
        lat, lon = utm_to_latlon(easting, northing, zone, northern)
        if low <= lat <= high:
            return lat, lon
        # Fall back to whichever cycle came closest, so a slightly out-of-band
        # point still produces a usable location instead of an exception.
        error = min(abs(lat - low), abs(lat - high))
        if best is None or error < best[0]:
            best = (error, lat, lon)
    if best is None:
        raise MgrsGridError(f"could not place {grid!r} northing for y={y}")
    return best[1], best[2]


def google_maps_url(lat: float, lon: float, zoom: int = 19) -> str:
    return f"https://www.google.com/maps/@{lat:.7f},{lon:.7f},{zoom}z"


def google_streetview_url(lat: float, lon: float) -> str:
    return (
        "https://www.google.com/maps/@?api=1&map_action=pano"
        f"&viewpoint={lat:.7f},{lon:.7f}"
    )
