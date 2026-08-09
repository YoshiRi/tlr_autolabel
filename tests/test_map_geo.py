"""Unit tests for MGRS local -> WGS84 conversion (tlr_autolabel/map/geo.py).

Reference latitudes/longitudes were produced with pyproj. pyproj is not a
declared dependency, so the expectations are baked in here; when it happens to
be installed the last test re-derives them to catch drift.
"""
import math
import unittest

from tlr_autolabel.map.geo import (
    BAND_LETTERS,
    MgrsGridError,
    band_latitudes,
    google_maps_url,
    google_streetview_url,
    local_to_latlon,
    parse_mgrs_grid,
    square_origin,
    utm_to_latlon,
)

# (grid, local_x, local_y, lat, lon) -- from pyproj
REFERENCE = [
    ("53SPA", 26585.4276, 29568.2646, 36.4029471, 136.4116871),
    ("53SPA", 0.0, 0.0, 36.1395604, 136.1114780),
    ("54SUE", 12345.0, 67890.0, 35.8372506, 138.9223023),
]
TOLERANCE_DEG = 1e-6  # ~0.1 m


class ParseMgrsGridTest(unittest.TestCase):
    def test_parses_a_standard_grid(self):
        self.assertEqual(parse_mgrs_grid("53SPA"), (53, "S", "P", "A"))

    def test_is_case_and_space_insensitive(self):
        self.assertEqual(parse_mgrs_grid(" 53spa "), (53, "S", "P", "A"))

    def test_rejects_missing_letters(self):
        for bad in ("53", "53S", "53SP", "", None):
            with self.assertRaises(MgrsGridError):
                parse_mgrs_grid(bad)

    def test_rejects_zone_out_of_range(self):
        with self.assertRaises(MgrsGridError):
            parse_mgrs_grid("61SPA")

    def test_rejects_column_not_valid_for_the_zone(self):
        # Zone 53 uses the JKLMNPQR column set, so 'A' cannot appear.
        with self.assertRaises(MgrsGridError):
            parse_mgrs_grid("53SAA")

    def test_rejects_letters_i_and_o(self):
        with self.assertRaises(MgrsGridError):
            parse_mgrs_grid("53SPI")

    def test_rejects_unknown_latitude_band(self):
        with self.assertRaises(MgrsGridError):
            parse_mgrs_grid("53APA")


class SquareOriginTest(unittest.TestCase):
    def test_column_letter_maps_to_a_100km_easting(self):
        easting, _ = square_origin(53, "P", "A")
        self.assertEqual(easting, 600000.0)

    def test_first_column_of_a_set_is_100km(self):
        self.assertEqual(square_origin(53, "J", "A")[0], 100000.0)

    def test_odd_zone_rows_start_at_a(self):
        self.assertEqual(square_origin(53, "P", "A")[1], 0.0)

    def test_even_zone_rows_are_shifted_by_five(self):
        # 'F' is the fifth row letter, so in an even zone it sits at zero.
        self.assertEqual(square_origin(54, "U", "F")[1], 0.0)

    def test_row_letters_advance_by_100km(self):
        base = square_origin(53, "P", "A")[1]
        self.assertEqual(square_origin(53, "P", "B")[1] - base, 100000.0)


class BandLatitudesTest(unittest.TestCase):
    def test_band_s_covers_32_to_40_north(self):
        self.assertEqual(band_latitudes("S"), (32.0, 40.0))

    def test_first_band_starts_at_minus_80(self):
        self.assertEqual(band_latitudes("C")[0], -80.0)

    def test_final_band_is_extended(self):
        self.assertEqual(band_latitudes("X")[1], 84.0)

    def test_every_band_letter_is_ordered(self):
        lows = [band_latitudes(b)[0] for b in BAND_LETTERS]
        self.assertEqual(lows, sorted(lows))


class UtmToLatLonTest(unittest.TestCase):
    def test_false_easting_on_the_equator_is_the_central_meridian(self):
        lat, lon = utm_to_latlon(500000.0, 0.0, 53)
        self.assertAlmostEqual(lat, 0.0, places=9)
        self.assertAlmostEqual(lon, 135.0, places=9)

    def test_southern_hemisphere_uses_the_false_northing(self):
        lat, _ = utm_to_latlon(500000.0, 10000000.0, 53, northern=False)
        self.assertAlmostEqual(lat, 0.0, places=7)

    def test_zone_central_meridian_follows_the_zone_number(self):
        _, lon = utm_to_latlon(500000.0, 0.0, 1)
        self.assertAlmostEqual(lon, -177.0, places=9)


class LocalToLatLonTest(unittest.TestCase):
    def test_matches_reference_coordinates(self):
        for grid, x, y, lat, lon in REFERENCE:
            with self.subTest(grid=grid, x=x, y=y):
                got_lat, got_lon = local_to_latlon(x, y, grid)
                self.assertAlmostEqual(got_lat, lat, delta=TOLERANCE_DEG)
                self.assertAlmostEqual(got_lon, lon, delta=TOLERANCE_DEG)

    def test_result_lands_inside_the_declared_latitude_band(self):
        low, high = band_latitudes("S")
        lat, _ = local_to_latlon(26585.0, 29568.0, "53SPA")
        self.assertTrue(low <= lat <= high)

    def test_moving_north_in_local_metres_increases_latitude(self):
        lat0, _ = local_to_latlon(0.0, 0.0, "53SPA")
        lat1, _ = local_to_latlon(0.0, 1000.0, "53SPA")
        self.assertGreater(lat1, lat0)

    def test_moving_east_in_local_metres_increases_longitude(self):
        _, lon0 = local_to_latlon(0.0, 0.0, "53SPA")
        _, lon1 = local_to_latlon(1000.0, 0.0, "53SPA")
        self.assertGreater(lon1, lon0)

    def test_one_kilometre_north_is_about_one_kilometre(self):
        lat0, _ = local_to_latlon(0.0, 0.0, "53SPA")
        lat1, _ = local_to_latlon(0.0, 1000.0, "53SPA")
        metres = (lat1 - lat0) * 111320.0
        self.assertAlmostEqual(metres, 1000.0, delta=5.0)

    def test_invalid_grid_propagates(self):
        with self.assertRaises(MgrsGridError):
            local_to_latlon(0.0, 0.0, "nonsense")


class UrlTest(unittest.TestCase):
    def test_maps_url_contains_the_coordinates_and_zoom(self):
        url = google_maps_url(36.4029471, 136.4116871)
        self.assertIn("36.4029471,136.4116871", url)
        self.assertTrue(url.endswith("19z"))

    def test_maps_url_zoom_is_overridable(self):
        self.assertTrue(google_maps_url(1.0, 2.0, zoom=12).endswith("12z"))

    def test_streetview_url_requests_a_panorama(self):
        url = google_streetview_url(36.4029471, 136.4116871)
        self.assertIn("map_action=pano", url)
        self.assertIn("viewpoint=36.4029471,136.4116871", url)


class PyprojAgreementTest(unittest.TestCase):
    """Re-derive the baked-in expectations when pyproj is available."""

    def test_agrees_with_pyproj(self):
        try:
            from pyproj import CRS, Transformer
        except ImportError:
            self.skipTest("pyproj not installed (not a declared dependency)")
        for grid, x, y, _, _ in REFERENCE:
            zone, _, column, row = parse_mgrs_grid(grid)
            easting, northing_base = square_origin(zone, column, row)
            lat, lon = local_to_latlon(x, y, grid)
            # Recover the 2000 km cycle our own conversion settled on.
            cycle = round(
                (_northing_for(lat, lon, zone) - northing_base - y) / 2_000_000.0
            )
            transformer = Transformer.from_crs(
                CRS.from_dict({"proj": "utm", "zone": zone, "datum": "WGS84"}),
                CRS.from_epsg(4326),
                always_xy=True,
            )
            ref_lon, ref_lat = transformer.transform(
                easting + x, northing_base + y + cycle * 2_000_000.0
            )
            self.assertAlmostEqual(lat, ref_lat, delta=1e-6)
            self.assertAlmostEqual(lon, ref_lon, delta=1e-6)


def _northing_for(lat: float, lon: float, zone: int) -> float:
    """Forward projection, only used to recover the northing cycle in tests."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    k0 = 0.9996
    phi = math.radians(lat)
    lam = math.radians(lon) - math.radians((zone - 1) * 6 - 180 + 3)
    n = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = e2 / (1 - e2) * math.cos(phi) ** 2
    aa = math.cos(phi) * lam
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * phi)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi)
        - (35 * e2**3 / 3072) * math.sin(6 * phi)
    )
    return k0 * (
        m
        + n * math.tan(phi) * (
            aa**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * e2 / (1 - e2)) * aa**6 / 720
        )
    )


if __name__ == "__main__":
    unittest.main()
