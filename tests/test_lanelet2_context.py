"""Unit tests for lanelet2 road-context parsing
(tlr_autolabel/map/lanelet2.py).
"""
import tempfile
import unittest
from pathlib import Path

from tlr_autolabel.map.lanelet2 import load_lanelet2_context

OSM = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
  <node id="1"><tag k="local_x" v="0"/><tag k="local_y" v="0"/></node>
  <node id="2"><tag k="local_x" v="10"/><tag k="local_y" v="0"/></node>
  <node id="3"><tag k="local_x" v="0"/><tag k="local_y" v="4"/></node>
  <node id="4"><tag k="local_x" v="10"/><tag k="local_y" v="4"/></node>
  <node id="5"><tag k="local_x" v="20"/><tag k="local_y" v="20"/></node>
  <node id="6"><tag k="local_x" v="30"/><tag k="local_y" v="20"/></node>
  <node id="7"><tag k="local_x" v="30"/><tag k="local_y" v="30"/></node>
  <node id="9"><tag k="note" v="no local coords"/></node>

  <way id="100"><nd ref="1"/><nd ref="2"/><tag k="type" v="line_thin"/></way>
  <way id="101"><nd ref="3"/><nd ref="4"/><tag k="type" v="line_thin"/></way>
  <way id="102">
    <nd ref="5"/><nd ref="6"/><nd ref="7"/>
    <tag k="type" v="intersection_area"/>
  </way>
  <way id="103"><nd ref="1"/><nd ref="3"/><tag k="type" v="stop_line"/></way>
  <way id="104"><nd ref="5"/><nd ref="6"/><tag k="type" v="crosswalk_polygon"/></way>
  <way id="105"><nd ref="1"/><tag k="type" v="stop_line"/></way>
  <way id="106"><nd ref="9"/><nd ref="9"/><tag k="type" v="stop_line"/></way>

  <relation id="200">
    <member type="way" ref="100" role="right"/>
    <member type="way" ref="101" role="left"/>
    <tag k="type" v="lanelet"/><tag k="subtype" v="road"/>
  </relation>
  <relation id="201">
    <member type="way" ref="100" role="left"/>
    <tag k="type" v="lanelet"/><tag k="subtype" v="crosswalk"/>
  </relation>
  <relation id="202">
    <member type="way" ref="100" role="left"/>
    <member type="way" ref="101" role="right"/>
    <tag k="type" v="regulatory_element"/><tag k="subtype" v="traffic_light"/>
  </relation>
</osm>
"""


class LoadLanelet2ContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        path = Path(cls._tmp.name) / "map.osm"
        path.write_text(OSM)
        cls.lanelets, cls.ways = load_lanelet2_context(path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_reads_lanelet_with_both_bounds(self):
        ids = [lane["id"] for lane in self.lanelets]
        self.assertEqual(ids, ["200"])

    def test_lanelet_bounds_carry_projected_points(self):
        lane = self.lanelets[0]
        self.assertEqual(lane["left"], [(0.0, 4.0), (10.0, 4.0)])
        self.assertEqual(lane["right"], [(0.0, 0.0), (10.0, 0.0)])
        self.assertEqual(lane["subtype"], "road")

    def test_lanelet_missing_a_bound_is_skipped(self):
        self.assertNotIn("201", [lane["id"] for lane in self.lanelets])

    def test_regulatory_elements_are_not_lanelets(self):
        self.assertNotIn("202", [lane["id"] for lane in self.lanelets])

    def test_collects_requested_context_way_types(self):
        by_type = {w["type"]: w for w in self.ways}
        self.assertEqual(
            sorted(by_type), ["crosswalk_polygon", "intersection_area", "stop_line"]
        )

    def test_context_way_keeps_point_order(self):
        area = next(w for w in self.ways if w["type"] == "intersection_area")
        self.assertEqual(area["points"], [(20.0, 20.0), (30.0, 20.0), (30.0, 30.0)])

    def test_lane_boundary_ways_are_not_returned_as_context(self):
        self.assertNotIn("line_thin", {w["type"] for w in self.ways})

    def test_way_with_a_single_point_is_skipped(self):
        self.assertNotIn("105", [w["id"] for w in self.ways])

    def test_way_whose_nodes_lack_coordinates_is_skipped(self):
        self.assertNotIn("106", [w["id"] for w in self.ways])

    def test_way_types_argument_narrows_the_result(self):
        path = Path(self._tmp.name) / "map.osm"
        _, ways = load_lanelet2_context(path, way_types=("stop_line",))
        self.assertEqual({w["type"] for w in ways}, {"stop_line"})


if __name__ == "__main__":
    unittest.main()
