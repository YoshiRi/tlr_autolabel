"""db_tlr category name -> canonical lamp elements.

db_tlr stores the state as the per-box category name, so reading a
human-annotated dataset back in means inverting `db_tlr_state()`. The
vocabulary is not tidy: 19 of its 31 names are underscore-joined colour-first
(`red_straight_left`) and 8 are legacy hyphen-joined arrow-first (`left-red`,
`right-yellow`), so both spellings have to decode and both have to land on the
same elements.
"""
import unittest

from tlr_autolabel.core.state_tokens import elements_key
from tlr_autolabel.t4.convert import db_tlr_state, db_tlr_to_elements, load_vocab

VOCAB = load_vocab()


def decode(name):
    return elements_key(db_tlr_to_elements(name, VOCAB))


class DecodeTest(unittest.TestCase):
    def test_the_categories_this_dataset_actually_uses(self):
        cases = {
            "crosswalk_red": "red-ped",
            "crosswalk_green": "green-ped",
            "green": "green-circle",
            "red": "red-circle",
            "yellow": "amber-circle",
            "red_right": "green-arrow-right,red-circle",
            "red_straight_left": "green-arrow-left,green-arrow-up,red-circle",
            "red_straight_left_right":
                "green-arrow-left,green-arrow-right,green-arrow-up,red-circle",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(decode(name), expected)

    def test_yellow_maps_onto_the_canonical_amber(self):
        # the map's light_bulbs tag says "yellow"; canonical says "amber"
        self.assertEqual(decode("yellow"), "amber-circle")

    def test_arrows_decode_as_green(self):
        # db_tlr never records the arrow colour, and db_tlr_state() drops it on
        # the way out, so green -- the JP convention -- is the only choice that
        # round-trips
        elements = db_tlr_to_elements("red_right", VOCAB)
        arrow = next(e for e in elements if e["shape"] == "arrow")
        self.assertEqual(arrow["color"], "green")

    def test_diagonal_arrow_names(self):
        self.assertEqual(decode("red_leftdiagonal"), "green-arrow-up_left,red-circle")

    def test_unknown_and_unreadable_give_no_elements(self):
        for name in ("unknown", "crosswalk_unknown", "", None, "not_a_category"):
            with self.subTest(name=name):
                self.assertEqual(db_tlr_to_elements(name, VOCAB), [])

    def test_a_name_with_one_unknown_part_is_rejected_whole(self):
        # partial decoding would invent a state the annotator never wrote
        self.assertEqual(db_tlr_to_elements("red_sideways", VOCAB), [])


class LegacySpellingTest(unittest.TestCase):
    def test_hyphen_names_decode(self):
        self.assertEqual(decode("left-red"), "green-arrow-left,red-circle")
        self.assertEqual(decode("right-yellow"), "amber-circle,green-arrow-right")
        self.assertEqual(decode("leftdiagonal-red"), "green-arrow-up_left,red-circle")

    def test_arrow_first_and_colour_first_agree(self):
        self.assertEqual(decode("left-red"), decode("red_left"))
        self.assertEqual(decode("left-red-straight"), decode("red_straight_left"))


class VocabularyCoverageTest(unittest.TestCase):
    def test_every_allowed_name_decodes(self):
        undecodable = [n for n in VOCAB["allowed"]
                       if n not in ("unknown", "crosswalk_unknown")
                       and not db_tlr_to_elements(n, VOCAB)]
        self.assertEqual(undecodable, [],
                         f"these allowed names produce no elements: {undecodable}")

    def test_round_trip_through_the_exporter(self):
        """decode -> db_tlr_state should return an allowed name.

        `red-rightdiagonal` is a known pre-existing gap in the vocabulary
        itself: the allowed list has the hyphen spelling but not the underscore
        one the exporter builds, so a right-diagonal arrow is exported as
        `unknown`. That is a bug in the vocabulary/exporter, not in the decoder,
        and it is asserted here so the fix has a test waiting.
        """
        lossy = []
        for name in sorted(VOCAB["allowed"]):
            if name in ("unknown", "crosswalk_unknown"):
                continue
            back = db_tlr_state(db_tlr_to_elements(name, VOCAB), VOCAB)
            if back == "unknown":
                lossy.append(name)
        self.assertEqual(lossy, ["red-rightdiagonal"],
                         "the set of names the exporter cannot represent changed")


if __name__ == "__main__":
    unittest.main()
