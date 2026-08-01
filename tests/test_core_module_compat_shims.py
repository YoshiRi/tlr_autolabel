"""Backward-compat test for the old top-level import paths.

state_tokens.py and temporal_association.py moved to
tlr_autolabel/core/state_tokens.py and tlr_autolabel/tracking/temporal.py
(REFACTOR_PLAN.md step 3). Existing scripts still do
`from state_tokens import ...` / `from temporal_association import ...`;
this test only needs to keep passing until every caller is migrated to the
package path, at which point the shim modules can be deleted.
"""
import unittest


class StateTokensShimTest(unittest.TestCase):
    def test_old_import_path_still_works(self):
        from state_tokens import parse_state, elements_key
        from tlr_autolabel.core.state_tokens import parse_state as new_parse_state

        self.assertIs(parse_state, new_parse_state)
        self.assertEqual(elements_key(parse_state("red-circle")), "red-circle")


class TemporalAssociationShimTest(unittest.TestCase):
    def test_old_import_path_still_works(self):
        from temporal_association import TemporalAssociator, TemporalTrackingConfig
        from tlr_autolabel.tracking.temporal import TemporalAssociator as NewTemporalAssociator

        self.assertIs(TemporalAssociator, NewTemporalAssociator)
        self.assertIsInstance(TemporalTrackingConfig(), TemporalTrackingConfig)


if __name__ == "__main__":
    unittest.main()
