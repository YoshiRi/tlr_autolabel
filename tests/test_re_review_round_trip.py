"""RE timeline review contract test (docs/re_timeline_review.md).

Runs the real pipeline end to end on the synthetic dataset used by the
temporal integration tests: match_traffic_lights.py (A' sidecar) ->
aggregate_regulatory_signals.py (RE timeseries) -> make_re_review_template.py
(review template) -> apply_re_review.py (reviewed A' sidecar). Checks that
`accepted` decisions propagate `state`/`review_status` while preserving
`box2d` and `map_traffic_light_id`, per REFACTOR_PLAN.md section 2.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_match_temporal_integration import ROOT, run_match, write_synthetic_dataset


def add_sample_table(root: Path) -> None:
    # aggregate_regulatory_signals.py needs annotation/sample.json (sample
    # token -> timestamp); the shared match_traffic_lights fixture only
    # builds sample_data.json since the matcher itself doesn't need it.
    sd_rows = json.loads((root / "annotation/sample_data.json").read_text())
    samples = {row["sample_token"]: row["timestamp"] for row in sd_rows}
    (root / "annotation/sample.json").write_text(json.dumps(
        [{"token": tok, "timestamp": ts} for tok, ts in samples.items()], indent=2))


def run_script(name: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / name), *args],
        cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


class ReReviewRoundTripTest(unittest.TestCase):
    def test_accepted_decision_propagates_state_and_preserves_geometry(self):
        with tempfile.TemporaryDirectory(prefix="tlr_re_review_") as tmp:
            root = Path(tmp)
            write_synthetic_dataset(root)
            add_sample_table(root)

            run_match(root, "--no-fill-gaps", "--no-map-fill", "--min-score", "0.5")
            a_prime_path = root / "annotation/traffic_signal_2d_ann.json"
            original = json.loads(a_prime_path.read_text())["annotations"]
            self.assertTrue(original)
            self.assertTrue(all(a["attributes"]["review_status"] == "unchecked"
                                 for a in original))

            run_script("aggregate_regulatory_signals.py", "--dataset-root", str(root))
            timeseries_path = root / "annotation/traffic_signal_re_timeseries.json"
            self.assertTrue(timeseries_path.exists())

            run_script(
                "make_re_review_template.py", "--dataset-root", str(root),
                "--review-status", "accepted",
                "--output", "annotation/re_review.template.json",
            )
            template_path = root / "annotation/re_review.template.json"
            template = json.loads(template_path.read_text())
            self.assertEqual(template["schema_version"], "traffic_signal_re_review/v1")
            self.assertTrue(template["groups"])
            self.assertTrue(all(d["review_status"] == "accepted"
                                 for g in template["groups"] for d in g["decisions"]))

            run_script(
                "apply_re_review.py", "--dataset-root", str(root),
                "--review", "annotation/re_review.template.json",
                "--output", "annotation/a_prime.reviewed.json",
            )
            reviewed = json.loads((root / "annotation/a_prime.reviewed.json").read_text())["annotations"]

            by_token = {a["token"]: a for a in reviewed}
            reviewed_count = sum(1 for a in reviewed if a["attributes"]["review_status"] == "accepted")
            self.assertGreater(reviewed_count, 0)
            for orig in original:
                got = by_token[orig["token"]]
                self.assertEqual(got["box2d"], orig["box2d"])
                self.assertEqual(got["attributes"]["map_traffic_light_id"],
                                  orig["attributes"]["map_traffic_light_id"])
                self.assertEqual(got["attributes"]["raw_state"], orig["attributes"]["raw_state"])
                if got["attributes"]["review_status"] == "accepted":
                    self.assertNotEqual(got["attributes"]["state"], None)


if __name__ == "__main__":
    unittest.main()
