"""Regression test: scripts/export_re_to_t4dataset.py must work from any cwd.

run_to_object_ann() launches `python -m tlr_autolabel.t4.convert` as a child
process. `-m` resolves the module against the child's import path, which only
includes the repo root when the child's cwd *is* the repo root (the parent's
in-process sys.path from the scripts/ bootstrap does not propagate to a
subprocess). The fix injects PYTHONPATH into the child so the import resolves
regardless of the caller's cwd.

This test runs the wrapper from a temp dir that is NOT the repo root; before the
fix it failed with `ModuleNotFoundError: No module named 'tlr_autolabel'`.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_to_object_ann_contract import write_minimal_t4_dataset, write_sidecar

ROOT = Path(__file__).resolve().parents[1]


class ReExportCwdRegressionTest(unittest.TestCase):
    def test_export_runs_from_non_repo_root_cwd(self):
        with tempfile.TemporaryDirectory(prefix="tlr_re_export_cwd_") as tmp:
            base = Path(tmp)
            src = base / "src"
            out = base / "out"
            foreign_cwd = base / "elsewhere"
            foreign_cwd.mkdir()

            write_minimal_t4_dataset(src)
            sidecar = src / "annotation" / "traffic_signal_2d_ann.json"
            write_sidecar(sidecar, map_traffic_light_id="501")
            review = src / "annotation" / "traffic_signal_re_review.json"
            review.write_text(json.dumps(
                {"schema_version": "traffic_signal_re_review/v1", "groups": []}))

            result = subprocess.run(
                [sys.executable,
                 str(ROOT / "scripts" / "export_re_to_t4dataset.py"),
                 "--dataset-root", str(src),
                 "--out", str(out),
                 "--sidecar", str(sidecar),
                 "--review", str(review),
                 "--allow-empty-application"],
                cwd=foreign_cwd, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

            self.assertEqual(
                result.returncode, 0,
                f"export failed from a non-repo cwd:\n{result.stderr}")
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertTrue((out / "annotation" / "object_ann.json").exists(),
                            "the delegated t4.convert step did not produce output")


if __name__ == "__main__":
    unittest.main()
