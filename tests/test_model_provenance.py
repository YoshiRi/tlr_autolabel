"""Tests for model provenance hashing + the known-good manifest
(tlr_autolabel/core/models.py, configs/models.yaml)."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from tlr_autolabel.core import models

REPO_ROOT = Path(__file__).resolve().parents[1]


class Sha256FileTest(unittest.TestCase):
    def test_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.onnx"
            p.write_bytes(b"hello-model-bytes")
            self.assertEqual(models.sha256_file(p),
                             hashlib.sha256(b"hello-model-bytes").hexdigest())

    def test_missing_file_returns_none(self):
        self.assertIsNone(models.sha256_file("/no/such/file.onnx"))
        self.assertIsNone(models.sha256_file(None))


class ManifestTest(unittest.TestCase):
    def test_shipped_manifest_loads_and_is_keyed_by_sha256(self):
        manifest = models.load_model_manifest()
        self.assertTrue(manifest, "configs/models.yaml should have entries")
        for digest, entry in manifest.items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertIn("name", entry)
            self.assertIn("role", entry)

    def test_missing_manifest_returns_empty(self):
        self.assertEqual(models.load_model_manifest("/no/such/models.yaml"), {})


class ProvenanceTest(unittest.TestCase):
    def test_known_hash_resolves_to_name(self):
        manifest = models.load_model_manifest()
        known_digest = next(iter(manifest))
        with tempfile.TemporaryDirectory() as tmp:
            # a file whose bytes hash to a known digest is impractical to forge;
            # instead exercise model_provenance against a synthesized manifest.
            fake_path = Path(tmp) / "m.onnx"
            fake_path.write_bytes(b"x")
            digest = models.sha256_file(fake_path)
            prov = models.model_provenance(fake_path, {digest: {"name": "unit-model"}})
            self.assertEqual(prov, {"sha256": digest, "model": "unit-model", "known": True})
        self.assertIn("name", manifest[known_digest])

    def test_unknown_hash_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.onnx"
            p.write_bytes(b"unlisted")
            prov = models.model_provenance(p, {})
            self.assertIsNotNone(prov["sha256"])
            self.assertIsNone(prov["model"])
            self.assertFalse(prov["known"])

    def test_missing_file_is_tolerated(self):
        prov = models.model_provenance("/no/such/file.onnx", {})
        self.assertEqual(prov, {"sha256": None, "model": None, "known": False})


if __name__ == "__main__":
    unittest.main()
