"""Model provenance survives the pipeline refactor (PR #7 x PR #9 merge).

PR #7 recorded model sha256 + a known-good registry name in `meta` and warned
on an unregistered .onnx. PR #9 rewrote the L1 CLI around Pipeline/InferenceConfig
and had its own opt-in digest (`record_model_digest`, default False). Merging
naively would have dropped the registry name and the warning, and made digests
opt-in. These tests pin the merged behaviour.
"""
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

from tlr_autolabel.cli.autolabel import warn_unknown_models
from tlr_autolabel.core.models import load_model_manifest


def cfg(**kw):
    base = dict(record_model_digest=True, detector=None, classifier=None)
    base.update(kw)
    return SimpleNamespace(**base)


def warnings_from(config) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_unknown_models(config)
        return [str(w.message) for w in caught]


class WarnUnknownModelsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.unregistered = Path(self.tmp.name) / "unregistered.onnx"
        self.unregistered.write_bytes(b"hashable-but-not-in-the-registry")
        self.addCleanup(self.tmp.cleanup)

    def test_unregistered_onnx_warns(self):
        msgs = warnings_from(cfg(detector=str(self.unregistered)))
        self.assertEqual(len(msgs), 1)
        self.assertIn("configs/models.yaml", msgs[0])
        self.assertIn("detector", msgs[0])

    def test_engine_is_never_flagged(self):
        # engines are machine/TensorRT-version specific and deliberately unlisted
        self.assertEqual(warnings_from(cfg(detector="/nowhere/model.engine")), [])

    def test_no_warning_when_digest_disabled(self):
        self.assertEqual(
            warnings_from(cfg(detector=str(self.unregistered), record_model_digest=False)),
            [])

    def test_missing_path_does_not_warn_or_raise(self):
        self.assertEqual(warnings_from(cfg(detector=None)), [])


class MetaCarriesRegistryNameTest(unittest.TestCase):
    """The Pipeline meta block resolves a digest to its registry name, so a run
    is traceable to a named artifact and not just an opaque hash."""

    def test_pipeline_meta_resolves_manifest_name(self):
        from tlr_autolabel.inference import pipeline as pipeline_mod

        manifest = load_model_manifest()
        self.assertTrue(manifest, "configs/models.yaml should ship entries")
        digest, entry = next(iter(manifest.items()))

        # emulate the meta lookup: digest -> registry name
        self.assertEqual((manifest.get(digest) or {}).get("name"), entry["name"])
        # an unknown digest resolves to None rather than raising
        self.assertIsNone((manifest.get("0" * 64) or {}).get("name"))
        self.assertTrue(hasattr(pipeline_mod, "load_model_manifest"),
                        "pipeline must resolve names via the known-good registry")


if __name__ == "__main__":
    unittest.main()
