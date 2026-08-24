"""Model artifact provenance: sha256 hashing + a known-good manifest.

Internal-use model management (no network fetch). Every L1 run records the
sha256 of the detector/classifier it actually loaded, and resolves that hash to
a human name via `configs/models.yaml`. Unknown hashes are surfaced so silent
model drift (e.g. a repo-vendored copy diverging from the store copy) is caught.

Engines (.engine) are machine/TensorRT-version specific and are intentionally
not tracked in the manifest.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs" / "models.yaml"


def sha256_file(path, chunk_size: int = 1 << 20) -> str | None:
    """Full-file sha256 as hex, or None if the path is missing/unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, TypeError):
        return None


def load_model_manifest(path=MANIFEST_PATH) -> dict[str, dict]:
    """Return {sha256: entry} from the known-good manifest. Empty if absent."""
    try:
        import yaml
        data = yaml.safe_load(Path(path).read_text()) or {}
    except OSError:
        return {}
    out: dict[str, dict] = {}
    for entry in data.get("models", []):
        digest = entry.get("sha256")
        if digest:
            out[digest] = entry
    return out


def model_provenance(path, manifest: dict[str, dict]) -> dict:
    """Provenance for one model file: its sha256, the manifest name it resolves
    to (or None), and whether it is a known-good artifact. Tolerant of a missing
    file (records sha256=None, known=False)."""
    digest = sha256_file(path)
    entry = manifest.get(digest) if digest else None
    return {
        "sha256": digest,
        "model": entry.get("name") if entry else None,
        "known": entry is not None,
    }
