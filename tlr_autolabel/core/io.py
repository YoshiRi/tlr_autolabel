"""Shared JSON/table helpers for T4 annotation I/O (REFACTOR_PLAN.md phase 4).

token_of: deterministic token derivation used by to_object_ann.py and
export_object_ann_to_t4dataset.py to mint stable t4dataset row tokens from
their component parts.

load_json: trivial read helper duplicated verbatim across several review/eval
CLI scripts (apply_re_review.py, export_cvat_signal_task.py,
import_cvat_signal_annotations.py).
"""
import hashlib
import json
from pathlib import Path


def token_of(*parts: object) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())
