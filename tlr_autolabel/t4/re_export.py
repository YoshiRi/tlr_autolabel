#!/usr/bin/env python3
"""Convert RE-based traffic-signal annotation into t4dataset B/B'.

This is an adapter/orchestration layer, not a new canonical IF. It takes:

  - geometry/map relation from traffic_signal_2d/v2
  - state decisions from traffic_signal_re_review/v1, or from traffic_signal_re/v1

and writes a derived t4dataset using tlr_autolabel.t4.convert:

  B:  object_ann/category/attribute/instance
  B': B + traffic_light.json when map relations exist

Intermediate traffic_signal_2d/v2 is temporary and deleted by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from tlr_autolabel.review.re_apply import apply_review, normalize_decisions, timestamps_by_sample
from tlr_autolabel.review.re_template import build_template

APPLY_STATUSES = {"accepted", "fixed", "rejected"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def default_review_path(root: Path) -> Path | None:
    for rel in (
        Path("annotation/traffic_signal_re_review.json"),
        Path("annotation/traffic_signal_re_review.template.json"),
        Path("build/tl_match/re_review.accept_autolabel.json"),
        Path("build/tl_match/re_review.template.json"),
    ):
        path = root / rel
        if path.exists():
            return path
    return None


def decisions_from_review(
    review: dict,
    sidecar: dict,
    dataset_root: Path,
    unchecked_as: str,
) -> list[dict]:
    if review.get("schema_version") != "traffic_signal_re_review/v1":
        raise SystemExit(
            f"unsupported review schema_version: {review.get('schema_version')!r}"
        )
    decisions = normalize_decisions(review, timestamps_by_sample(sidecar, dataset_root))
    if unchecked_as != "skip":
        for decision in decisions:
            if decision["review_status"] == "unchecked":
                decision["review_status"] = unchecked_as
    return decisions


def decisions_from_timeseries(
    timeseries: dict,
    sidecar: dict,
    dataset_root: Path,
    source: str,
    status: str,
) -> list[dict]:
    if timeseries.get("schema_version") != "traffic_signal_re/v1":
        raise SystemExit(
            f"unsupported timeseries schema_version: {timeseries.get('schema_version')!r}"
        )
    review = build_template(timeseries, source, status)
    return normalize_decisions(review, timestamps_by_sample(sidecar, dataset_root))


def summarize_decisions(decisions: list[dict]) -> dict:
    by_status: dict[str, int] = {}
    for decision in decisions:
        by_status[decision["review_status"]] = by_status.get(decision["review_status"], 0) + 1
    return {
        "decisions": len(decisions),
        "apply_decisions": sum(d["review_status"] in APPLY_STATUSES for d in decisions),
        "by_status": by_status,
    }


def write_temp_sidecar(reviewed: dict, keep_path: Path | None) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        "w",
        prefix="traffic_signal_2d_ann.re_applied.",
        suffix=".json",
        delete=False,
    )
    with tmp:
        json.dump(reviewed, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
    temp_path = Path(tmp.name)
    if keep_path is not None:
        keep_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temp_path, keep_path)
    return temp_path


def run_to_object_ann(args: argparse.Namespace, sidecar_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "tlr_autolabel.t4.convert",
        "--t4-dataset",
        str(args.dataset_root),
        "--sidecar",
        str(sidecar_path),
        "--out",
        str(args.out),
        "--mask",
        args.mask,
    ]
    if args.write_deprecated_map_association:
        cmd.append("--write-deprecated-map-association")
    # `python -m tlr_autolabel.t4.convert` only resolves when the repo root is on
    # the child's import path; the parent's in-process sys.path (set via the
    # scripts/ bootstrap) does not propagate to the subprocess. Inject it via
    # PYTHONPATH so this works regardless of the caller's cwd. cwd is left
    # untouched because --dataset-root / --out may be relative to it.
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--sidecar",
        default=Path("annotation/traffic_signal_2d_ann.json"),
        type=Path,
        help="traffic_signal_2d/v2 geometry/map source",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--review",
        type=Path,
        help="traffic_signal_re_review/v1 sidecar. If omitted, common defaults are searched.",
    )
    source.add_argument(
        "--timeseries",
        type=Path,
        help="traffic_signal_re/v1 sidecar; converted to review decisions internally.",
    )
    parser.add_argument(
        "--unchecked-as",
        default="skip",
        choices=["skip", "accepted", "fixed", "rejected"],
        help="How to treat unchecked review decisions. Use accepted only when the RE file "
             "is being promoted as annotation data.",
    )
    parser.add_argument(
        "--timeseries-status",
        default="accepted",
        choices=["accepted", "fixed", "rejected"],
        help="review_status assigned to decisions generated from traffic_signal_re/v1",
    )
    parser.add_argument("--mask", choices=["null", "placeholder", "real"], default="null")
    parser.add_argument(
        "--keep-applied-sidecar",
        type=Path,
        help="Optional path to keep the temporary reviewed traffic_signal_2d/v2 sidecar.",
    )
    parser.add_argument(
        "--write-deprecated-map-association",
        action="store_true",
        help="temporary compatibility only; passes through to tlr_autolabel.t4.convert",
    )
    parser.add_argument(
        "--allow-empty-application",
        action="store_true",
        help="allow zero applicable RE decisions or zero matched annotations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.out = args.out.resolve()

    sidecar_path = resolve_path(args.dataset_root, args.sidecar)
    sidecar = load_json(sidecar_path)

    if args.timeseries:
        timeseries_path = resolve_path(args.dataset_root, args.timeseries)
        decisions = decisions_from_timeseries(
            load_json(timeseries_path),
            sidecar,
            args.dataset_root,
            str(args.timeseries),
            args.timeseries_status,
        )
        source_desc = str(timeseries_path)
    else:
        review_path = resolve_path(args.dataset_root, args.review) if args.review else default_review_path(args.dataset_root)
        if review_path is None:
            raise SystemExit(
                "no RE review file found; pass --review or --timeseries explicitly"
            )
        decisions = decisions_from_review(
            load_json(review_path),
            sidecar,
            args.dataset_root,
            args.unchecked_as,
        )
        source_desc = str(review_path)

    decision_summary = summarize_decisions(decisions)
    reviewed, apply_summary = apply_review(sidecar, decisions)
    if not args.allow_empty_application:
        if decision_summary["apply_decisions"] == 0:
            raise SystemExit(
                "no applicable RE decisions; pass --unchecked-as accepted/fixed/rejected "
                "when promoting an unchecked template, or use --allow-empty-application"
            )
        if apply_summary["applied_annotations"] == 0:
            raise SystemExit(
                "RE decisions did not match any 2D annotations; check member_ways/"
                "regulatory_element_ids or pass --allow-empty-application"
            )
    temp_path = write_temp_sidecar(reviewed, args.keep_applied_sidecar)
    try:
        print(
            "re source="
            f"{source_desc} decisions={decision_summary['decisions']} "
            f"apply_decisions={decision_summary['apply_decisions']} "
            f"by_status={decision_summary['by_status']} "
            f"applied_annotations={apply_summary['applied_annotations']} "
            f"overlaps={apply_summary['overlapping_annotation_matches']}",
            flush=True,
        )
        run_to_object_ann(args, temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
