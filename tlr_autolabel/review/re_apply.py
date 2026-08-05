#!/usr/bin/env python3
"""Apply traffic_signal_re_review/v1 decisions to a Tier B sidecar.

This is the join point between CVAT-style per-frame geometry fixes and the
regulatory-element timeline review. Per-group `decisions` change only
review-owned attributes: `state`, `signal_kind`, and `review_status`. A
group may also carry per-camera-channel `visibility_decisions` (visibility
is inherently camera-view-dependent -- a passing vehicle occludes one
camera's view of a signal without affecting another's -- so these are
segmented and applied per channel, independently of the state decisions).
Geometry, map ids, raw_state, detector scores, and provenance stay as they
are either way.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tlr_autolabel.core.state_tokens import CANON_RE, LEGACY_RE, elements_key, parse_state
from tlr_autolabel.core.io import load_json

VALID_REVIEW_STATUS = {"unchecked", "accepted", "rejected", "fixed"}
APPLY_STATUSES = {"accepted", "rejected", "fixed"}
VALID_VISIBILITY = {"full", "partial", "occluded", "unknown"}
APPLY_VISIBILITY_STATUSES = {"accepted", "fixed"}


def invalid_state_tokens(state: str) -> list[str]:
    bad = []
    for token in filter(None, (t.strip() for t in (state or "").split(","))):
        if token == "unknown":
            continue
        if not (CANON_RE.match(token) or LEGACY_RE.match(token)):
            bad.append(token)
    return bad


def normalize_state(state: str) -> str:
    bad = invalid_state_tokens(state)
    if bad:
        raise ValueError(f"invalid state token(s) {bad} in {state!r}")
    return elements_key(parse_state(state)) or "unknown"


def derive_signal_kind(state: str) -> str:
    elements = parse_state(state)
    if any(e["shape"] == "ped" for e in elements):
        return "pedestrian"
    return "vehicle" if elements else "unknown"


def split_ids(value: str) -> set[str]:
    return {v.strip() for v in (value or "").split(",") if v.strip()}


def timestamps_by_sample(sidecar: dict, dataset_root: Path) -> dict[str, int]:
    lookup = {
        ann["sample_token"]: ann["timestamp"]
        for ann in sidecar.get("annotations", [])
        if ann.get("sample_token") and ann.get("timestamp") is not None
    }
    sample_path = dataset_root / "annotation/sample.json"
    if sample_path.exists():
        for row in load_json(sample_path):
            if row.get("token") and row.get("timestamp") is not None:
                lookup.setdefault(row["token"], row["timestamp"])
    return lookup


def parse_signal_group_id(value: str) -> set[str]:
    if not value.startswith("ways:"):
        return set()
    return split_ids(value.removeprefix("ways:"))


def normalize_decisions(review: dict, sample_ts: dict[str, int]) -> list[dict]:
    decisions = []
    errors = []
    for group_index, group in enumerate(review.get("groups", [])):
        member_ways = set(str(w) for w in group.get("member_ways", []))
        member_ways |= parse_signal_group_id(group.get("signal_group_id", ""))
        regulatory_ids = set(str(r) for r in group.get("regulatory_element_ids", []))
        raw_decisions = group.get("decisions", group.get("segments", []))
        if not member_ways and not regulatory_ids:
            errors.append(f"group#{group_index}: no member_ways/regulatory ids")
            continue
        for decision_index, raw in enumerate(raw_decisions):
            status = raw.get("review_status", "unchecked")
            if status not in VALID_REVIEW_STATUS:
                errors.append(
                    f"group#{group_index} decision#{decision_index}: "
                    f"invalid review_status {status!r}"
                )
                continue
            try:
                state = normalize_state(raw.get("state", "unknown"))
            except ValueError as exc:
                errors.append(f"group#{group_index} decision#{decision_index}: {exc}")
                continue

            start_token = raw.get("start_sample_token")
            end_token = raw.get("end_sample_token") or start_token
            start_ts = raw.get("start_timestamp")
            end_ts = raw.get("end_timestamp")
            if start_ts is None and start_token:
                start_ts = sample_ts.get(start_token)
            if end_ts is None and end_token:
                end_ts = sample_ts.get(end_token)
            if start_ts is None or end_ts is None:
                errors.append(
                    f"group#{group_index} decision#{decision_index}: "
                    "missing start/end timestamp and sample token lookup failed"
                )
                continue
            if int(start_ts) > int(end_ts):
                errors.append(
                    f"group#{group_index} decision#{decision_index}: "
                    f"start_timestamp > end_timestamp ({start_ts} > {end_ts})"
                )
                continue
            decisions.append(
                {
                    "group_index": group_index,
                    "decision_index": decision_index,
                    "signal_group_id": group.get("signal_group_id", ""),
                    "member_ways": member_ways,
                    "regulatory_element_ids": regulatory_ids,
                    "start_timestamp": int(start_ts),
                    "end_timestamp": int(end_ts),
                    "state": state,
                    "review_status": status,
                }
            )
    if errors:
        detail = "\n  ".join(errors)
        raise SystemExit(f"review validation failed, {len(errors)} problem(s):\n  {detail}")
    return decisions


def normalize_visibility_decisions(review: dict, sample_ts: dict[str, int]) -> list[dict]:
    decisions = []
    errors = []
    for group_index, group in enumerate(review.get("groups", [])):
        member_ways = set(str(w) for w in group.get("member_ways", []))
        member_ways |= parse_signal_group_id(group.get("signal_group_id", ""))
        regulatory_ids = set(str(r) for r in group.get("regulatory_element_ids", []))
        by_channel = group.get("visibility_decisions", {})
        if not by_channel:
            continue
        if not member_ways and not regulatory_ids:
            errors.append(f"group#{group_index}: no member_ways/regulatory ids for visibility_decisions")
            continue
        for channel, raw_decisions in by_channel.items():
            for decision_index, raw in enumerate(raw_decisions):
                status = raw.get("review_status", "unchecked")
                if status not in VALID_REVIEW_STATUS:
                    errors.append(
                        f"group#{group_index} channel={channel!r} decision#{decision_index}: "
                        f"invalid review_status {status!r}"
                    )
                    continue
                visibility = raw.get("visibility")
                if visibility not in VALID_VISIBILITY:
                    errors.append(
                        f"group#{group_index} channel={channel!r} decision#{decision_index}: "
                        f"invalid visibility {visibility!r} (must be one of {sorted(VALID_VISIBILITY)})"
                    )
                    continue

                start_token = raw.get("start_sample_token")
                end_token = raw.get("end_sample_token") or start_token
                start_ts = raw.get("start_timestamp")
                end_ts = raw.get("end_timestamp")
                if start_ts is None and start_token:
                    start_ts = sample_ts.get(start_token)
                if end_ts is None and end_token:
                    end_ts = sample_ts.get(end_token)
                if start_ts is None or end_ts is None:
                    errors.append(
                        f"group#{group_index} channel={channel!r} decision#{decision_index}: "
                        "missing start/end timestamp and sample token lookup failed"
                    )
                    continue
                if int(start_ts) > int(end_ts):
                    errors.append(
                        f"group#{group_index} channel={channel!r} decision#{decision_index}: "
                        f"start_timestamp > end_timestamp ({start_ts} > {end_ts})"
                    )
                    continue
                decisions.append(
                    {
                        "group_index": group_index,
                        "decision_index": decision_index,
                        "signal_group_id": group.get("signal_group_id", ""),
                        "member_ways": member_ways,
                        "regulatory_element_ids": regulatory_ids,
                        "channel": channel,
                        "start_timestamp": int(start_ts),
                        "end_timestamp": int(end_ts),
                        "visibility": visibility,
                        "review_status": status,
                    }
                )
    if errors:
        detail = "\n  ".join(errors)
        raise SystemExit(f"review validation failed, {len(errors)} problem(s):\n  {detail}")
    return decisions


def decision_matches_annotation(decision: dict, ann: dict) -> bool:
    timestamp = ann.get("timestamp")
    if timestamp is None:
        return False
    if not (decision["start_timestamp"] <= int(timestamp) <= decision["end_timestamp"]):
        return False

    attrs = ann.get("attributes") or {}
    way_id = attrs.get("map_traffic_light_id", "")
    if way_id and way_id in decision["member_ways"]:
        return True

    regulatory_ids = split_ids(attrs.get("regulatory_element_id", ""))
    return bool(regulatory_ids & decision["regulatory_element_ids"])


def visibility_decision_matches_annotation(decision: dict, ann: dict) -> bool:
    if ann.get("channel") != decision["channel"]:
        return False
    return decision_matches_annotation(decision, ann)


def apply_review(
    sidecar: dict, decisions: list[dict], visibility_decisions: list[dict] | None = None,
) -> tuple[dict, dict]:
    out = json.loads(json.dumps(sidecar))
    applied = Counter()
    applied_by_status = Counter()
    overlaps = 0

    active_decisions = [d for d in decisions if d["review_status"] in APPLY_STATUSES]
    active_visibility = [
        d for d in (visibility_decisions or []) if d["review_status"] in APPLY_VISIBILITY_STATUSES
    ]
    visibility_overlaps = 0
    applied_visibility = 0
    for ann in out.get("annotations", []):
        hits = [d for d in active_decisions if decision_matches_annotation(d, ann)]
        if hits:
            if len(hits) > 1:
                overlaps += 1
            decision = hits[-1]
            attrs = ann.setdefault("attributes", {})
            if decision["review_status"] == "rejected":
                attrs["review_status"] = "rejected"
            else:
                attrs["state"] = decision["state"]
                attrs["signal_kind"] = derive_signal_kind(decision["state"])
                attrs["review_status"] = decision["review_status"]
            applied["annotations"] += 1
            applied_by_status[decision["review_status"]] += 1

        vis_hits = [d for d in active_visibility if visibility_decision_matches_annotation(d, ann)]
        if vis_hits:
            if len(vis_hits) > 1:
                visibility_overlaps += 1
            attrs = ann.setdefault("attributes", {})
            attrs["visibility"] = vis_hits[-1]["visibility"]
            applied_visibility += 1

    out["schema_version"] = sidecar.get("schema_version", "traffic_signal_2d/v2")
    out["source"] = "re_timeline_review"
    out["created_at"] = datetime.now(timezone.utc).isoformat()
    out["re_review"] = {
        "schema_version": "traffic_signal_re_review/v1",
        "applied_decisions": len(active_decisions),
        "applied_annotations": applied["annotations"],
        "applied_by_status": dict(applied_by_status),
        "overlapping_annotation_matches": overlaps,
        "applied_visibility_decisions": len(active_visibility),
        "applied_visibility_annotations": applied_visibility,
        "overlapping_visibility_matches": visibility_overlaps,
    }
    return out, out["re_review"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument(
        "--input",
        default=Path("annotation/traffic_signal_2d_ann.json"),
        type=Path,
        help="Tier B sidecar to update",
    )
    parser.add_argument(
        "--review",
        default=Path("annotation/traffic_signal_re_review.json"),
        type=Path,
        help="traffic_signal_re_review/v1 sidecar",
    )
    parser.add_argument(
        "--output",
        default=Path("annotation/traffic_signal_2d_ann.reviewed.json"),
        type=Path,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    input_path = args.input if args.input.is_absolute() else root / args.input
    review_path = args.review if args.review.is_absolute() else root / args.review
    output_path = args.output if args.output.is_absolute() else root / args.output

    sidecar = load_json(input_path)
    review = load_json(review_path)
    if review.get("schema_version") != "traffic_signal_re_review/v1":
        raise SystemExit(
            f"unsupported review schema_version: {review.get('schema_version')!r}"
        )

    sample_ts = timestamps_by_sample(sidecar, root)
    decisions = normalize_decisions(review, sample_ts)
    visibility_decisions = normalize_visibility_decisions(review, sample_ts)
    reviewed, summary = apply_review(sidecar, decisions, visibility_decisions)

    print(
        "decisions="
        f"{len(decisions)} apply_decisions={summary['applied_decisions']} "
        f"applied_annotations={summary['applied_annotations']} "
        f"by_status={summary['applied_by_status']} "
        f"overlaps={summary['overlapping_annotation_matches']}"
    )
    print(
        "visibility_decisions="
        f"{len(visibility_decisions)} apply_visibility_decisions={summary['applied_visibility_decisions']} "
        f"applied_visibility_annotations={summary['applied_visibility_annotations']} "
        f"overlaps={summary['overlapping_visibility_matches']}"
    )
    if args.dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reviewed, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
