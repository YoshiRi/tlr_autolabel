#!/usr/bin/env python3
"""Create a traffic_signal_re_review/v1 template from RE time series.

The template is a review sidecar, not a GT file by itself. It groups regulatory
elements that share the same physical traffic-light ways, then emits run-length
state segments that a reviewer can accept or edit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tlr_autolabel.core.state_tokens import CANON_RE, LEGACY_RE, elements_key, parse_state

VALID_REVIEW_STATUS = {"unchecked", "accepted", "rejected", "fixed"}


def id_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def sorted_ids(values) -> list[str]:
    return sorted((str(v) for v in values), key=id_sort_key)


def signal_group_id(member_ways: list[str]) -> str:
    return "ways:" + ",".join(sorted_ids(member_ways))


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


def state_segments(observations: list[dict], review_status: str) -> list[dict]:
    segments: list[dict] = []
    for obs in sorted(observations, key=lambda o: o.get("timestamp") or 0):
        state = normalize_state(obs.get("state", "unknown"))
        if segments and segments[-1]["state"] == state:
            cur = segments[-1]
            cur["end_sample_token"] = obs["sample_token"]
            cur["end_timestamp"] = obs.get("timestamp")
            cur["n_frames"] += 1
            cur["flags"] = sorted(set(cur["flags"] + obs.get("flags", [])))
            continue
        segments.append(
            {
                "start_sample_token": obs["sample_token"],
                "end_sample_token": obs["sample_token"],
                "start_timestamp": obs.get("timestamp"),
                "end_timestamp": obs.get("timestamp"),
                "state": state,
                "review_status": review_status,
                "source": "autolabel_segment",
                "n_frames": 1,
                "flags": sorted(obs.get("flags", [])),
                "note": "",
            }
        )
    return segments


def build_template(timeseries: dict, source: str, review_status: str) -> dict:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for series in timeseries.get("series", []):
        grouped[tuple(sorted_ids(series["member_ways"]))].append(series)

    groups = []
    for member_ways, members in sorted(grouped.items(), key=lambda kv: kv[0]):
        representative = max(members, key=lambda s: s.get("n_observations", 0))
        groups.append(
            {
                "signal_group_id": signal_group_id(list(member_ways)),
                "member_ways": list(member_ways),
                "regulatory_element_ids": sorted_ids(
                    s["regulatory_element_id"] for s in members
                ),
                "decisions": state_segments(
                    representative.get("observations", []), review_status
                ),
            }
        )

    return {
        "schema_version": "traffic_signal_re_review/v1",
        "source_timeseries": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "groups": groups,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument(
        "--input",
        default=Path("annotation/traffic_signal_re_timeseries.json"),
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path("annotation/traffic_signal_re_review.template.json"),
        type=Path,
    )
    parser.add_argument(
        "--review-status",
        default="unchecked",
        choices=sorted(VALID_REVIEW_STATUS),
        help="status assigned to every emitted segment",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    input_path = args.input if args.input.is_absolute() else root / args.input
    output_path = args.output if args.output.is_absolute() else root / args.output

    payload = json.loads(input_path.read_text())
    template = build_template(payload, str(args.input), args.review_status)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n")

    n_segments = sum(len(g["decisions"]) for g in template["groups"])
    print(f"wrote {output_path}")
    print(f"groups={len(template['groups'])} segments={n_segments}")


if __name__ == "__main__":
    main()
