#!/usr/bin/env python3
"""Report where the lanelet2 map and the annotated images disagree.

The review views made the disagreement visible; this makes it measurable, so a
map replacement can be judged by a number rather than by scrolling.

The check is deliberately independent of the matcher's own verdict: it
re-projects every map traffic light into each frame with
`project_traffic_lights()` -- the same projection the matcher uses -- and
associates the results with the annotated boxes in image space. A box the
matcher rejected still counts as an observation here if it lands where a map
signal projects, so a matcher bug and a map error do not look alike.

Each frame yields three classes:

  paired      a projected map signal and a detected box agree
  map_only    a map signal projects as readable, nothing was detected there
  image_only  a box with a readable state, no map signal projects near it

Findings are aggregated over the whole run, never per frame: one occluded
frame or one missed detection means nothing, a persistent asymmetry means a
map problem.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tlr_autolabel.map.lanelet2 import load_lanelet2_traffic_lights
from tlr_autolabel.map.projection import project_traffic_lights
from tlr_autolabel.review.re_review_timeline import (
    DEFAULT_CROP_CHANNELS,
    parse_float,
    resolve_crop_channels,
)
from tlr_autolabel.t4.dataset import T4Dataset

# A state the detector could actually read. `unknown` is usually the back of a
# housing or a signal too small to call, and asserting the map is wrong on that
# basis would be noise.
UNREADABLE_STATES = {"", "unknown"}
READABLE_FACINGS = ("front", "front_oblique")


def box_center(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_longer_side(box) -> float:
    return max(box[2] - box[0], box[3] - box[1])


def center_distance(a, b) -> float:
    (ax, ay), (bx, by) = box_center(a), box_center(b)
    return math.hypot(ax - bx, ay - by)


def association_radius(projected_box, min_px: float, scale: float) -> float:
    """How far from its projection a real detection may sit.

    Scaled by the projected size because map/ego pose error shows up as a
    larger pixel offset on a near (large) signal than on a distant one.
    """
    return max(min_px, scale * box_longer_side(projected_box))


def associate(projected: list[dict], detections: list[dict], min_px: float, scale: float):
    """Greedy nearest-centre association, closest pairs first.

    Returns (pairs, map_only, image_only) where pairs is a list of
    (projected, detection, distance_px).
    """
    scored = []
    for p_index, proj in enumerate(projected):
        radius = association_radius(proj["bbox"], min_px, scale)
        for d_index, det in enumerate(detections):
            distance = center_distance(proj["bbox"], det["box2d"])
            if distance <= radius:
                scored.append((distance, p_index, d_index))
    scored.sort()

    used_projected: set[int] = set()
    used_detections: set[int] = set()
    pairs = []
    for distance, p_index, d_index in scored:
        if p_index in used_projected or d_index in used_detections:
            continue
        used_projected.add(p_index)
        used_detections.add(d_index)
        pairs.append((projected[p_index], detections[d_index], distance))

    map_only = [p for i, p in enumerate(projected) if i not in used_projected]
    image_only = [d for i, d in enumerate(detections) if i not in used_detections]
    return pairs, map_only, image_only


def nearest_projection(detection: dict, projected: list[dict]):
    """Closest projected signal regardless of the association radius.

    A detection sitting just outside the radius of one particular way, run
    after run, reads as a map *position* error; one with nothing anywhere near
    reads as a *missing* entry. Distinguishing them is the useful part.
    """
    best = None
    for proj in projected:
        distance = center_distance(proj["bbox"], detection["box2d"])
        if best is None or distance < best[1]:
            best = (proj, distance)
    return best


def attrs(annotation: dict) -> dict:
    return annotation.get("attributes") or {}


def is_readable(annotation: dict, min_score: float) -> bool:
    a = attrs(annotation)
    if (a.get("state") or "") in UNREADABLE_STATES:
        return False
    return parse_float(a.get("detector_score"), 0.0) >= min_score


def analyse(
    dataset: T4Dataset,
    traffic_lights: dict,
    annotations: list[dict],
    channels: set[str] | None,
    max_distance: float,
    assoc_min_px: float,
    assoc_scale: float,
    min_score: float,
) -> dict:
    by_sample_data: dict[str, list[dict]] = defaultdict(list)
    for annotation in annotations:
        token = annotation.get("sample_data_token")
        if token:
            by_sample_data[token].append(annotation)

    # Facing is tracked separately because `front_oblique` is the projection's
    # own "front face, too steep to read" class -- its matched rate collapses
    # by design, so counting it toward observability manufactures findings.
    per_way = defaultdict(
        lambda: {"projected": 0, "front": 0, "oblique": 0,
                 "paired": 0, "paired_front": 0, "offsets": []}
    )
    unmapped = defaultdict(lambda: {
        "frames": 0, "states": Counter(), "kinds": Counter(),
        "nearest_way": Counter(), "offsets": [],
    })
    n_frames = 0
    n_pairs = 0
    n_map_only = 0
    n_image_only = 0
    per_channel = defaultdict(lambda: {"frames": 0, "paired": 0, "map_only": 0, "image_only": 0})

    for token, frame in dataset.camera_frames_by_token.items():
        if channels is not None and frame.channel not in channels:
            continue
        detections = by_sample_data.get(token)
        if not detections:
            continue
        readable_detections = [d for d in detections if is_readable(d, min_score)]
        image_wh = (frame.sample_data.get("width"), frame.sample_data.get("height"))
        if not all(image_wh):
            continue

        projected = [
            candidate
            for candidate in project_traffic_lights(
                {"ego_pose": frame.ego_pose, "calib": frame.calibrated_sensor},
                traffic_lights,
                max_distance,
                image_wh,
            )
            if candidate["facing"] in READABLE_FACINGS
        ]
        pairs, map_only, image_only = associate(
            projected, readable_detections, assoc_min_px, assoc_scale
        )

        n_frames += 1
        n_pairs += len(pairs)
        n_map_only += len(map_only)
        n_image_only += len(image_only)
        channel_stats = per_channel[frame.channel]
        channel_stats["frames"] += 1
        channel_stats["paired"] += len(pairs)
        channel_stats["map_only"] += len(map_only)
        channel_stats["image_only"] += len(image_only)

        for candidate in projected:
            entry = per_way[candidate["way_id"]]
            entry["projected"] += 1
            if candidate["facing"] == "front":
                entry["front"] += 1
            else:
                entry["oblique"] += 1
        for candidate, _detection, distance in pairs:
            entry = per_way[candidate["way_id"]]
            entry["paired"] += 1
            if candidate["facing"] == "front":
                entry["paired_front"] += 1
            entry["offsets"].append(distance)

        for detection in image_only:
            near = nearest_projection(detection, projected)
            key = frame.channel
            bucket = unmapped[key]
            bucket["frames"] += 1
            bucket["states"][attrs(detection).get("state", "")] += 1
            bucket["kinds"][attrs(detection).get("signal_kind", "") or "unknown"] += 1
            if near is not None:
                bucket["nearest_way"][near[0]["way_id"]] += 1
                bucket["offsets"].append(near[1])

    ways = {}
    for way_id, entry in sorted(per_way.items()):
        offsets = entry["offsets"]
        ways[way_id] = {
            "projected_readable_frames": entry["projected"],
            "projected_front_frames": entry["front"],
            "projected_oblique_frames": entry["oblique"],
            "paired_frames": entry["paired"],
            "paired_front_frames": entry["paired_front"],
            "observation_rate": (
                round(entry["paired"] / entry["projected"], 3) if entry["projected"] else None
            ),
            # The rate findings key off this one: only head-on frames are a fair
            # test of whether the mapped signal is really there.
            "front_observation_rate": (
                round(entry["paired_front"] / entry["front"], 3) if entry["front"] else None
            ),
            "median_offset_px": round(median(offsets), 1) if offsets else None,
        }

    return {
        "frames": n_frames,
        "paired": n_pairs,
        "map_only": n_map_only,
        "image_only": n_image_only,
        "per_channel": {k: dict(v) for k, v in sorted(per_channel.items())},
        "ways": ways,
        "unmapped_by_channel": {
            channel: {
                "boxes": bucket["frames"],
                "states": dict(bucket["states"].most_common()),
                "kinds": dict(bucket["kinds"].most_common()),
                "nearest_way": dict(bucket["nearest_way"].most_common(5)),
                "median_offset_px": (
                    round(median(bucket["offsets"]), 1) if bucket["offsets"] else None
                ),
            }
            for channel, bucket in sorted(unmapped.items())
        },
    }


def median(values) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def find_warnings(
    report: dict,
    min_projected_frames: int,
    min_observation_rate: float,
    min_unmapped_boxes: int,
    offset_hint_px: float,
) -> list[dict]:
    """Turn the aggregate into things worth acting on.

    Thresholds gate on how often something was *observable*, not on raw counts,
    so a signal the run barely saw cannot raise an alarm.
    """
    warnings = []

    for way_id, stats in report["ways"].items():
        # Only head-on frames count: a way the run only ever saw at a steep
        # angle was never really testable, and reporting it is noise.
        front = stats["projected_front_frames"]
        if front < min_projected_frames:
            continue
        rate = stats["front_observation_rate"] or 0.0
        if stats["paired_front_frames"] == 0:
            warnings.append({
                "kind": "signal_never_observed",
                "way": way_id,
                "detail": (
                    f"projects head-on in {front} frames but was never detected "
                    "-- the map may hold a signal that is not there"
                ),
                "projected_front_frames": front,
                "front_observation_rate": rate,
            })
        elif rate < min_observation_rate:
            warnings.append({
                "kind": "low_observation_rate",
                "way": way_id,
                "detail": (
                    f"detected in only {stats['paired_front_frames']} of {front} "
                    f"head-on frames (rate {rate:.2f}) -- likely a map "
                    "position/orientation error or persistent occlusion"
                ),
                "projected_front_frames": front,
                "front_observation_rate": rate,
            })
        elif stats["median_offset_px"] is not None and stats["median_offset_px"] >= offset_hint_px:
            warnings.append({
                "kind": "large_projection_offset",
                "way": way_id,
                "detail": (
                    f"pairs sit a median {stats['median_offset_px']} px from where the "
                    "map projects it -- the association still holds, but the map "
                    "position looks off"
                ),
                "median_offset_px": stats["median_offset_px"],
            })

    for channel, bucket in report["unmapped_by_channel"].items():
        if bucket["boxes"] < min_unmapped_boxes:
            continue
        kinds = ", ".join(f"{k}={v}" for k, v in bucket["kinds"].items())
        warnings.append({
            "kind": "unmapped_signal",
            "channel": channel,
            "detail": (
                f"{bucket['boxes']} boxes with a readable state have no map signal "
                f"projecting near them ({kinds}) -- the map is missing entries"
            ),
            "boxes": bucket["boxes"],
            "states": bucket["states"],
            "nearest_way": bucket["nearest_way"],
        })

    order = {
        "unmapped_signal": 0,
        "signal_never_observed": 1,
        "low_observation_rate": 2,
        "large_projection_offset": 3,
    }
    warnings.sort(key=lambda w: (order.get(w["kind"], 99), -w.get("boxes", 0)))
    return warnings


def format_report(report: dict, warnings: list[dict]) -> str:
    lines = [
        f"frames analysed : {report['frames']}",
        f"paired          : {report['paired']}",
        f"map only        : {report['map_only']}  (projects readable, nothing detected)",
        f"image only      : {report['image_only']}  (readable detection, no map signal)",
        "",
        "per way:",
        f"  {'way':>8} {'head-on':>8} {'oblique':>8} {'paired':>7} "
        f"{'rate':>6} {'offset px':>10}",
    ]
    for way_id, stats in report["ways"].items():
        rate = stats["front_observation_rate"]
        offset = stats["median_offset_px"]
        lines.append(
            f"  {way_id:>8} {stats['projected_front_frames']:>8} "
            f"{stats['projected_oblique_frames']:>8} "
            f"{stats['paired_front_frames']:>7} "
            f"{'-' if rate is None else format(rate, '.2f'):>6} "
            f"{'-' if offset is None else offset:>10}"
        )
    if not warnings:
        lines += ["", "no findings"]
        return "\n".join(lines)

    lines += ["", f"{len(warnings)} finding(s):"]
    for warning in warnings:
        subject = warning.get("way") or warning.get("channel") or ""
        lines.append(f"  [{warning['kind']}] {subject}")
        lines.append(f"      {warning['detail']}")
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument(
        "--sidecar", default=Path("annotation/traffic_signal_2d_ann.json"), type=Path
    )
    parser.add_argument("--map", default=Path("map/lanelet2_map.osm"), type=Path)
    parser.add_argument("--crop-channels", default=DEFAULT_CROP_CHANNELS)
    parser.add_argument(
        "--max-distance", default=120.0, type=float,
        help="projection range in metres",
    )
    parser.add_argument(
        "--assoc-min-px", default=60.0, type=float,
        help="minimum association radius between a projection and a detection",
    )
    parser.add_argument(
        "--assoc-scale", default=2.0, type=float,
        help="association radius as a multiple of the projected box's longer side",
    )
    parser.add_argument(
        "--min-detector-score", default=0.3, type=float,
        help="ignore detections weaker than this when asserting the map is wrong",
    )
    parser.add_argument(
        "--min-projected-frames", default=10, type=int,
        help="a way must be observable this often before it can raise a finding",
    )
    parser.add_argument(
        "--min-observation-rate", default=0.3, type=float,
        help="observation rate below which a mapped signal is reported",
    )
    parser.add_argument(
        "--min-unmapped-boxes", default=20, type=int,
        help="unmapped detections needed before reporting a missing map entry",
    )
    parser.add_argument(
        "--offset-hint-px", default=80.0, type=float,
        help="median pairing offset above which a map position looks off",
    )
    parser.add_argument(
        "--output", default=None, type=Path,
        help="write the full JSON report here as well as summarising it",
    )
    parser.add_argument(
        "--fail-on-finding",
        action="store_true",
        help="exit non-zero when there is at least one finding (for CI)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


def run(args: argparse.Namespace) -> None:
    root = args.dataset_root.resolve()
    sidecar_path = args.sidecar if args.sidecar.is_absolute() else root / args.sidecar
    map_path = args.map if args.map.is_absolute() else root / args.map
    for path, what in ((sidecar_path, "sidecar"), (map_path, "lanelet2 map")):
        if not path.exists():
            raise SystemExit(f"{what} not found: {path}")

    annotations = json.loads(sidecar_path.read_text()).get("annotations", [])
    if not annotations:
        raise SystemExit(f"no annotations in {sidecar_path}")
    channels = resolve_crop_channels(
        args.crop_channels,
        {a.get("channel") for a in annotations if a.get("channel")},
    )
    traffic_lights, _ = load_lanelet2_traffic_lights(map_path)

    report = analyse(
        T4Dataset.load(root),
        traffic_lights,
        annotations,
        channels,
        args.max_distance,
        args.assoc_min_px,
        args.assoc_scale,
        args.min_detector_score,
    )
    if not report["frames"]:
        raise SystemExit(
            "no frames to analyse -- check --crop-channels and that the sidecar "
            "annotations carry sample_data_token"
        )

    warnings = find_warnings(
        report,
        args.min_projected_frames,
        args.min_observation_rate,
        args.min_unmapped_boxes,
        args.offset_hint_px,
    )
    report["findings"] = warnings
    print(format_report(report, warnings))

    if args.output is not None:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"\nwrote {output}")

    if warnings and args.fail_on_finding:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
