#!/usr/bin/env python3
"""Generate all three review views and serve them from one command.

The views are separate tools so each can be debugged on its own, but the
normal workflow uses them together: correct annotations in the timeline, then
check the result per image and on the map. This wires that loop up --
including regenerating the read-only views against the committed review after
every commit, so "Frame view" and "Map view" show what the corrections
actually produced rather than the raw detector output.

Serving is the default, unlike `re_review_timeline` where `--serve` opts in:
the commit-then-refresh loop only works over the save endpoint. Pass
`--no-serve` to just generate the pages.

Each underlying tool keeps its own entry point:

    python3 -m tlr_autolabel.review.re_review_timeline --dataset-root <ds> --serve
    python3 -m tlr_autolabel.review.re_frame_view      --dataset-root <ds>
    python3 -m tlr_autolabel.review.re_map_view        --dataset-root <ds>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tlr_autolabel.review import (
    re_frame_view,
    re_map_consistency,
    re_map_view,
    re_review_timeline,
)
from tlr_autolabel.review.re_review_timeline import (
    DEFAULT_CROP_CHANNELS,
    draft_path_for,
)

TIMELINE_HTML = Path("build/tl_match/re_review_timeline.html")
FRAME_HTML = Path("build/tl_match/re_frame_view.html")
MAP_HTML = Path("build/tl_match/re_map_view.html")
CONSISTENCY_JSON = Path("build/tl_match/map_consistency.json")


def view_argv(args: argparse.Namespace, review: Path | None) -> list[str]:
    """Flags shared by both read-only views."""
    argv = [
        "--dataset-root", str(args.dataset_root),
        "--sidecar", str(args.sidecar),
        "--crop-channels", args.crop_channels,
    ]
    if review is not None:
        argv += ["--review", str(review)]
    return argv


def build_consistency(args: argparse.Namespace, review: Path | None) -> Path | None:
    """Run the map/image check and return the report path, or None.

    Skipped rather than fatal when the dataset has no lanelet2 map: the views
    are still worth having, and a missing map is not the reviewer's problem.
    """
    if args.no_consistency:
        return None
    root = args.dataset_root.resolve()
    map_path = args.map if args.map.is_absolute() else root / args.map
    if not map_path.exists():
        print(f"skipping consistency check: no map at {map_path}", flush=True)
        return None
    argv = view_argv(args, review) + [
        "--map", str(args.map),
        "--output", str(args.consistency_out),
    ]
    re_map_consistency.run(re_map_consistency.parse_args(argv))
    return args.consistency_out


def build_views(args: argparse.Namespace, review: Path | None) -> None:
    """(Re)generate the two read-only views, optionally against a review."""
    consistency = build_consistency(args, review)
    re_frame_view.run(re_frame_view.parse_args(
        view_argv(args, review)
        + ["--output", str(args.frame_view),
           "--map-view", str(args.map_view),
           "--timeline", str(args.timeline)]
    ))
    map_argv = view_argv(args, review) + [
        "--output", str(args.map_view),
        "--map", str(args.map),
        "--context-radius", str(args.context_radius),
        "--frame-view", str(args.frame_view),
        "--timeline", str(args.timeline),
    ]
    if consistency is not None:
        map_argv += ["--consistency", str(consistency)]
    re_map_view.run(re_map_view.parse_args(map_argv))


def timeline_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--dataset-root", str(args.dataset_root),
        "--input", str(args.input),
        "--sidecar", str(args.sidecar),
        "--crop-channels", args.crop_channels,
        "--output", str(args.timeline),
        "--frame-view", str(args.frame_view),
        "--map-view", str(args.map_view),
        "--review-out", str(args.review_out),
        "--port", str(args.port),
    ]
    if args.show_empty_crop_segments:
        argv.append("--show-empty-crop-segments")
    if not args.no_serve:
        argv.append("--serve")
    return argv


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument(
        "--input", default=Path("annotation/traffic_signal_re_timeseries.json"), type=Path
    )
    parser.add_argument(
        "--sidecar", default=Path("annotation/traffic_signal_2d_ann.json"), type=Path
    )
    parser.add_argument("--map", default=Path("map/lanelet2_map.osm"), type=Path)
    parser.add_argument(
        "--review-out",
        default=Path("annotation/traffic_signal_re_review.json"),
        type=Path,
        help="committed review; the views are regenerated against it after commit",
    )
    parser.add_argument("--crop-channels", default=DEFAULT_CROP_CHANNELS)
    parser.add_argument("--context-radius", default=120.0, type=float)
    parser.add_argument("--show-empty-crop-segments", action="store_true")
    parser.add_argument("--port", default=8765, type=int)
    # Serving is the default here, the opposite of re_review_timeline: the
    # commit-then-refresh loop this launcher exists for only works over the
    # save endpoint. --serve is accepted anyway so the habit from the single
    # tool does not fail.
    serve = parser.add_mutually_exclusive_group()
    serve.add_argument(
        "--serve",
        action="store_true",
        help="serve the pages (this is the default; accepted for symmetry with "
             "re_review_timeline --serve)",
    )
    serve.add_argument(
        "--no-serve",
        action="store_true",
        help="generate the three pages and exit instead of serving them",
    )
    parser.add_argument(
        "--no-consistency",
        action="store_true",
        help="skip the map/image consistency check and its overlay",
    )
    parser.add_argument("--consistency-out", default=CONSISTENCY_JSON, type=Path)
    parser.add_argument("--timeline", default=TIMELINE_HTML, type=Path)
    parser.add_argument("--frame-view", default=FRAME_HTML, type=Path)
    parser.add_argument("--map-view", default=MAP_HTML, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


def run(args: argparse.Namespace) -> None:
    root = args.dataset_root.resolve()
    review_out = (
        args.review_out if args.review_out.is_absolute() else root / args.review_out
    )
    draft_out = draft_path_for(review_out)

    # Show the reviewed output when there is one, so reopening the workspace
    # continues from what the reviewer last confirmed. The draft is deliberately
    # not used here: the views exist to check committed results.
    initial_review = review_out if review_out.exists() else None
    build_views(args, initial_review)

    def refresh_views(committed: Path) -> None:
        print("refreshing views against the committed review...", flush=True)
        build_views(args, committed)

    if args.no_serve:
        # Still generate the timeline; --no-serve makes it a plain file:// page,
        # which has no save endpoint and so no commit hook to wire.
        re_review_timeline.run(re_review_timeline.parse_args(timeline_argv(args)))
        print(f"draft would be {draft_out}", flush=True)
        return

    re_review_timeline.run(
        re_review_timeline.parse_args(timeline_argv(args)), on_commit=refresh_views
    )


if __name__ == "__main__":
    main()
