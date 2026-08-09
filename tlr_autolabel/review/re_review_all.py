#!/usr/bin/env python3
"""Generate all three review views and serve them from one command.

The views are separate tools so each can be debugged on its own, but the
normal workflow uses them together: correct annotations in the timeline, then
check the result per image and on the map. This wires that loop up --
including regenerating the read-only views against the committed review after
every commit, so "Frame view" and "Map view" show what the corrections
actually produced rather than the raw detector output.

Each underlying tool keeps its own entry point:

    python3 -m tlr_autolabel.review.re_review_timeline --dataset-root <ds> --serve
    python3 -m tlr_autolabel.review.re_frame_view      --dataset-root <ds>
    python3 -m tlr_autolabel.review.re_map_view        --dataset-root <ds>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tlr_autolabel.review import re_frame_view, re_map_view, re_review_timeline
from tlr_autolabel.review.re_review_timeline import (
    DEFAULT_CROP_CHANNELS,
    draft_path_for,
)

TIMELINE_HTML = Path("build/tl_match/re_review_timeline.html")
FRAME_HTML = Path("build/tl_match/re_frame_view.html")
MAP_HTML = Path("build/tl_match/re_map_view.html")


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


def build_views(args: argparse.Namespace, review: Path | None) -> None:
    """(Re)generate the two read-only views, optionally against a review."""
    re_frame_view.run(re_frame_view.parse_args(
        view_argv(args, review)
        + ["--output", str(args.frame_view),
           "--map-view", str(args.map_view),
           "--timeline", str(args.timeline)]
    ))
    re_map_view.run(re_map_view.parse_args(
        view_argv(args, review)
        + ["--output", str(args.map_view),
           "--map", str(args.map),
           "--context-radius", str(args.context_radius),
           "--frame-view", str(args.frame_view),
           "--timeline", str(args.timeline)]
    ))


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
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="generate the three pages and exit instead of serving them",
    )
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
