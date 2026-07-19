#!/usr/bin/env python3
"""Run the full autolabel pipeline (L1 -> L3 + timeline) over T4 datasets.

For each dataset root:
  1. L1 per camera channel: tlr_autolabel.py (via run_gpu.sh when present)
     data/<CH> -> tlr_autolabel/<CH>/<frame>.json   (--skip-existing resume)
  2. match_traffic_lights.py       -> annotation/traffic_signal_2d_ann.json
  3. aggregate_regulatory_signals.py -> annotation/traffic_signal_re_timeseries.json
  4. render_re_timeline.py         -> build/tl_match/re_timeline.html

Usage:
  python3 run_dataset.py <dataset_root> [<dataset_root> ...] \
      [--preset yolox-1920-int8] [--run-id ID] [--channels CAM_FRONT,CAM_...] \
      [--skip-l1] [--l1-args "--viz"]

Channels default to every data/* subdirectory that contains images.
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def l1_runner():
    wrapper = os.path.join(HERE, "run_gpu.sh")
    if os.path.exists(wrapper):
        return [wrapper]
    return [sys.executable, os.path.join(HERE, "tlr_autolabel.py")]


def camera_channels(root):
    chans = []
    for d in sorted(glob.glob(os.path.join(root, "data", "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.jpg")):
            chans.append(os.path.basename(d))
    return chans


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="+", help="T4 dataset roots")
    ap.add_argument("--preset", default="yolox-1920-int8")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--channels", default=None,
                    help="comma-separated channel names (default: all data/* with jpgs)")
    ap.add_argument("--skip-l1", action="store_true",
                    help="labels already exist; run only match/aggregate/render")
    ap.add_argument("--l1-args", default="",
                    help="extra args passed through to tlr_autolabel.py")
    args = ap.parse_args()

    for root in args.datasets:
        root = os.path.realpath(root)
        print(f"=== dataset: {root} ===", flush=True)
        if not args.skip_l1:
            chans = (args.channels.split(",") if args.channels
                     else camera_channels(root))
            if not chans:
                print("  no camera channels found, skipping dataset")
                continue
            for ch in chans:
                cmd = l1_runner() + [
                    os.path.join(root, "data", ch),
                    "--preset", args.preset,
                    "--t4-dataset", root,
                    "--out-dir", os.path.join(root, "tlr_autolabel", ch),
                    "--skip-existing",
                ]
                if args.run_id:
                    cmd += ["--run-id", args.run_id]
                cmd += args.l1_args.split()
                sh(cmd)
        for tool in ("match_traffic_lights.py",
                     "aggregate_regulatory_signals.py",
                     "render_re_timeline.py"):
            sh([sys.executable, os.path.join(HERE, tool), "--dataset-root", root])
        print(f"=== done: {root} (timeline: build/tl_match/re_timeline.html) ===",
              flush=True)


if __name__ == "__main__":
    main()
