#!/usr/bin/env python3
"""Combined TLR autolabeling: detector -> per-ROI YOLOX classifier (LampRecognizer).

frames  ─> detector (find traffic_light boxes, NMS)
        ─> for each box: crop ─> lamp recognizer (color + shape + arrow)
        ─> annotated image (--viz) + JSON labels (--out-dir)

Detector and classifier families are plug-ins (`detector_type` /
`classifier_type`, see docs/model_interface.md); the shipped families are the
YOLOX traffic-light detector, the CoMLOps darknet detector, and the
LampRecognizer classifier.

Frames come from a frame source (docs/inference_comparison.md), not necessarily
a directory: images, a video, a rosbag, or a T4 dataset all feed the same
pipeline and produce the same Tier A `tlr_autolabel/v1` output.

Usage:
  # 1) verification pass on a few images, write annotated pngs to eyeball
  python3 scripts/tlr_autolabel.py <image_or_dir> --viz --out-dir ./out

  # 2) once inference looks right, run over the dataset for labels
  python3 scripts/tlr_autolabel.py <dir> --out-dir ./labels

  # detector selection: named preset (configs/detectors/*.yaml) or explicit path
  python3 scripts/tlr_autolabel.py <dir> --preset yolox-1920-int8
  python3 scripts/tlr_autolabel.py <dir> --detector .../CoMLOps-Large-Detection-Model-v1.0.1.onnx

  # other inputs (the image argument is then unused/optional)
  python3 scripts/tlr_autolabel.py --video drive.mp4 --frame-stride 5 --preset yolox-960-int8 --out-dir ./labels
  python3 scripts/tlr_autolabel.py --bag ./rosbag2_2026 --preset yolox-960-int8 --out-dir ./labels

  # compare several detector/classifier combinations instead of one:
  python3 scripts/run_compare.py --matrix configs/compare/<name>.yaml --out ./build/compare
"""
import argparse
import json
import os
from pathlib import Path

import cv2

from tlr_autolabel.frames import build_frame_source
from tlr_autolabel.inference.config import (
    DEFAULT_CLASSIFIER, DEFAULT_CLASSIFIER_PARAM, DEFAULT_COMLOPS_PARAM,
    autoware_mlmodels_root, config_from_args, expand_path, list_presets,
    load_preset, model_root,
)
from tlr_autolabel.inference.detector import Detector
from tlr_autolabel.inference.lamp_recognizer import LampClassifier, normalize_lamps, signal_state
from tlr_autolabel.inference.models import list_classifiers, list_detectors
from tlr_autolabel.inference.pipeline import (
    Pipeline, new_run_id, process_image, process_image_with_candidates,
)

REPO_ROOT = str(Path(__file__).resolve().parents[2])

# re-exported for callers that used to import them from here
__all__ = ["main", "draw", "process_image", "process_image_with_candidates",
           "expand_path", "model_root", "autoware_mlmodels_root", "list_presets",
           "load_preset", "Detector", "LampClassifier", "normalize_lamps",
           "signal_state", "list_detectors", "list_classifiers"]


def draw(img, results):
    vis = img.copy()
    color_bgr = {"green": (0, 200, 0), "amber": (0, 200, 255),
                 "red": (0, 0, 255), None: (200, 200, 200), "unknown": (200, 200, 200)}
    for r in results:
        x0, y0, x1, y1 = r["box_xyxy"]
        main = r["lamps"][0]["color"] if r["lamps"] else None
        c = color_bgr.get(main, (200, 200, 200))
        cv2.rectangle(vis, (x0, y0), (x1, y1), c, 2)
        text = f"{r['state']} {r['detector_score']:.2f}"
        ty = y0 - 6 if y0 - 6 > 10 else y1 + 16
        cv2.putText(vis, text, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return vis


def add_frame_source_args(ap):
    """Input selection, shared with scripts/run_compare.py."""
    ap.add_argument("--video", default=None, help="read frames from a video file")
    ap.add_argument("--bag", default=None,
                    help="read camera images from a rosbag2 (mcap/sqlite3); needs a sourced ROS 2")
    ap.add_argument("--bag-topics", default=None,
                    help="comma-separated image topics (default: every image topic in the bag)")
    ap.add_argument("--channels", default=None,
                    help="with --t4-dataset: comma-separated camera channels (default: all)")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="keep every Nth frame of a video/bag (default: 1)")
    ap.add_argument("--frame-start", type=int, default=0,
                    help="first video frame number to read")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames (per topic for a bag)")


def frame_source_spec(args, *, prefer_t4_source=False):
    """Map the input flags onto a `build_frame_source` spec."""
    if args.video:
        return {"kind": "video", "uri": args.video, "stride": args.frame_stride,
                "start": args.frame_start, "max_frames": args.max_frames}
    if args.bag:
        return {"kind": "rosbag", "uri": args.bag, "topics": args.bag_topics,
                "stride": args.frame_stride, "max_frames": args.max_frames}
    if prefer_t4_source and args.t4_dataset and not getattr(args, "image", None):
        return {"kind": "t4", "root": args.t4_dataset, "channels": args.channels}
    if not getattr(args, "image", None):
        raise SystemExit("no input: pass an image/directory, --video, --bag, or --t4-dataset")
    return {"kind": "images", "path": args.image, "image_root": args.image_root,
            "t4_dataset": args.t4_dataset}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=None, help="image file or directory")
    ap.add_argument("--preset", default=None,
                    help="named detector preset from configs/detectors/ "
                         f"(available: {', '.join(list_presets())})")
    ap.add_argument("--detector", default=None,
                    help="detector model path (.onnx or .engine); overrides the preset's")
    ap.add_argument("--detector-type", default=None,
                    help="detector family (e.g. yolox, comlops); overrides the preset "
                         "and the .onnx output-count auto-detect")
    ap.add_argument("--classifier", default=DEFAULT_CLASSIFIER,
                    help="classifier model path, or 'none' for a detector-only run")
    ap.add_argument("--classifier-type", default=None,
                    help="classifier family (default: lamp_recognizer)")
    ap.add_argument("--classifier-param", default=DEFAULT_CLASSIFIER_PARAM)
    ap.add_argument("--comlops-param", default=DEFAULT_COMLOPS_PARAM,
                    help="decode params (anchors etc.) for the CoMLOps darknet detector")
    ap.add_argument("--det-classes", default="TRAFFIC_LIGHT",
                    help="comma-separated class names kept from the CoMLOps detector")
    # Score default matches the ROS node (0.35). An earlier 0.5 was used on the
    # theory that offline has no map_based_detector ROI prior to filter false
    # positives -- but L3 (match_traffic_lights) DOES match against the map, so
    # the map filters FPs after the fact. Measured on CAM_FRONT (L1920+tiles):
    # dropping 0.5->0.35 recovered +30 real matched signals for only +6 unmatched
    # (FP-ish), mostly mid-range; going to 0.2 reached coverage 0.60. So run the
    # detector at node-parity recall and let the map matching sort out FPs.
    # nms stays tighter than the node's 0.7 to merge duplicate boxes on one signal.
    ap.add_argument("--det-score-thr", type=float, default=0.35)
    ap.add_argument("--det-low-score-thr", type=float, default=None,
                    help="Optional low detector threshold for temporal tracking candidates. "
                         "When set below --det-score-thr, detections between the two "
                         "thresholds are written to raw_detections but do not enter "
                         "signals, so they cannot create new L1 labels.")
    ap.add_argument("--classify-low-detections", action="store_true",
                    help="Also run the classifier on low-threshold raw_detections. "
                         "By default low candidates keep only bbox/score so L3 can "
                         "first filter them by temporal/map association.")
    ap.add_argument("--det-nms-thr", type=float, default=0.35)
    ap.add_argument("--cls-score-thr", type=float, default=0.2)
    ap.add_argument("--cls-nms-thr", type=float, default=0.2)
    ap.add_argument("--min-box", type=float, default=8.0,
                    help="drop detections whose shorter side (px, original image) is below this")
    ap.add_argument("--drop-unknown", action="store_true",
                    help="drop signals whose classifier found no lamp (signal=='unknown')")
    ap.add_argument("--crop-pad", type=float, default=0.0, help="ROI padding ratio before classifying")
    ap.add_argument("--tiles", action="store_true",
                    help="add native-resolution tile passes (detector-input-sized crops "
                         "covering the image) on top of the full-frame pass; recovers "
                         "small distant signals lost to the letterbox downscale")
    ap.add_argument("--no-tiles", action="store_true",
                    help="force tile passes off (overrides a preset's tiles: true)")
    ap.add_argument("--tile-overlap", type=int, default=128,
                    help="minimum overlap in px between neighbouring tiles")
    ap.add_argument("--out-dir", default=None, help="write <name>.json (and <name>.viz.png with --viz)")
    ap.add_argument("--viz", action="store_true", help="save annotated images")
    ap.add_argument("--image-root", default=None,
                    help="write image paths relative to this dir (default: the input dir; "
                         "with --t4-dataset, the dataset root)")
    ap.add_argument("--t4-dataset", default=None,
                    help="T4 dataset root: fills sample_data_token/channel from "
                         "annotation/sample_data.json and uses it as image root")
    add_frame_source_args(ap)
    ap.add_argument("--timing", action="store_true",
                    help="record per-frame timing_ms (detector/classifier/total) in Tier A")
    ap.add_argument("--model-digest", action="store_true",
                    help="record model sha256 in meta (two runs are only comparable "
                         "if these match)")
    ap.add_argument("--run-id", default=None,
                    help="identifier stored in meta.run_id (default: timestamp + random)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip images whose <name>.json already exists in --out-dir "
                         "(resume an interrupted batch)")
    args = ap.parse_args()

    cfg = config_from_args(ap, args, extra_overrides={
        **({"tiles": False} if args.no_tiles else {}),
        "record_timing": bool(args.timing),
        "record_model_digest": bool(args.model_digest),
    })
    source = build_frame_source(frame_source_spec(args, prefer_t4_source=True))

    pipeline = build_pipeline(cfg, run_id=args.run_id or new_run_id())
    print(pipeline.describe())

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    def already_done(frame_id):
        return bool(args.skip_existing and args.out_dir and
                    os.path.exists(os.path.join(args.out_dir, frame_id + ".json")))

    for frame in source.iter_frames(skip=already_done):
        payload = pipeline.run(frame)
        w, h = frame.size
        print(f"\n=== {os.path.basename(frame.rel_path)} ({w}x{h}): "
              f"{len(payload['signals'])} signal(s) ===")
        for r in payload["signals"]:
            lamp_str = ", ".join("{}({:.2f})".format(l["label"], l["confidence"])
                                 for l in r["lamps"])
            print(f"  [{r['state']}] det={r['detector_score']:.2f} "
                  f"box={r['box_xyxy']} lamps=[{lamp_str}]")
        if args.out_dir:
            out_path = os.path.join(args.out_dir, frame.frame_id + ".json")
            os.makedirs(os.path.dirname(out_path) or args.out_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            if args.viz:
                cv2.imwrite(out_path[: -len(".json")] + ".viz.png",
                            draw(frame.image, payload["signals"]))
    pipeline.close()


def build_pipeline(cfg, run_id=""):
    """Build the models through this module's names, so the CLI smoke test can
    keep monkeypatching `Detector` / `LampClassifier` here."""
    from tlr_autolabel.inference.config import validate_config

    validate_config(cfg)
    detector = Detector(cfg.detector, cfg.comlops_param, model_type=cfg.detector_type)
    if detector.kind == "comlops":
        detector.set_keep_classes(
            [s.strip() for s in cfg.det_classes.split(",") if s.strip()])
    classifier = None
    if cfg.classifier_enabled:
        classifier = LampClassifier(cfg.classifier, cfg.classifier_param, cfg,
                                    model_type=cfg.classifier_type)
    return Pipeline(cfg=cfg, detector=detector, classifier=classifier, run_id=run_id)


if __name__ == "__main__":
    main()
