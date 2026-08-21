#!/usr/bin/env python3
"""Combined TLR autolabeling: detector -> per-ROI YOLOX classifier (LampRecognizer).

full image ─> detector (find traffic_light boxes, NMS)
           ─> for each box: crop ─> lamp recognizer (color + shape + arrow)
           ─> annotated image (--viz) + JSON labels (--out-dir)

Two detector families are supported; the type is auto-detected from the ONNX
output count:
  1 output  -> YOLOX traffic_light detector (tensorrt_yolox style: BGR, no /255,
               grid/stride decode, num_class=1)
  3 outputs -> CoMLOps-Large-Detection-Model (darknet YOLO style: RGB, /255,
               5 anchors x (4 box + obj + 10 cls) per scale, strides 8/16/32,
               bw = pw*(2*sig(tw))^2; anchors from configs/model_params/comlops_large_detector_ml.param.yaml,
               empirically fitted -- see that yaml's header).
               Kept classes default to TRAFFIC_LIGHT only (--det-classes).

Reuses the faithful classifier decode from tlr_autolabel.inference.lamp_recognizer_onnx.

Usage:
  # 1) verification pass on a few images, write annotated pngs to eyeball
  python3 scripts/tlr_autolabel.py <image_or_dir> --viz --out-dir ./out

  # 2) once inference looks right, run over the dataset for labels
  python3 scripts/tlr_autolabel.py <dir> --out-dir ./labels

  # detector selection: named preset (configs/detectors/*.yaml) or explicit path
  python3 scripts/tlr_autolabel.py <dir> --preset yolox-1920-int8
  python3 scripts/tlr_autolabel.py <dir> --detector .../CoMLOps-Large-Detection-Model-v1.0.1.onnx
"""
import argparse
import glob
import json
import os
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cv2

from tlr_autolabel.core.models import load_model_manifest, model_provenance
from tlr_autolabel.inference.detector import (
    Detector,
    clipped_detection_box,
    detect_full_and_tiles,
)
from tlr_autolabel.inference.lamp_recognizer import LampClassifier, normalize_lamps, signal_state

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def process_image_with_candidates(img, detector, classifier, args):
    ih, iw = img.shape[:2]
    score_thr = (args.det_low_score_thr
                 if args.det_low_score_thr is not None
                 else args.det_score_thr)
    boxes = detect_full_and_tiles(detector, img, args, score_thr=score_thr)

    results = []
    raw_detections = []
    for b in boxes:
        box_xyxy = clipped_detection_box(b, iw, ih, args.min_box)
        if box_xyxy is None:
            continue
        is_high = b["prob"] >= args.det_score_thr
        lamps = []
        if is_high or args.classify_low_detections:
            lamps = normalize_lamps(classifier.classify(img, box_xyxy))
        candidate = {
            "detector_score": round(b["prob"], 4),
            "box_xyxy": list(box_xyxy),
            "lamps": lamps,
            "state": signal_state(lamps),
            "detection_level": "high" if is_high else "low",
        }
        raw_detections.append(dict(candidate))
        if is_high:
            if args.drop_unknown and not lamps:
                continue
            results.append({
                "detector_score": candidate["detector_score"],
                "box_xyxy": candidate["box_xyxy"],
                "lamps": candidate["lamps"],
                "state": candidate["state"],
            })
    return results, raw_detections


def process_image(img, detector, classifier, args):
    return process_image_with_candidates(img, detector, classifier, args)[0]


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


PRESET_DIR = os.path.join(REPO_ROOT, "configs", "detectors")

# Search order for the ML model root, matching Autoware launch's `data_path`
# default at the end so presets resolve the same on a deployed machine.
MODEL_ROOT_CANDIDATES = ["~/autoware_data", "/opt/autoware/mlmodels"]


def autoware_mlmodels_root():
    """Resolve the standard Autoware model-store mount without changing the
    older TLR_MODEL_ROOT search order used by existing presets."""
    env = os.environ.get("AUTOWARE_MLMODELS") or os.environ.get("AUTOWARE_MLMODELS_ROOT")
    return os.path.expanduser(env) if env else "/opt/autoware/mlmodels"


def model_root():
    """Resolve the model root: $TLR_MODEL_ROOT if set, else the first existing
    candidate. Presets reference models as ${TLR_MODEL_ROOT}/..., so this keeps
    them free of machine-specific absolute paths."""
    env = os.environ.get("TLR_MODEL_ROOT")
    if env:
        return os.path.expanduser(env)
    for cand in MODEL_ROOT_CANDIDATES:
        p = os.path.expanduser(cand)
        if os.path.isdir(p):
            return p
    return os.path.expanduser(MODEL_ROOT_CANDIDATES[0])


def expand_path(val):
    """Expand ~, $VARS, ${TLR_MODEL_ROOT}, and ${AUTOWARE_MLMODELS} in a
    preset/CLI path string."""
    return os.path.expanduser(os.path.expandvars(
        val.replace("${TLR_MODEL_ROOT}", model_root())
           .replace("$TLR_MODEL_ROOT", model_root())
           .replace("${AUTOWARE_MLMODELS}", autoware_mlmodels_root())
           .replace("$AUTOWARE_MLMODELS", autoware_mlmodels_root())))


def list_presets():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(PRESET_DIR)
                  if f.endswith(".yaml")) if os.path.isdir(PRESET_DIR) else []


def apply_preset(ap, args):
    """Overlay preset values onto args, but only where the user did not pass an
    explicit CLI flag (explicit CLI always beats the preset)."""
    import yaml
    path = os.path.join(PRESET_DIR, args.preset + ".yaml")
    if not os.path.exists(path):
        raise SystemExit(f"unknown preset {args.preset!r}; available: {', '.join(list_presets())}")
    with open(path) as f:
        preset = yaml.safe_load(f)
    defaults = {a.dest: a.default for a in ap._actions}
    for key, val in preset.items():
        dest = key.replace("-", "_")
        if dest not in defaults:
            raise SystemExit(f"preset {args.preset}: unknown key {key!r}")
        if isinstance(val, str):
            val = expand_path(val)
        if getattr(args, dest) == defaults[dest]:
            setattr(args, dest, val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="image file or directory")
    ap.add_argument("--preset", default=None,
                    help="named detector preset from configs/detectors/ "
                         f"(available: {', '.join(list_presets())})")
    ap.add_argument("--detector", default=None,
                    help="detector model path (.onnx or .engine); overrides the preset's")
    ap.add_argument("--detector-type", default=None,
                    help="detector family (e.g. yolox, comlops); overrides the preset "
                         "and the .onnx output-count auto-detect")
    ap.add_argument("--classifier", default=os.path.join(REPO_ROOT, "models", "traffic_light_lamp_recognizer_comlops.onnx"))
    ap.add_argument("--classifier-param", default=os.path.join(REPO_ROOT, "configs", "model_params", "lamp_recognizer_ml.param.yaml"))
    ap.add_argument("--comlops-param", default=os.path.join(REPO_ROOT, "configs", "model_params", "comlops_large_detector_ml.param.yaml"),
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
    ap.add_argument("--run-id", default=None,
                    help="identifier stored in meta.run_id (default: timestamp + random)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip images whose <name>.json already exists in --out-dir "
                         "(resume an interrupted batch)")
    args = ap.parse_args()

    if args.preset:
        apply_preset(ap, args)
    if args.no_tiles:
        args.tiles = False
    if not args.detector:
        raise SystemExit("choose a detector: --preset <name> "
                         f"(available: {', '.join(list_presets())}) or --detector <model path>")
    if args.det_low_score_thr is not None and args.det_low_score_thr > args.det_score_thr:
        raise SystemExit("--det-low-score-thr must be <= --det-score-thr")
    # model paths given directly may also use ~ / $VARS / ${TLR_MODEL_ROOT}
    args.detector = expand_path(args.detector)
    args.classifier = expand_path(args.classifier)
    args.classifier_param = expand_path(args.classifier_param)
    args.comlops_param = expand_path(args.comlops_param)
    if not os.path.exists(args.detector):
        raise SystemExit(
            f"detector model not found: {args.detector}"
            + (f" (from preset {args.preset})" if args.preset else "")
            + f"\nmodel root = {model_root()} "
            "(set $TLR_MODEL_ROOT to point at your ML model directory).")
    if not os.path.exists(args.classifier):
        raise SystemExit(
            f"classifier model not found: {args.classifier}"
            + (f" (from preset {args.preset})" if args.preset else "")
            + f"\nmodel root = {model_root()} "
            "(set $TLR_MODEL_ROOT to point at your ML model directory).")
    if not os.path.exists(args.classifier_param):
        raise SystemExit(
            f"classifier param not found: {args.classifier_param}"
            + (f" (from preset {args.preset})" if args.preset else ""))

    detector = Detector(args.detector, args.comlops_param, model_type=args.detector_type)
    if detector.kind == "comlops":
        detector.set_keep_classes([s.strip() for s in args.det_classes.split(",") if s.strip()])

    classifier = LampClassifier(args.classifier, args.classifier_param, args)

    # model provenance: record the exact bytes used, resolve to a known-good name,
    # and flag an unlisted .onnx (silent drift detection). Engines are machine-
    # specific and intentionally unlisted, so they are not flagged.
    manifest = load_model_manifest()
    det_prov = model_provenance(args.detector, manifest)
    cls_prov = model_provenance(args.classifier, manifest)
    for role, path, prov in (("detector", args.detector, det_prov),
                             ("classifier", args.classifier, cls_prov)):
        if prov["sha256"] and not prov["known"] and path.endswith(".onnx"):
            warnings.warn(
                f"{role} {os.path.basename(path)} (sha256 {prov['sha256'][:12]}) "
                "is not in the configs/models.yaml known-good registry; add it "
                "if this model is intended.", stacklevel=2)

    backend = ("tensorrt-engine" if detector.sess is None
               else str(detector.sess.get_providers()))
    print(f"detector={os.path.basename(args.detector)} [{detector.kind}] "
          f"in={detector.w}x{detector.h} backend={backend}")
    print(f"classifier={os.path.basename(args.classifier)} "
          f"in={classifier.width}x{classifier.height} backend={classifier.backend}")

    if os.path.isdir(args.image):
        paths = sorted(sum([glob.glob(os.path.join(args.image, e))
                            for e in ("*.png", "*.jpg", "*.jpeg", "*.bmp")], []))
    else:
        paths = [args.image]

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    # image paths in the JSON are relative to image_root (portable across
    # machines/containers); the realpath is kept alongside as a convenience.
    if args.t4_dataset and not args.image_root:
        image_root = os.path.realpath(args.t4_dataset)
    else:
        image_root = os.path.realpath(
            args.image_root or (args.image if os.path.isdir(args.image)
                                else os.path.dirname(args.image) or "."))
    t4map = {}
    if args.t4_dataset:
        with open(os.path.join(args.t4_dataset, "annotation", "sample_data.json")) as f:
            for sd in json.load(f):
                key = os.path.realpath(os.path.join(args.t4_dataset, sd["filename"]))
                t4map[key] = sd

    # provenance block written into every per-image JSON (schema tlr_autolabel/v1)
    run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                             + "-" + uuid.uuid4().hex[:8])
    meta = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preset": args.preset,
        "detector": os.path.basename(args.detector),
        "detector_backend": backend,
        "detector_sha256": det_prov["sha256"],
        "detector_model": det_prov["model"],
        "model_root": model_root(),
        "classifier": os.path.basename(args.classifier),
        "classifier_backend": classifier.backend,
        "classifier_sha256": cls_prov["sha256"],
        "classifier_model": cls_prov["model"],
        "tiles": bool(args.tiles),
        "det_score_thr": args.det_score_thr,
        "det_low_score_thr": args.det_low_score_thr,
        "classify_low_detections": bool(args.classify_low_detections),
        "det_nms_thr": args.det_nms_thr,
        "cls_score_thr": args.cls_score_thr,
        "cls_nms_thr": args.cls_nms_thr,
        "min_box": args.min_box,
        "crop_pad": args.crop_pad,
    }

    for seq, p in enumerate(paths):
        name = os.path.splitext(os.path.basename(p))[0]
        if args.skip_existing and args.out_dir and \
                os.path.exists(os.path.join(args.out_dir, name + ".json")):
            continue
        img = cv2.imread(p)
        if img is None:
            print(f"[skip] {p}")
            continue
        results, raw_detections = process_image_with_candidates(img, detector, classifier, args)
        name = os.path.splitext(os.path.basename(p))[0]
        print(f"\n=== {os.path.basename(p)} ({img.shape[1]}x{img.shape[0]}): "
              f"{len(results)} signal(s) ===")
        for r in results:
            lamp_str = ", ".join("{}({:.2f})".format(l["label"], l["confidence"]) for l in r["lamps"])
            print(f"  [{r['state']}] det={r['detector_score']:.2f} box={r['box_xyxy']} lamps=[{lamp_str}]")
        if args.out_dir:
            rp = os.path.realpath(p)
            rel = os.path.relpath(rp, image_root)
            sd = t4map.get(rp)
            # channel: from any CAM_*/etc. path component; frame_index: numeric
            # stem when the file is numbered (T4 style), else sequence order
            channel = next((c for c in rel.split(os.sep)[:-1]
                            if c.upper() == c and not c.startswith(".")), None)
            frame_index = int(name) if name.isdigit() else seq
            for i, r in enumerate(results):
                results[i] = {"signal_id": f"{name}-{i:02d}", **r}
            if args.det_low_score_thr is not None:
                for i, r in enumerate(raw_detections):
                    raw_detections[i] = {"raw_detection_id": f"{name}-raw-{i:02d}", **r}
            payload = {"schema_version": "tlr_autolabel/v1",
                       "image": rel,
                       "image_realpath": rp,
                       "sample_data_token": sd["token"] if sd else None,
                       "channel": channel,
                       "frame_index": frame_index,
                       "width": img.shape[1], "height": img.shape[0],
                       "meta": meta,
                       "signals": results}
            if args.det_low_score_thr is not None:
                payload["raw_detections"] = raw_detections
            with open(os.path.join(args.out_dir, name + ".json"), "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            if args.viz:
                cv2.imwrite(os.path.join(args.out_dir, name + ".viz.png"), draw(img, results))


if __name__ == "__main__":
    main()
