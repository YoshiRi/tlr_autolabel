#!/usr/bin/env python3
"""Export tlr_autolabel per-image JSONs to COCO or CVAT.

The per-image JSON written by tlr_autolabel.py --out-dir stays the source of
truth (it keeps everything: xyxy boxes, detector score, per-lamp color/shape/
arrow with confidences). This tool converts a directory of them to either
interchange format, so the choice of annotation tool stays reversible:

  # COCO (single labels.json; attributes kept on each annotation)
  python3 export_labels.py <labels_dir> --format coco --out labels_coco.json \
      [--image-root <dataset_dir>] [--category-mode single|state]

  # CVAT for images 1.1 XML (state/confidence as CVAT attributes)
  python3 export_labels.py <labels_dir> --format cvat --out annotations.xml \
      [--image-root <dataset_dir>]

--image-root makes file names relative to it (recommended: the dataset root, so
COCO file_name / CVAT image name match the paths the tool will see). Without it
only the basename is used.

CVAT import expects the task to define the label `traffic_light` with (text)
attributes `state` and `confidence`; with --category-mode state the COCO export
instead uses one category per distinct signal string (for training a
state-aware detector directly).
"""
import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET


def sig_state(s):
    """Canonical state; falls back to the pre-2026-07-18 'signal' key."""
    return s.get("state", s.get("signal", "unknown"))


def load_frames(labels_dir, image_root):
    frames = []
    for p in sorted(glob.glob(os.path.join(labels_dir, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        if not all(k in d for k in ("image", "width", "height", "signals")):
            continue  # not a tlr_autolabel per-image json
        img = d["image"]
        if os.path.isabs(img):  # pre-v1 files stored the realpath here
            name = os.path.relpath(img, image_root) if image_root else os.path.basename(img)
        else:  # v1: already relative to the run's image root
            name = img
        frames.append((name, d))
    frames.sort(key=lambda t: t[0])
    return frames


def export_coco(frames, out_path, category_mode):
    if category_mode == "single":
        cats = {"traffic_light": 1}
    else:  # one category per signal state string
        names = sorted({sig_state(s) for _, d in frames for s in d["signals"]})
        cats = {n: i + 1 for i, n in enumerate(names)}
    images, annotations = [], []
    for img_id, (name, d) in enumerate(frames, start=1):
        images.append({"id": img_id, "file_name": name,
                       "width": d["width"], "height": d["height"]})
        for s in d["signals"]:
            x0, y0, x1, y1 = s["box_xyxy"]
            cat = cats["traffic_light"] if category_mode == "single" \
                else cats[sig_state(s)]
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": img_id,
                "category_id": cat,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
                "score": s["detector_score"],
                "attributes": {"state": sig_state(s), "lamps": s["lamps"]},
            })
    coco = {
        "info": {"description": "tlr_autolabel export", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": i, "name": n, "supercategory": "traffic_light"}
                       for n, i in sorted(cats.items(), key=lambda t: t[1])],
        "images": images,
        "annotations": annotations,
    }
    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2, ensure_ascii=False)
    return len(images), len(annotations)


def export_cvat(frames, out_path):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    n_boxes = 0
    for img_id, (name, d) in enumerate(frames):
        im = ET.SubElement(root, "image", id=str(img_id), name=name,
                           width=str(d["width"]), height=str(d["height"]))
        for s in d["signals"]:
            x0, y0, x1, y1 = s["box_xyxy"]
            box = ET.SubElement(im, "box", label="traffic_light",
                                source="auto", occluded="0", z_order="0",
                                xtl=f"{x0:.2f}", ytl=f"{y0:.2f}",
                                xbr=f"{x1:.2f}", ybr=f"{y1:.2f}")
            ET.SubElement(box, "attribute", name="state").text = sig_state(s)
            ET.SubElement(box, "attribute", name="confidence").text = \
                f"{s['detector_score']:.4f}"
            n_boxes += 1
    ET.indent(root)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    return len(frames), n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir", help="directory of tlr_autolabel per-image JSONs")
    ap.add_argument("--format", required=True, choices=["coco", "cvat"])
    ap.add_argument("--out", required=True, help="output file path")
    ap.add_argument("--image-root", default=None,
                    help="make image names relative to this directory")
    ap.add_argument("--category-mode", default="single", choices=["single", "state"],
                    help="coco only: one traffic_light category, or one per signal state")
    args = ap.parse_args()

    frames = load_frames(args.labels_dir, args.image_root)
    if not frames:
        raise SystemExit(f"no per-image label JSONs found in {args.labels_dir}")
    if args.format == "coco":
        ni, na = export_coco(frames, args.out, args.category_mode)
    else:
        ni, na = export_cvat(frames, args.out)
    print(f"{args.format}: wrote {args.out} ({ni} images, {na} boxes)")


if __name__ == "__main__":
    main()
