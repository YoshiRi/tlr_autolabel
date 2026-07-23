#!/usr/bin/env python3
"""AWML adapter: build a derived T4 training dataset from tlr_autolabel v1 JSONs.

Generates a training-only *derived view* of a T4 dataset for AWML's TLR
pipeline (tools/detection2d/create_data_t4dataset.py):

  derived/
    data/, input_bag/, map/, ...      -> symlinks into the source dataset
    annotation/*.json                 -> symlinks, EXCEPT:
    annotation/object_ann.json        -> generated: one 2D box per autolabel
                                         signal, category = db_tlr state name
    annotation/category.json          -> source categories + the db_tlr state
                                         categories actually used (merged)

The canonical dataset is never edited in place, and the canonical annotation
formats (tlr_autolabel/v1, traffic_signal_2d/v1) stay upstream: this is a
one-way, lossy projection into the db_tlr vocabulary defined in
configs/state_vocab/db_tlr.yaml (names outside that vocabulary become
`unknown`).

Usage:
  python3 export_awml.py <labels_dir> --t4-dataset <src_root> --out <derived_dir>

Requires the v1 JSONs to carry sample_data_token (i.e. the L1 run used
--t4-dataset). Frames without a token are skipped with a warning.
"""
import argparse
import glob
import hashlib
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(HERE, "..", "configs", "state_vocab", "db_tlr.yaml")


def load_vocab():
    with open(VOCAB_PATH) as f:
        return yaml.safe_load(f)


def db_tlr_state(lamps, vocab):
    """Project canonical lamps onto a db_tlr category name (lossy, one-way)."""
    if not lamps:
        return "unknown"
    if any(l["shape"] == "ped" for l in lamps):
        color = next((l["color"] for l in lamps if l["shape"] == "ped"), None)
        name = {"red": "crosswalk_red", "green": "crosswalk_green"}.get(color,
                                                                        "crosswalk_unknown")
        return name if name in vocab["allowed"] else "unknown"
    colors = [vocab["color_names"].get(l["color"])
              for l in lamps if l["shape"] == "circle"]
    arrows = [vocab["arrow_names"].get(l["arrow"])
              for l in lamps if l["shape"] == "arrow"]
    if None in colors or None in arrows:      # unmapped color/direction (e.g. down)
        return "unknown"
    parts = sorted(set(colors)) + [a for a in vocab["arrow_order"]
                                   if a in set(arrows)]
    if not parts:
        return "unknown"                      # only u_turn/number/cross lamps
    name = "_".join(parts)
    return name if name in vocab["allowed"] else "unknown"


def token_of(*parts):
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def main():
    import sys
    print("DEPRECATED: export_awml.py is superseded by to_object_ann.py + t4devkit AWML tooling "
          "(see deprecated/README.md). Running anyway.", file=sys.stderr)
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir", help="directory of tlr_autolabel v1 per-image JSONs")
    ap.add_argument("--t4-dataset", required=True, help="source T4 dataset root")
    ap.add_argument("--out", required=True, help="derived dataset directory to create")
    args = ap.parse_args()

    vocab = load_vocab()
    src = os.path.realpath(args.t4_dataset)
    out = os.path.realpath(args.out)
    if out == src:
        raise SystemExit("--out must differ from --t4-dataset (derived view only)")
    os.makedirs(os.path.join(out, "annotation"), exist_ok=True)

    # symlink everything except annotation/, then annotation/* except the two
    # tables we generate
    for entry in os.listdir(src):
        if entry == "annotation":
            continue
        dst = os.path.join(out, entry)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src, entry), dst)
    for entry in os.listdir(os.path.join(src, "annotation")):
        if entry in ("object_ann.json", "category.json"):
            continue
        dst = os.path.join(out, "annotation", entry)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src, "annotation", entry), dst)

    with open(os.path.join(src, "annotation", "category.json")) as f:
        categories = json.load(f)
    cat_token = {c["name"]: c["token"] for c in categories}

    object_ann = []
    counts, skipped = {}, 0
    for p in sorted(glob.glob(os.path.join(args.labels_dir, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        if "signals" not in d:
            continue
        sd_token = d.get("sample_data_token")
        if not sd_token:
            skipped += 1
            continue
        for s in d["signals"]:
            name = db_tlr_state(s.get("lamps", []), vocab)
            if name not in cat_token:
                cat_token[name] = token_of("category", name)
                categories.append({"token": cat_token[name], "name": name,
                                   "description": "db_tlr state (tlr_autolabel export)",
                                   "index": None,
                                   "has_orientation": False, "has_number": False})
            x0, y0, x1, y1 = s["box_xyxy"]
            object_ann.append({
                "token": token_of("object_ann", sd_token, s.get("signal_id", str(s["box_xyxy"]))),
                "sample_data_token": sd_token,
                "instance_token": None,
                "category_token": cat_token[name],
                "attribute_tokens": [],
                "bbox": [x0, y0, x1, y1],
                "mask": None,
                "automatic_annotation": True,
            })
            counts[name] = counts.get(name, 0) + 1

    with open(os.path.join(out, "annotation", "object_ann.json"), "w") as f:
        json.dump(object_ann, f, indent=2)
    with open(os.path.join(out, "annotation", "category.json"), "w") as f:
        json.dump(categories, f, indent=2)

    print(f"derived dataset: {out}")
    print(f"object_ann: {len(object_ann)} boxes "
          f"({skipped} label files skipped: no sample_data_token)")
    for name, n in sorted(counts.items(), key=lambda t: -t[1]):
        print(f"  {n:5d}  {name}")


if __name__ == "__main__":
    main()
