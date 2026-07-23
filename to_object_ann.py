#!/usr/bin/env python3
"""Core A->B converter: tlr_autolabel (Tier A) or Tier B/v2 sidecar (Tier A')
-> standard t4dataset object_ann.json (Tier B).

The solid interface (agreed 2026-07-23): once labels are standard object_ann,
the existing t4devkit/webauto tooling (AWML create_data, Deepen, CVAT, COCO)
consumes them -- we don't maintain those converters. B holds only:

  object_ann.json  bbox + category_token(db_tlr state) + attribute_tokens
                   (occlusion_state + truncation_state) + instance_token
                   (2D bbox tracking; OPTIONAL, empty until tracked) + mask
  category.json    source categories + the db_tlr state categories used
  attribute.json   occlusion_state.{none,partial,most} + truncation_state.{...}

Map linkage is a SEPARATE optional identity ("which map signal"), distinct from
the 2D instance, written to a sidecar (not into object_ann) so B stays strictly
standard and the field is simply absent when there is no map / no match:

  traffic_light_map_association.json  [{object_ann_token, map_traffic_light_id}]

The regulatory-element relation (which lanes) is NOT persisted: re-derive it
from the lanelet2 map at evaluation time given map_traffic_light_id.

Input is either a Tier A autolabel dir (--autolabel-dir, lamps) or a Tier B/v2
sidecar (--sidecar, canonical state + map_traffic_light_id + visibility). Output
is a derived dataset dir (source symlinked, generated tables replaced) -- the
source annotation is never edited.

Usage:
  python3 to_object_ann.py --t4-dataset <src> --out <derived> \
      (--autolabel-dir <dir> | --sidecar <traffic_signal_2d_ann.json>)
"""
import argparse
import glob
import hashlib
import json
import os

import yaml

from state_tokens import parse_state

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(HERE, "configs", "state_vocab", "db_tlr.yaml")

# our visibility -> standard occlusion_state (truncation not tracked in autolabel;
# defaults to non-truncated, a reviewer sets it). Absent visibility -> none.
OCCLUSION = {"full": "none", "partial": "partial", "occluded": "most", "unknown": "none"}
ATTRIBUTES = ["occlusion_state.none", "occlusion_state.partial", "occlusion_state.most",
              "truncation_state.non-truncated", "truncation_state.truncated"]


def load_vocab():
    with open(VOCAB_PATH) as f:
        return yaml.safe_load(f)


def db_tlr_state(elements, vocab):
    """Project canonical lamp elements onto a db_tlr category name (lossy)."""
    if not elements:
        return "unknown"
    if any(e["shape"] == "ped" for e in elements):
        color = next((e["color"] for e in elements if e["shape"] == "ped"), None)
        name = {"red": "crosswalk_red", "green": "crosswalk_green"}.get(color, "crosswalk_unknown")
        return name if name in vocab["allowed"] else "unknown"
    colors = [vocab["color_names"].get(e["color"]) for e in elements if e["shape"] == "circle"]
    arrows = [vocab["arrow_names"].get(e["arrow"]) for e in elements if e["shape"] == "arrow"]
    if None in colors or None in arrows:
        return "unknown"
    parts = sorted(set(colors)) + [a for a in vocab["arrow_order"] if a in set(arrows)]
    if not parts:
        return "unknown"
    name = "_".join(parts)
    return name if name in vocab["allowed"] else "unknown"


def token_of(*parts):
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


def load_records(args):
    """Yield (sample_data_token, box_xyxy, elements, visibility, map_id, uid).
    elements = canonical lamp dicts; map_id/visibility may be None."""
    if args.sidecar:
        data = json.load(open(args.sidecar))
        anns = data["annotations"] if isinstance(data, dict) else data
        for a in anns:
            at = a.get("attributes", {})
            state = at.get("state") or at.get("raw_state") or ""
            yield (a["sample_data_token"], a["box2d"], parse_state(state),
                   at.get("visibility"), at.get("map_traffic_light_id") or None,
                   a.get("token"))
    else:
        for p in sorted(glob.glob(os.path.join(args.autolabel_dir, "**", "*.json"),
                                  recursive=True)):
            d = json.load(open(p))
            if "signals" not in d or not d.get("sample_data_token"):
                continue
            for i, s in enumerate(d["signals"]):
                elements = [{"color": l["color"], "shape": l["shape"], "arrow": l.get("arrow")}
                            for l in s.get("lamps", [])]
                yield (d["sample_data_token"], s["box_xyxy"], elements, None, None,
                       s.get("signal_id") or f"{d.get('frame_index')}-{i}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t4-dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--autolabel-dir")
    ap.add_argument("--sidecar")
    args = ap.parse_args()
    if bool(args.autolabel_dir) == bool(args.sidecar):
        raise SystemExit("give exactly one of --autolabel-dir / --sidecar")

    vocab = load_vocab()
    src = os.path.realpath(args.t4_dataset)
    out = os.path.realpath(args.out)
    if out == src:
        raise SystemExit("--out must differ from --t4-dataset (derived view only)")
    os.makedirs(os.path.join(out, "annotation"), exist_ok=True)

    generated = {"object_ann.json", "category.json", "attribute.json",
                 "traffic_light_map_association.json"}
    for entry in os.listdir(src):
        if entry == "annotation":
            continue
        dst = os.path.join(out, entry)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src, entry), dst)
    for entry in os.listdir(os.path.join(src, "annotation")):
        if entry in generated:
            continue
        dst = os.path.join(out, "annotation", entry)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src, "annotation", entry), dst)

    # categories: start from source, add db_tlr states as used
    categories = json.load(open(os.path.join(src, "annotation", "category.json")))
    cat_token = {c["name"]: c["token"] for c in categories}
    # attributes: standard occlusion/truncation set
    attr_token = {name: token_of("attribute", name) for name in ATTRIBUTES}
    attributes = [{"token": attr_token[n], "name": n, "description": ""} for n in ATTRIBUTES]

    object_ann, assoc, counts = [], [], {}
    for sd_token, box, elements, vis, map_id, uid in load_records(args):
        name = db_tlr_state(elements, vocab)
        if name not in cat_token:
            cat_token[name] = token_of("category", name)
            categories.append({"token": cat_token[name], "name": name,
                               "description": "db_tlr state (tlr_autolabel)",
                               "index": None, "has_orientation": False, "has_number": False})
        occ = "occlusion_state." + OCCLUSION.get(vis or "unknown", "none")
        attrs = [attr_token[occ], attr_token["truncation_state.non-truncated"]]
        x0, y0, x1, y1 = box
        oa_token = token_of("object_ann", sd_token, uid, x0, y0, x1, y1)
        object_ann.append({
            "token": oa_token,
            "sample_data_token": sd_token,
            "instance_token": None,          # 2D tracking: optional, filled later
            "category_token": cat_token[name],
            "attribute_tokens": attrs,
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "mask": None,
        })
        if map_id:                            # optional map-signal identity, separate from object_ann
            assoc.append({"object_ann_token": oa_token, "map_traffic_light_id": map_id})
        counts[name] = counts.get(name, 0) + 1

    ann_out = os.path.join(out, "annotation")
    json.dump(object_ann, open(os.path.join(ann_out, "object_ann.json"), "w"), indent=2)
    json.dump(categories, open(os.path.join(ann_out, "category.json"), "w"), indent=2)
    json.dump(attributes, open(os.path.join(ann_out, "attribute.json"), "w"), indent=2)
    json.dump(assoc, open(os.path.join(ann_out, "traffic_light_map_association.json"), "w"), indent=2)

    print(f"derived t4 dataset: {out}")
    print(f"object_ann: {len(object_ann)} boxes | map associations: {len(assoc)} "
          f"({'present' if assoc else 'absent — no map/match, optional'})")
    for name, n in sorted(counts.items(), key=lambda t: -t[1]):
        print(f"  {n:5d}  {name}")


if __name__ == "__main__":
    main()
