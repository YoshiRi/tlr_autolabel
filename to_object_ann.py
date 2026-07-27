#!/usr/bin/env python3
"""Core A->B converter: tlr_autolabel (Tier A) or Tier B/v2 sidecar (Tier A')
-> standard t4dataset object_ann.json (Tier B).

The solid interface (agreed 2026-07-23): once labels are standard object_ann,
the existing t4devkit/webauto tooling (AWML create_data, Deepen, CVAT, COCO)
consumes them -- we don't maintain those converters. B holds only:

  object_ann.json  bbox + category_token(db_tlr state) + attribute_tokens
                   (occlusion_state + truncation_state) + instance_token
                   (2D bbox tracking, greedy IoU per camera) + mask
                   (box-rectangle RLE; --no-masks for null)
  instance.json    the tracked 2D instances (token, category, name, counts)
  traffic_light.json  2D instance -> lanelet2 traffic-light linestring relation
  category.json    source categories + the db_tlr state categories used
  attribute.json   occlusion_state.{none,partial,most} + truncation_state.{...}

Map linkage is a SEPARATE relation, distinct from the 2D instance name. The B'
target is t4devkit-defined `traffic_light.json`:

  {"token": ..., "instance_token": ..., "traffic_light_linestring_id": ...}

Regulatory-element and group relationships are resolved from the map through
traffic_light_linestring_id. Do not encode map IDs in object_ann,
instance_token, or instance_name.

During migration this converter can optionally write deprecated legacy
`traffic_light_map_association.json` with matched object_ann -> map way id.
Do not add new consumers for that file.

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

import numpy as np
import yaml

from state_tokens import parse_state


def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# The working t4 TLR datasets (JapanTaxi5, odaiba) store an identical EMPTY
# placeholder mask on every 2D annotation and render the ROI from `bbox`. Real
# per-box RLE (pycocotools) uses a variant t4devkit's mask decoder chokes on, so
# the annotation fails to load and nothing is displayed. Match the placeholder.
PLACEHOLDER_MASK_2880x1860 = "UFhfUzU='"


def placeholder_mask(w, h):
    return {"size": [w, h], "counts": PLACEHOLDER_MASK_2880x1860}


def box_rle(box, w, h):
    """Real box-rectangle RLE (pycocotools, [W,H] self-consistent). Opt-in via
    --real-masks; NOT what t4devkit renders, kept for tools that want a mask."""
    from pycocotools import mask as cocomask
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    m = np.zeros((w, h), np.uint8, order="F")
    if x1 > x0 and y1 > y0:
        m[x0:x1, y0:y1] = 1
    r = cocomask.encode(m)
    return {"size": [w, h], "counts": r["counts"].decode("ascii")}


def instance_token_for(ch, record_index, record):
    uid = record.get("uid") or f"record-{record_index}"
    box = ",".join(f"{float(v):.3f}" for v in record["box"])
    return hashlib.md5(f"instance|{ch}|{record['sd']}|{uid}|{box}".encode()).hexdigest()


def map_ids_compatible(track_map_id, record_map_id):
    return not track_map_id or not record_map_id or str(track_map_id) == str(record_map_id)


def merged_map_id(track_map_id, record_map_id):
    return str(track_map_id or record_map_id or "")


def assign_instances(records, sd_meta):
    """Greedy IoU tracking per camera channel across time -> instance_token per
    record. Instance identity is a 2D annotation concept; map ids are used only
    to avoid writing one instance -> multiple traffic-light linestring relations.
    Returns {record_idx: instance_token} and the instance table."""
    by_channel = {}
    for i, r in enumerate(records):
        ch, ts = sd_meta.get(r["sd"], ("", 0))
        by_channel.setdefault(ch, []).append((ts, i))
    inst_of = {}
    instances = []
    for ch, items in by_channel.items():
        items.sort()
        # (instance_token, last_box, first_ann_uid, category, last_ts, map_id)
        # The instance remains a 2D image identity, but a single instance cannot
        # point to multiple map linestrings in traffic_light.json. Preventing
        # incompatible map merges here keeps the relation table unambiguous.
        active = []
        last_ts = None
        for ts, i in items:
            r = records[i]
            if last_ts is not None and ts != last_ts:
                # new frame: keep only tracks touched last frame (active reset per frame)
                active = [a for a in active if a[4] == last_ts]
            best, bi = 0.3, -1
            for k, a in enumerate(active):
                if not map_ids_compatible(a[5], r.get("map_id")):
                    continue
                v = box_iou(r["box"], a[1])
                if v > best:
                    best, bi = v, k
            if bi >= 0:
                tok = active[bi][0]
                active[bi] = (tok, r["box"], active[bi][2], r["cat"], ts,
                              merged_map_id(active[bi][5], r.get("map_id")))
            else:
                tok = instance_token_for(ch, i, r)
                active.append((tok, r["box"], r["uid"], r["cat"], ts,
                               str(r.get("map_id") or "")))
                instances.append({"token": tok, "category_token": None,
                                  "instance_name": None, "_cat": r["cat"],
                                  "nbr_annotations": 0,
                                  "first_annotation_token": "", "last_annotation_token": ""})
            inst_of[i] = tok
            last_ts = ts
    return inst_of, instances

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(HERE, "configs", "state_vocab", "db_tlr.yaml")

# our visibility -> standard occlusion_state (truncation not tracked in autolabel;
# defaults to non-truncated, a reviewer sets it). Absent visibility -> none.
# Attribute names must match the working t4 TLR datasets EXACTLY (t4devkit maps
# them by name): occlusion_state is lowercase, Truncation_State is capitalized.
OCCLUSION = {"full": "none", "partial": "partial", "occluded": "most", "unknown": "none"}
TRUNCATION_DEFAULT = "Truncation_State.non-truncated"
ATTRIBUTES = ["occlusion_state.none", "occlusion_state.partial", "occlusion_state.most",
              "Truncation_State.non-truncated", "Truncation_State.truncated"]


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
    ap.add_argument("--out", help="derived dataset dir (source symlinked, unchanged)")
    ap.add_argument("--in-place", action="store_true",
                    help="update --t4-dataset itself (backs up the changed tables). "
                         "3D annotations (sample_annotation + their instances) are preserved.")
    ap.add_argument("--autolabel-dir")
    ap.add_argument("--sidecar")
    ap.add_argument("--mask", choices=["null", "placeholder", "real"], default="null",
                    help="null (default): no mask, t4devkit shows the bbox ROI only. "
                         "placeholder: the working-dataset placeholder RLE (renders a big "
                         "fill in t4devkit). real: box-rectangle pycocotools RLE (not "
                         "decoded by t4devkit).")
    ap.add_argument("--write-deprecated-map-association", action="store_true",
                    help="temporary compatibility only: also write deprecated "
                         "traffic_light_map_association.json. New consumers must use "
                         "traffic_light.json.")
    args = ap.parse_args()
    if bool(args.autolabel_dir) == bool(args.sidecar):
        raise SystemExit("give exactly one of --autolabel-dir / --sidecar")
    if bool(args.out) == bool(args.in_place):
        raise SystemExit("give exactly one of --out / --in-place")

    vocab = load_vocab()
    src = os.path.realpath(args.t4_dataset)
    src_ann = os.path.join(src, "annotation")

    # These tables we (re)generate. Everything else — crucially
    # sample_annotation.json (3D boxes) and its ego_pose etc. — is untouched.
    generated = {"object_ann.json", "category.json", "attribute.json",
                 "instance.json", "traffic_light.json",
                 "traffic_light_map_association.json"}

    if args.in_place:
        out = src
        # back up the tables we will rewrite, so 3D data is recoverable
        import datetime
        bdir = os.path.join(src, "build", "annotation_backup_" +
                            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(bdir, exist_ok=True)
        for f in generated:
            p = os.path.join(src_ann, f)
            if os.path.exists(p):
                import shutil
                shutil.copy2(p, os.path.join(bdir, f))
        print(f"backed up existing tables to {bdir}")
    else:
        out = os.path.realpath(args.out)
        if out == src:
            raise SystemExit("--out must differ from --t4-dataset (or use --in-place)")
        os.makedirs(os.path.join(out, "annotation"), exist_ok=True)
        for entry in os.listdir(src):
            if entry == "annotation":
                continue
            dst = os.path.join(out, entry)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(src, entry), dst)
        for entry in os.listdir(src_ann):
            if entry in generated:
                continue
            dst = os.path.join(out, "annotation", entry)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(src_ann, entry), dst)

    # image size + (channel, timestamp) per sample_data, for masks and tracking
    sd_wh, sd_meta = {}, {}
    calib = {r["token"]: r for r in json.load(open(os.path.join(src_ann, "calibrated_sensor.json")))}
    sensor = {r["token"]: r for r in json.load(open(os.path.join(src_ann, "sensor.json")))}
    for sd in json.load(open(os.path.join(src_ann, "sample_data.json"))):
        sd_wh[sd["token"]] = (sd.get("width"), sd.get("height"))
        ch = sensor[calib[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        sd_meta[sd["token"]] = (ch, sd.get("timestamp", 0))
    scene = json.load(open(os.path.join(src_ann, "scene.json")))
    scene_name = scene[0]["name"] if scene else "scene"

    # MERGE with existing tables so 3D annotations survive: keep source
    # categories (car/truck/...), attributes (pseudo-label...), and 3D
    # object instances; append our TLR 2D entries.
    categories = json.load(open(os.path.join(src_ann, "category.json")))
    cat_token = {c["name"]: c["token"] for c in categories}
    src_attributes = json.load(open(os.path.join(src_ann, "attribute.json")))
    attr_token = {a["name"]: a["token"] for a in src_attributes}
    for name in ATTRIBUTES:
        attr_token.setdefault(name, token_of("attribute", name))
    attributes = src_attributes + [{"token": attr_token[n], "name": n, "description": ""}
                                   for n in ATTRIBUTES
                                   if n not in {a["name"] for a in src_attributes}]
    src_instances = json.load(open(os.path.join(src_ann, "instance.json")))
    src_object_ann = json.load(open(os.path.join(src_ann, "object_ann.json")))

    # gather records (with db_tlr category), assign 2D instances, then emit
    records, counts = [], {}
    for sd_token, box, elements, vis, map_id, uid in load_records(args):
        name = db_tlr_state(elements, vocab)
        if name not in cat_token:
            cat_token[name] = token_of("category", name)
            categories.append({"token": cat_token[name], "name": name,
                               "description": "db_tlr state (tlr_autolabel)",
                               "index": None, "has_orientation": False, "has_number": False})
        records.append({"sd": sd_token, "box": [float(v) for v in box], "vis": vis,
                        "map_id": map_id, "uid": uid, "cat": name})
        counts[name] = counts.get(name, 0) + 1

    inst_of, instances = assign_instances(records, sd_meta)

    object_ann, traffic_light, traffic_light_pairs, assoc = [], [], set(), []
    map_ids_by_instance = {}
    inst_anns = {}
    for i, r in enumerate(records):
        x0, y0, x1, y1 = r["box"]
        oa_token = token_of("object_ann", r["sd"], r["uid"], x0, y0, x1, y1)
        occ = "occlusion_state." + OCCLUSION.get(r["vis"] or "unknown", "none")
        attrs = [attr_token[occ], attr_token[TRUNCATION_DEFAULT]]
        w, h = sd_wh.get(r["sd"], (2880, 1860))
        instance_token = inst_of.get(i)
        rec = {
            "token": oa_token,
            "sample_data_token": r["sd"],
            "instance_token": instance_token,
            "category_token": cat_token[r["cat"]],
            "attribute_tokens": attrs,
            "bbox": [x0, y0, x1, y1],
            "mask": (None if args.mask == "null"
                     else placeholder_mask(w, h) if args.mask == "placeholder"
                     else box_rle(r["box"], w, h)),
        }
        object_ann.append(rec)
        inst_anns.setdefault(instance_token, []).append(oa_token)
        if r["map_id"]:
            map_ids_by_instance.setdefault(instance_token, set()).add(str(r["map_id"]))
            pair = (instance_token, str(r["map_id"]))
            if pair not in traffic_light_pairs:
                traffic_light_pairs.add(pair)
                traffic_light.append({
                    "token": token_of("traffic_light", instance_token, r["map_id"]),
                    "instance_token": instance_token,
                    "traffic_light_linestring_id": str(r["map_id"]),
                })
            assoc.append({"object_ann_token": oa_token, "map_traffic_light_id": r["map_id"]})

    ambiguous = {k: sorted(v) for k, v in map_ids_by_instance.items() if len(v) > 1}
    if ambiguous:
        examples = ", ".join(f"{k}:{'/'.join(v)}" for k, v in list(ambiguous.items())[:5])
        raise SystemExit(
            "one instance_token maps to multiple traffic_light_linestring_id values; "
            f"split the 2D instance before writing B' ({examples})"
        )

    # finalize instance table (nbr/first/last + human-readable name)
    for k, inst in enumerate(instances):
        toks = inst_anns.get(inst["token"], [])
        inst["category_token"] = cat_token[inst.pop("_cat")]
        inst["instance_name"] = f"{scene_name}::{inst['token'][:8]}:{k}"
        inst["nbr_annotations"] = len(toks)
        inst["first_annotation_token"] = toks[0] if toks else ""
        inst["last_annotation_token"] = toks[-1] if toks else ""

    # Preserve only the 3D side (instances referenced by sample_annotation, and
    # any object_ann tied to them); regenerate the 2D TLR side every run. This
    # keeps 3D intact AND lets re-runs cleanly replace our own TLR rows instead
    # of accumulating stale ones.
    sample_ann = json.load(open(os.path.join(src_ann, "sample_annotation.json")))
    inst_3d = {s["instance_token"] for s in sample_ann}
    kept_instances = [i for i in src_instances if i["token"] in inst_3d]
    kept_object_ann = [o for o in src_object_ann if o.get("instance_token") in inst_3d]
    final_object_ann = kept_object_ann + object_ann
    final_instances = kept_instances + instances

    ann_out = os.path.join(out, "annotation")
    json.dump(final_object_ann, open(os.path.join(ann_out, "object_ann.json"), "w"), indent=2)
    json.dump(categories, open(os.path.join(ann_out, "category.json"), "w"), indent=2)
    json.dump(attributes, open(os.path.join(ann_out, "attribute.json"), "w"), indent=2)
    json.dump(final_instances, open(os.path.join(ann_out, "instance.json"), "w"), indent=2)
    traffic_light_path = os.path.join(ann_out, "traffic_light.json")
    if traffic_light:
        json.dump(traffic_light, open(traffic_light_path, "w"), indent=2)
    elif os.path.exists(traffic_light_path):
        os.remove(traffic_light_path)
    legacy_assoc_path = os.path.join(ann_out, "traffic_light_map_association.json")
    if args.write_deprecated_map_association:
        json.dump(assoc, open(legacy_assoc_path, "w"), indent=2)
    elif os.path.exists(legacy_assoc_path):
        os.remove(legacy_assoc_path)

    print(f"{'IN-PLACE update of' if args.in_place else 'derived dataset'}: {out}")
    print(f"object_ann: {len(kept_object_ann)} kept(3D-linked) + {len(object_ann)} TLR "
          f"= {len(final_object_ann)} | instances: {len(kept_instances)} 3D + "
          f"{len(instances)} TLR 2D = {len(final_instances)} | masks: {args.mask}")
    print(f"traffic_light relations: {len(traffic_light)} "
          f"({'wrote traffic_light.json' if traffic_light else 'traffic_light.json absent -> Tier B'})")
    if args.write_deprecated_map_association:
        print(f"deprecated legacy map associations: {len(assoc)}")
    for name, n in sorted(counts.items(), key=lambda t: -t[1]):
        print(f"  {n:5d}  {name}")


if __name__ == "__main__":
    main()
