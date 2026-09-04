#!/usr/bin/env python3
"""Read a dataset's existing human object_ann into Tier A (tlr_autolabel/v1).

Per docs/existing_annotation_if.md the join point for foreign annotations is the
L3 adapter, not the inference path: the boxes and states stay exactly as a human
drew them and no model runs. That makes the rest of the stack -- map matching, RE
aggregation, the consistency check, the review views -- usable as a way to
interrogate a ground truth, and any disagreement it reports is a statement about
our projection or the map rather than about a detector.

db_tlr carries the state as the per-box category name, so it is decoded through
the same vocabulary the exporter writes (configs/state_vocab/db_tlr.yaml).

Two details that matter, both learned the hard way:

- `detector_score` is set to 1.0. Human GT has no score, and a missing one reads
  as 0.0 downstream, which drops every box at the default --min-score.
- `state_score` is set to 1.0 as well: a human read the lamp, so the state is
  asserted as confidently as the box. Leaving it empty would make every GT box
  look like a housing whose state could not be read.

Occlusion and truncation the annotator recorded are kept as `gt_occlusion` /
`gt_truncation` rather than folded into `visibility`, so a comparison can tell
the annotator's judgement from ours.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tlr_autolabel.core.state_tokens import elements_key           # noqa: E402
from tlr_autolabel.t4.convert import db_tlr_to_elements, load_vocab  # noqa: E402


def load(path):
    return json.loads(Path(path).read_text())


def channel_of(filename):
    parts = Path(filename).parts
    return parts[1] if len(parts) >= 2 and parts[0] == "data" else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("--out-dir", default="tlr_autolabel_gt",
                    help="output dir, relative to the dataset root")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    root = a.dataset_root.resolve()
    ann = root / "annotation"
    out = root / a.out_dir if not Path(a.out_dir).is_absolute() else Path(a.out_dir)
    if out.exists() and any(out.rglob("*.json")) and not a.overwrite:
        raise SystemExit(f"output dir already has JSON (use --overwrite): {out}")

    vocab = load_vocab()
    categories = {c["token"]: c.get("name", "") for c in load(ann / "category.json")}
    attributes = {x["token"]: x.get("name", "") for x in load(ann / "attribute.json")} \
        if (ann / "attribute.json").exists() else {}
    instances = {i["token"]: i for i in load(ann / "instance.json")} \
        if (ann / "instance.json").exists() else {}
    frames = {r["token"]: r for r in load(ann / "sample_data.json")}
    object_ann = load(ann / "object_ann.json")

    by_frame = defaultdict(list)
    skipped, cats, undecoded = Counter(), Counter(), Counter()
    for row in object_ann:
        frame = frames.get(row.get("sample_data_token"))
        if frame is None:
            skipped["no_sample_data"] += 1
            continue
        box = [float(v) for v in row.get("bbox", [])]
        if len(box) != 4:
            skipped["bad_bbox"] += 1
            continue
        name = categories.get(row.get("category_token"), "")
        elements = db_tlr_to_elements(name, vocab)
        cats[name] += 1
        if not elements and name not in ("unknown", "crosswalk_unknown"):
            undecoded[name] += 1
        attrs = [attributes.get(t, "") for t in row.get("attribute_tokens", [])
                 if attributes.get(t, "")]
        instance = instances.get(row.get("instance_token"), {})
        by_frame[frame["token"]].append({
            "signal_id": row.get("token", ""),
            # human GT has no score; absent reads as 0.0 and the default
            # --min-score would then drop every box
            "detector_score": 1.0,
            "state_score": 1.0,
            "box_xyxy": box,
            "box_level": "housing",
            "lamps": [{"label": elements_key([e]), "color": e["color"],
                       "shape": e["shape"], "arrow": e.get("arrow"),
                       "confidence": 1.0} for e in elements],
            "state": elements_key(elements) or "unknown",
            "source_object_ann_token": row.get("token", ""),
            "source_instance_token": row.get("instance_token", ""),
            "source_instance_name": instance.get("instance_name", ""),
            "source_category": name,
            "source_attributes": attrs,
            # the annotator's own judgement, kept separate from ours
            "gt_occlusion": next((x for x in attrs if x.startswith("occlusion")), ""),
            "gt_truncation": next((x for x in attrs if "runcation" in x), ""),
        })

    written = signals = 0
    for token, frame in sorted(frames.items(), key=lambda kv: kv[1].get("timestamp", 0)):
        channel = channel_of(frame.get("filename", ""))
        if not channel:
            continue
        stem = Path(frame["filename"]).stem
        path = out / channel / f"{stem}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(by_frame.get(token, []), key=lambda s: s["signal_id"])
        path.write_text(json.dumps({
            "schema_version": "tlr_autolabel/v1",
            "box_level": "housing",
            "image": frame.get("filename", ""),
            "sample_data_token": token,
            "channel": channel,
            "frame_index": int(stem) if stem.isdigit() else 0,
            "width": int(frame.get("width") or 0),
            "height": int(frame.get("height") or 0),
            "meta": {"source": "human_object_ann", "dataset_root": str(root)},
            "signals": rows,
        }, indent=2, ensure_ascii=False))
        written += 1
        signals += len(rows)

    print(f"wrote {written} frame JSON files to {out}")
    print(f"wrote {signals} signals from {len(object_ann)} object_ann rows")
    print("categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))
    if undecoded:
        print("NOT decodable by the db_tlr vocabulary: "
              + ", ".join(f"{k}={v}" for k, v in sorted(undecoded.items())))
    if skipped:
        print("skipped: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))


if __name__ == "__main__":
    main()
