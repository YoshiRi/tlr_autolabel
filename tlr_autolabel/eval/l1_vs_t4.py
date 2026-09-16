#!/usr/bin/env python3
"""Evaluate L1 tlr_autolabel/v1 JSONs against T4 object_ann GT.

This is the map-less evaluator for datasets where GT is stored as nuScenes/T4
2D annotations:

  annotation/object_ann.json  +  annotation/category.json

The GT category name is the traffic-light state in db_tlr style
(`red_right`, `crosswalk_green`, ...). Predictions are L1 per-frame JSONs
(`tlr_autolabel/v1`). Matching is per `sample_data_token` by box IoU, so no
Lanelet2 map, regulatory element id, or ego pose is required.

Usage:
  python3 eval_l1_vs_t4_gt.py \
    --dataset-root <t4_dataset> \
    --pred-dir <t4_dataset>/tlr_autolabel/CAM_TRAFFIC_LIGHT_NEAR
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tlr_autolabel.core.state_tokens import elements_key, parse_state
from tlr_autolabel.t4.convert import db_tlr_to_elements, load_vocab


def iou(a: list[float], b: list[float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def canonical_state(raw: str) -> str:
    return elements_key(parse_state(raw)) or "unknown"


def db_tlr_to_canonical(name: str) -> str:
    """Convert a db_tlr category name to the canonical state string.

    Decoding lives in one place -- t4.convert.db_tlr_to_elements -- because this
    function used to carry a second copy that gave the arrow the *circle's*
    colour. On a dataset whose GT is `red_right` that read as a red right-arrow
    while the model reported the green one that is actually lit, and 93 of 457
    matched boxes were scored wrong for it. The JP arrow panel is green, and
    db_tlr stores no arrow colour at all -- db_tlr_state() drops it on the way
    out, so green is also the only choice that round-trips.
    """
    # `or "unknown"` keeps the original contract: an unreadable or absent
    # category is the string "unknown", not the empty string.
    return elements_key(db_tlr_to_elements(name, load_vocab())) or "unknown"


def signal_kind(state: str) -> str:
    elems = parse_state(state)
    if any(e["shape"] == "ped" for e in elems):
        return "pedestrian"
    return "vehicle" if elems else "unknown"


def load_t4_gt(dataset_root: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    ann = dataset_root / "annotation"
    categories = json.loads((ann / "category.json").read_text())
    cat_by_token = {c["token"]: c["name"] for c in categories}
    sample_data = json.loads((ann / "sample_data.json").read_text())
    sd_by_token = {r["token"]: r for r in sample_data}

    rows = []
    for obj in json.loads((ann / "object_ann.json").read_text()):
        cat = cat_by_token.get(obj.get("category_token"), "unknown")
        state = db_tlr_to_canonical(cat)
        sd = sd_by_token.get(obj.get("sample_data_token"), {})
        row = {
            "token": obj.get("token"),
            "sample_data_token": obj.get("sample_data_token"),
            "sample_token": sd.get("sample_token"),
            "filename": sd.get("filename"),
            "timestamp": sd.get("timestamp"),
            "box": [float(v) for v in obj["bbox"]],
            "category": cat,
            "state": state,
            "signal_kind": signal_kind(state),
            "attribute_tokens": obj.get("attribute_tokens", []),
            "instance_token": obj.get("instance_token"),
        }
        rows.append(row)

    by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[row["sample_data_token"]].append(row)
    return rows, by_frame


def load_l1_predictions(pred_dir: Path, min_score: float) -> tuple[list[dict], dict[str, list[dict]], int]:
    rows = []
    skipped_no_token = 0
    for path in sorted(pred_dir.rglob("*.json")):
        if path.name.endswith(".viz.json"):
            continue
        payload = json.loads(path.read_text())
        if "signals" not in payload:
            continue
        sd_token = payload.get("sample_data_token")
        if not sd_token:
            skipped_no_token += len(payload.get("signals", []))
            continue
        for signal in payload.get("signals", []):
            score = signal.get("detector_score")
            if score is not None and float(score) < min_score:
                continue
            state = canonical_state(signal.get("state") or signal.get("signal") or "unknown")
            rows.append({
                "token": f"{path.stem}:{signal.get('signal_id', len(rows))}",
                "sample_data_token": sd_token,
                "channel": payload.get("channel"),
                "filename": payload.get("image"),
                "box": [float(v) for v in signal["box_xyxy"]],
                "state": state,
                "signal_kind": signal_kind(state),
                "detector_score": score,
                "raw_state": signal.get("state") or signal.get("signal") or "unknown",
            })

    by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[row["sample_data_token"]].append(row)
    return rows, by_frame, skipped_no_token


def match_frame(gt: list[dict], pred: list[dict], iou_thr: float):
    candidates = sorted(
        ((iou(g["box"], p["box"]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
        key=lambda item: -item[0],
    )
    gt_used, pred_used, pairs = set(), set(), []
    for value, gi, pi in candidates:
        if value < iou_thr or gi in gt_used or pi in pred_used:
            continue
        gt_used.add(gi)
        pred_used.add(pi)
        pairs.append((gi, pi, value))
    return (
        pairs,
        [i for i in range(len(gt)) if i not in gt_used],
        [i for i in range(len(pred)) if i not in pred_used],
    )


def div(num: int, den: int):
    return round(num / den, 4) if den else None


def evaluate(gt_by_frame: dict[str, list[dict]], pred_by_frame: dict[str, list[dict]],
             iou_thr: float, score_stateless_as_fp: bool = False):
    det = {"tp": 0, "fp": 0, "fn": 0, "not_evaluated": 0, "iou": []}
    cls = {"n": 0, "exact": 0, "known_n": 0, "known_exact": 0, "elem_tp": 0, "elem_fp": 0, "elem_fn": 0}
    by_kind = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "cls_n": 0, "cls_exact": 0})
    by_category = defaultdict(lambda: {"gt": 0, "tp": 0, "fn": 0, "cls_n": 0, "cls_exact": 0})
    pred_fp_state = Counter()
    confusion = Counter()
    matches_out = []
    gt_misses = []
    pred_false_positives = []
    pred_not_evaluated = []

    for token in sorted(set(gt_by_frame) | set(pred_by_frame)):
        gt = gt_by_frame.get(token, [])
        pred = pred_by_frame.get(token, [])
        pairs, gt_unmatched, pred_unmatched = match_frame(gt, pred, iou_thr)

        for g in gt:
            by_category[g["category"]]["gt"] += 1

        for gi, pi, value in pairs:
            g, p = gt[gi], pred[pi]
            det["tp"] += 1
            det["iou"].append(value)
            by_kind[g["signal_kind"]]["tp"] += 1
            by_category[g["category"]]["tp"] += 1

            cls["n"] += 1
            by_kind[g["signal_kind"]]["cls_n"] += 1
            by_category[g["category"]]["cls_n"] += 1
            exact = g["state"] == p["state"]
            cls["exact"] += exact
            by_kind[g["signal_kind"]]["cls_exact"] += exact
            by_category[g["category"]]["cls_exact"] += exact
            if g["state"] != "unknown":
                cls["known_n"] += 1
                cls["known_exact"] += exact
            ge = {(e["color"], e["shape"], e["arrow"]) for e in parse_state(g["state"])}
            pe = {(e["color"], e["shape"], e["arrow"]) for e in parse_state(p["state"])}
            cls["elem_tp"] += len(ge & pe)
            cls["elem_fp"] += len(pe - ge)
            cls["elem_fn"] += len(ge - pe)
            if not exact:
                confusion[(g["state"], p["state"])] += 1
            matches_out.append({
                "sample_data_token": token,
                "filename": g.get("filename") or p.get("filename"),
                "iou": round(value, 4),
                "gt_box": g["box"],
                "pred_box": p["box"],
                "gt_category": g["category"],
                "gt_state": g["state"],
                "pred_state": p["state"],
                "pred_score": p.get("detector_score"),
            })

        for gi in gt_unmatched:
            g = gt[gi]
            det["fn"] += 1
            by_kind[g["signal_kind"]]["fn"] += 1
            by_category[g["category"]]["fn"] += 1
            gt_misses.append({
                "sample_data_token": token,
                "filename": g.get("filename"),
                "gt_box": g["box"],
                "gt_category": g["category"],
                "gt_state": g["state"],
            })

        for pi in pred_unmatched:
            p = pred[pi]
            row = {
                "sample_data_token": token,
                "filename": p.get("filename"),
                "pred_box": p["box"],
                "pred_state": p["state"],
                "pred_score": p.get("detector_score"),
            }
            if score_stateless_as_fp or p["state"] != "unknown":
                det["fp"] += 1
                by_kind[p["signal_kind"]]["fp"] += 1
                pred_fp_state[p["state"]] += 1
                pred_false_positives.append(row)
            else:
                # Out of scope, not a mistake: GT does not annotate the back of
                # a housing, because a face with no lamp has no state to
                # evaluate. A detection with no readable state is that back
                # face in the large majority of cases -- on 2821adc7, 202 of
                # 236 unmatched detections, and every one of the three samples
                # inspected by eye was a back plate. Counting them as FP put
                # precision at 0.660 while the model and the annotator actually
                # agreed. They are kept here so the exclusion stays auditable.
                det["not_evaluated"] += 1
                pred_not_evaluated.append(row)

    ious = sorted(det["iou"])
    overall = {
        "tp": det["tp"],
        "fp": det["fp"],
        "fn": det["fn"],
        "not_evaluated": det["not_evaluated"],
        "precision": div(det["tp"], det["tp"] + det["fp"]),
        "recall": div(det["tp"], det["tp"] + det["fn"]),
        "f1": div(2 * det["tp"], 2 * det["tp"] + det["fp"] + det["fn"]),
        "precision_counting_stateless_as_fp": div(
            det["tp"], det["tp"] + det["fp"] + det["not_evaluated"]),
        "iou_mean": round(sum(ious) / len(ious), 4) if ious else None,
        "iou_median": round(ious[len(ious) // 2], 4) if ious else None,
    }
    classification = {
        "matched_boxes": cls["n"],
        "state_accuracy": div(cls["exact"], cls["n"]),
        "known_gt_state_accuracy": div(cls["known_exact"], cls["known_n"]),
        "element_precision": div(cls["elem_tp"], cls["elem_tp"] + cls["elem_fp"]),
        "element_recall": div(cls["elem_tp"], cls["elem_tp"] + cls["elem_fn"]),
        "top_confusions": [
            {"gt": gt, "pred": pred, "n": n}
            for (gt, pred), n in confusion.most_common(20)
        ],
    }

    by_kind_out = {}
    for kind, row in sorted(by_kind.items()):
        by_kind_out[kind] = {
            "tp": row["tp"],
            "fp": row["fp"],
            "fn": row["fn"],
            "precision": div(row["tp"], row["tp"] + row["fp"]),
            "recall": div(row["tp"], row["tp"] + row["fn"]),
            "state_accuracy": div(row["cls_exact"], row["cls_n"]),
        }

    by_category_out = {}
    for category, row in sorted(by_category.items()):
        by_category_out[category] = {
            "gt": row["gt"],
            "tp": row["tp"],
            "fn": row["fn"],
            "recall": div(row["tp"], row["tp"] + row["fn"]),
            "state_accuracy": div(row["cls_exact"], row["cls_n"]),
        }

    return {
        "detection": {"overall": overall, "by_signal_kind": by_kind_out, "by_gt_category": by_category_out},
        "classification": classification,
        "pred_false_positive_states": [{"state": k, "n": v} for k, v in pred_fp_state.most_common(20)],
        "matches": matches_out,
        "gt_misses": gt_misses,
        "pred_false_positives": pred_false_positives,
        "pred_not_evaluated": pred_not_evaluated,
    }


def write_markdown(report: dict, path: Path):
    det = report["detection"]["overall"]
    cls = report["classification"]
    lines = [
        "# L1 vs T4 GT Evaluation",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| TP | {det['tp']} |",
        f"| FP | {det['fp']} |",
        f"| 評価対象外 (状態なし・未マッチ) | {det.get('not_evaluated', 0)} |",
        f"| FN | {det['fn']} |",
        f"| precision | {det['precision']} |",
        f"| recall | {det['recall']} |",
        f"| F1 | {det['f1']} |",
        f"| IoU median | {det['iou_median']} |",
        f"| state accuracy | {cls['state_accuracy']} |",
        f"| known GT state accuracy | {cls['known_gt_state_accuracy']} |",
        "",
        "## By Signal Kind",
        "",
        "| kind | TP | FP | FN | precision | recall | state acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, row in report["detection"]["by_signal_kind"].items():
        lines.append(
            f"| {kind} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['precision']} | {row['recall']} | {row['state_accuracy']} |"
        )
    lines += [
        "",
        "## By GT Category",
        "",
        "| category | GT | TP | FN | recall | state acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for category, row in report["detection"]["by_gt_category"].items():
        lines.append(
            f"| {category} | {row['gt']} | {row['tp']} | {row['fn']} | "
            f"{row['recall']} | {row['state_accuracy']} |"
        )
    if report["classification"]["top_confusions"]:
        lines += ["", "## Top Confusions", "", "| GT | Pred | n |", "|---|---|---:|"]
        for row in report["classification"]["top_confusions"]:
            lines.append(f"| {row['gt']} | {row['pred']} | {row['n']} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path,
                        help="Directory containing tlr_autolabel/v1 JSONs; searched recursively.")
    parser.add_argument("--iou", default=0.3, type=float)
    parser.add_argument("--min-score", default=0.0, type=float,
                        help="Ignore L1 predictions below this detector_score.")
    parser.add_argument("--score-stateless-as-fp", action="store_true",
                        help="Count an unmatched detection with no readable "
                             "state as a false positive. Off by default: GT "
                             "does not annotate a housing's back face, so a "
                             "detection with no state has nothing to be "
                             "scored against. Turn this on to reproduce "
                             "pre-decision numbers.")
    parser.add_argument("--output", default=Path("build/tl_eval/l1_vs_t4_gt.json"), type=Path)
    parser.add_argument("--markdown", default=Path("build/tl_eval/l1_vs_t4_gt.md"), type=Path)
    args = parser.parse_args()

    gt_rows, gt_by_frame = load_t4_gt(args.dataset_root)
    pred_rows, pred_by_frame, skipped_no_token = load_l1_predictions(args.pred_dir, args.min_score)
    report = evaluate(gt_by_frame, pred_by_frame, args.iou,
                      score_stateless_as_fp=args.score_stateless_as_fp)
    report = {
        "schema_version": "tlr_l1_vs_t4_gt/v1",
        "dataset_root": str(args.dataset_root),
        "pred_dir": str(args.pred_dir),
        "iou_threshold": args.iou,
        "min_score": args.min_score,
        "score_stateless_as_fp": args.score_stateless_as_fp,
        "inputs": {
            "gt_boxes": len(gt_rows),
            "gt_frames": len(gt_by_frame),
            "pred_boxes": len(pred_rows),
            "pred_frames": len(pred_by_frame),
            "pred_skipped_no_sample_data_token": skipped_no_token,
        },
        **report,
    }

    out = args.output if args.output.is_absolute() else args.dataset_root / args.output
    md = args.markdown if args.markdown.is_absolute() else args.dataset_root / args.markdown
    out.parent.mkdir(parents=True, exist_ok=True)
    md.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown(report, md)

    det = report["detection"]["overall"]
    cls = report["classification"]
    print(
        f"detection: P={det['precision']} R={det['recall']} F1={det['f1']} "
        f"(tp={det['tp']} fp={det['fp']} fn={det['fn']}"
        + (f" not-evaluated={det['not_evaluated']}" if det.get("not_evaluated") else "")
        + f") @IoU>={args.iou}"
    )
    print(
        f"classification: state_acc={cls['state_accuracy']} "
        f"known_gt_state_acc={cls['known_gt_state_accuracy']}"
    )
    if skipped_no_token:
        print(f"WARNING: skipped {skipped_no_token} predictions without sample_data_token")
    print(f"wrote {out}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
