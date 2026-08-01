#!/usr/bin/env python3
"""Two-source evaluation: score a PREDICTION run against a GT run.

Source-agnostic (see docs/eval_design.md): the prediction can be the ROS node's
output or another offline config; GT is a reviewed Tier B sidecar. Both sides are
Tier B (traffic_signal_2d/v2). Levels:

  1 detection    — match pred boxes to GT boxes per frame by IoU -> TP/FP/FN ->
                   precision / recall / IoU, by distance bin.
  2 classification — on TP matches, compare canonical state -> state accuracy,
                   element precision/recall, top confusions. Decoupled from
                   detection (only scored on boxes both sides found).
  3 RE (optional)  — with --gt-re/--pred-re timeseries, per-RE state agreement.

GT box selection: reviewed `accepted`/`fixed` count as real, `rejected` excluded;
if NO annotation is reviewed (provisional GT) every non-rejected box is treated
as GT (machinery test / self-consistency, not accuracy — a warning is printed).

Usage:
  python3 eval_vs_gt.py --gt gt_sidecar.json --pred pred_sidecar.json \
      [--iou 0.3] [--output build/tl_match/eval_vs_gt.json]
Distances come from build/tl_match/match_report.json when --dataset-root given.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tlr_autolabel.core.state_tokens import elements_key, parse_state

DIST_BINS = [(0, 30), (30, 60), (60, 100), (100, 150)]
GT_STATUSES = {"accepted", "fixed"}


def bin_of(d):
    if d is None:
        return "unknown"
    for lo, hi in DIST_BINS:
        if lo <= d < hi:
            return f"{lo}-{hi}m"
    return f">={DIST_BINS[-1][1]}m"


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def state_of(ann):
    a = ann["attributes"]
    raw = a.get("raw_state") or a.get("state") or ""
    return elements_key(parse_state(raw)) or "unknown"


def load(sidecar_path):
    anns = json.loads(Path(sidecar_path).read_text())["annotations"]
    by_frame = defaultdict(list)
    for a in anns:
        by_frame[a["sample_data_token"]].append(a)
    return anns, by_frame


def gt_boxes(by_frame):
    """Keep real GT boxes; if nothing is reviewed, use all non-rejected (provisional)."""
    reviewed = any(a["attributes"].get("review_status", "unchecked") in
                   (GT_STATUSES | {"rejected"})
                   for fr in by_frame.values() for a in fr)
    out = defaultdict(list)
    for tok, fr in by_frame.items():
        for a in fr:
            rs = a["attributes"].get("review_status", "unchecked")
            if reviewed:
                if rs in GT_STATUSES:
                    out[tok].append(a)
            elif rs != "rejected":
                out[tok].append(a)
    return out, reviewed


def match_frame(gt, pred, thr):
    """Greedy IoU matching within a frame. Returns (pairs, gt_unmatched, pred_unmatched)."""
    cand = sorted(((iou(g["box2d"], p["box2d"]), gi, pi)
                   for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
                  key=lambda t: -t[0])
    gused, pused, pairs = set(), set(), []
    for v, gi, pi in cand:
        if v < thr or gi in gused or pi in pused:
            continue
        gused.add(gi); pused.add(pi); pairs.append((gi, pi, v))
    return (pairs, [i for i in range(len(gt)) if i not in gused],
            [i for i in range(len(pred)) if i not in pused])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--output", default="build/tl_match/eval_vs_gt.json")
    args = ap.parse_args()

    _, gt_bf = load(args.gt)
    _, pred_bf = load(args.pred)
    gt_bf, reviewed = gt_boxes(gt_bf)

    # distance per (token, rounded box) from the GT match_report if available
    dist = {}
    if args.dataset_root:
        rep = Path(args.dataset_root) / "build/tl_match/match_report.json"
        if rep.exists():
            for f in json.loads(rep.read_text())["frames"]:
                for p in f["pairs"]:
                    if p.get("distance_m") is not None:
                        key = (f["sample_data_token"],
                               tuple(round(float(v), 1) for v in p["detection_box"]))
                        dist[key] = p["distance_m"]

    def dbin(tok, box):
        return bin_of(dist.get((tok, tuple(round(float(v), 1) for v in box))))

    det = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "iou": []})
    cls = defaultdict(lambda: {"n": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0})
    confusion = Counter()
    for tok in set(gt_bf) | set(pred_bf):
        g, p = gt_bf.get(tok, []), pred_bf.get(tok, [])
        pairs, gu, pu = match_frame(g, p, args.iou)
        for gi, pi, v in pairs:
            b = dbin(tok, g[gi]["box2d"])
            det[b]["tp"] += 1; det[b]["iou"].append(v)
            gs, ps = state_of(g[gi]), state_of(p[pi])
            c = cls[b]; c["n"] += 1; c["exact"] += (gs == ps)
            ge = {(e["color"], e["shape"], e["arrow"]) for e in parse_state(gs)}
            pe = {(e["color"], e["shape"], e["arrow"]) for e in parse_state(ps)}
            c["tp"] += len(ge & pe); c["fp"] += len(pe - ge); c["fn"] += len(ge - pe)
            if gs != ps:
                confusion[(gs, ps)] += 1
        for gi in gu:
            det[dbin(tok, g[gi]["box2d"])]["fn"] += 1
        for pi in pu:
            det[dbin(tok, p[pi]["box2d"])]["fp"] += 1

    def f(x, y):
        return round(x / y, 3) if y else None

    order = [f"{lo}-{hi}m" for lo, hi in DIST_BINS] + [f">={DIST_BINS[-1][1]}m", "unknown"]
    det_rows, cls_rows = {}, {}
    for b in order:
        if b in det:
            d = det[b]; ious = sorted(d["iou"])
            det_rows[b] = {"tp": d["tp"], "fp": d["fp"], "fn": d["fn"],
                           "precision": f(d["tp"], d["tp"] + d["fp"]),
                           "recall": f(d["tp"], d["tp"] + d["fn"]),
                           "iou_median": round(ious[len(ious) // 2], 3) if ious else None}
        if b in cls and cls[b]["n"]:
            c = cls[b]
            cls_rows[b] = {"n": c["n"], "state_accuracy": f(c["exact"], c["n"]),
                           "element_precision": f(c["tp"], c["tp"] + c["fp"]),
                           "element_recall": f(c["tp"], c["tp"] + c["fn"])}
    tot = {k: sum(det[b][k] for b in det) for k in ("tp", "fp", "fn")}
    report = {
        "schema_version": "tlr_eval_vs_gt/v1",
        "gt": args.gt, "pred": args.pred, "iou_threshold": args.iou,
        "gt_reviewed": reviewed,
        "detection": {"overall": {"tp": tot["tp"], "fp": tot["fp"], "fn": tot["fn"],
                                  "precision": f(tot["tp"], tot["tp"] + tot["fp"]),
                                  "recall": f(tot["tp"], tot["tp"] + tot["fn"])},
                      "by_distance": det_rows},
        "classification": {"by_distance": cls_rows,
                           "top_confusions": [{"gt": g, "pred": p, "n": n}
                                              for (g, p), n in confusion.most_common(12)]},
    }
    out = Path(args.dataset_root or ".") / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    if not reviewed:
        print("WARNING: GT has no reviewed annotations — treating all non-rejected "
              "boxes as GT. Numbers are self-consistency/machinery, NOT accuracy.")
    o = report["detection"]["overall"]
    print(f"detection: P={o['precision']} R={o['recall']} "
          f"(tp={o['tp']} fp={o['fp']} fn={o['fn']}) @IoU>={args.iou}")
    print("  by distance:", {b: (r["precision"], r["recall"]) for b, r in det_rows.items()})
    print("classification state accuracy by distance:",
          {b: r["state_accuracy"] for b, r in cls_rows.items()})
    if confusion:
        print("  top state confusions (gt->pred):",
              [(g, p, n) for (g, p), n in confusion.most_common(5)])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
