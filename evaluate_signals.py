#!/usr/bin/env python3
"""L6 evaluation: metrics over Tier B + RE time series (pure analysis, no inference).

Positioning (2026-07-19): GT is always human-made; this layer is the metrics
engine. It always computes GT-free metrics, and the GT-dependent block
activates automatically when the Tier B sidecar contains reviewed annotations
(`review_status` in {accepted, fixed, rejected}).

Data design: three tidy long-format ledgers are the reusable artifacts (one
row = one unit of observation); every report table is a `pivot()` view of them,
so slicing by pedestrian/vehicle, lamp color/shape/arrow, facing, channel,
distance, matched/unmatched, reviewed/not is a group-by, not new code. See
docs/eval_records.md for the column schemas.

  eval_detections.jsonl — unit: one detected box (Tier B annotation)
  eval_candidates.jsonl — unit: one projected map traffic_light (front/back)
  eval_lamps.jsonl      — unit: one lamp of a detection (exploded state tokens)

Metric blocks (all views over the ledgers):
  A. detection profile by distance bin: candidate coverage, matched IoU, unknown
  B. temporal stability (from traffic_signal_re_timeseries.json)
  C. GT metrics (only when reviewed): state accuracy + element P/R, sliceable
     by signal_kind / distance / facing; FP (rejected), FN (manual boxes)
  D. optional --baseline <sidecar.json>: run-to-run comparison
  cuts: detections by signal_kind / facing / channel; lamps by color×shape and
        arrow direction (add any group-by you need on the jsonl files)

Outputs: build/tl_match/eval_report.{json,md} + eval_{detections,candidates,lamps}.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from state_tokens import elements_key, parse_state

DIST_BINS = [(0, 30), (30, 60), (60, 100), (100, 150)]
GT_STATUSES = {"accepted", "fixed"}


def bin_of(distance: float | None):
    if distance is None:
        return None
    for lo, hi in DIST_BINS:
        if lo <= distance < hi:
            return f"{lo}-{hi}m"
    return f">={DIST_BINS[-1][1]}m"


def ann_state(ann: dict) -> str:
    """Canonical detector state of a Tier B annotation (v2 or v1 fallback)."""
    a = ann["attributes"]
    raw = a.get("raw_state") or a.get("detector_signal") or a.get("state") or ""
    return elements_key(parse_state(raw)) or "unknown"


# --------------------------------------------------------- A: detection profile


MIN_DETECTABLE_PX = 8.0  # detector --min-box: below this a miss is expected


def detection_profile(match_report: dict, sidecar_by_token: dict):
    bins = defaultdict(lambda: {
        "map_candidates": 0, "detectable": 0, "detectable_matched": 0,
        "back": 0, "back_matched": 0,
        "too_small": 0, "too_small_matched": 0,
        "matched_iou": [], "unknown_matched": 0, "matched": 0})
    unmatched_detections = 0
    for frame in match_report["frames"]:
        for cand in frame.get("candidates", []):
            b = bins[bin_of(cand["distance_m"])]
            b["map_candidates"] += 1
            small = cand.get("proj_min_side_px") is not None and \
                cand["proj_min_side_px"] < MIN_DETECTABLE_PX
            if cand.get("facing") == "back":
                # the housing's back: lamps unreadable, an `unknown` box is
                # possible but absence is not a miss -> excluded from coverage
                b["back"] += 1
                b["back_matched"] += bool(cand["matched"])
            elif small:
                b["too_small"] += 1
                b["too_small_matched"] += bool(cand["matched"])
            else:
                b["detectable"] += 1
                b["detectable_matched"] += bool(cand["matched"])
        for pair in frame["pairs"]:
            if pair["map_traffic_light_id"] is None:
                unmatched_detections += 1
                continue
            b = bins[bin_of(pair["distance_m"])]
            b["matched"] += 1
            b["matched_iou"].append(pair["iou"])
            state = elements_key(parse_state(pair.get("signal", ""))) or "unknown"
            b["unknown_matched"] += state == "unknown"

    profile = {}
    for name in sorted(bins, key=lambda n: float(n.split("-")[0].lstrip(">=").rstrip("m"))):
        b = bins[name]
        ious = sorted(b["matched_iou"])
        profile[name] = {
            "map_candidates": b["map_candidates"],
            "detectable_candidates": b["detectable"],
            "candidate_coverage": round(b["detectable_matched"] / b["detectable"], 3)
                                  if b["detectable"] else None,
            "back_candidates": b["back"],
            "back_matched": b["back_matched"],
            "too_small_candidates": b["too_small"],
            "too_small_matched": b["too_small_matched"],
            "matched_detections": b["matched"],
            "matched_iou_median": round(ious[len(ious) // 2], 3) if ious else None,
            "unknown_rate": round(b["unknown_matched"] / b["matched"], 3) if b["matched"] else None,
        }
    return profile, unmatched_detections


# ------------------------------------------------------------- B: stability


def stability_metrics(timeseries: dict):
    groups = {}
    seen_ways = set()
    for s in timeseries["series"]:
        key = tuple(sorted(s["member_ways"]))
        if key in seen_ways:
            continue  # regulatory elements sharing the same heads duplicate the series
        seen_ways.add(key)
        obs = s["observations"]
        states = [o["state"] for o in obs]
        flips = sum(1 for a, b in zip(states, states[1:]) if a != b)
        groups["+".join(key)] = {
            "n_obs": len(obs),
            "n_segments": len(s["segments"]),
            "flips_per_obs": round(flips / max(len(obs) - 1, 1), 3),
            "single_frame_flips": sum("single_frame_flip" in o["flags"] for o in obs),
            "unknown_frac": round(sum(st == "unknown" for st in states) / max(len(states), 1), 3),
            "mean_confidence": round(sum(o.get("confidence", 0) for o in obs) / max(len(obs), 1), 3),
        }
    total_obs = sum(g["n_obs"] for g in groups.values())
    overall = {
        "head_groups": len(groups),
        "observations": total_obs,
        "single_frame_flips": sum(g["single_frame_flips"] for g in groups.values()),
        "mean_unknown_frac": round(sum(g["unknown_frac"] * g["n_obs"] for g in groups.values())
                                   / max(total_obs, 1), 3),
    }
    return {"overall": overall, "per_head_group": groups}


# ------------------------------------------------------------------ C: GT block


def element_set(state: str) -> set[tuple]:
    return {(e["color"], e["shape"], e["arrow"]) for e in parse_state(state)}


def gt_metrics(sidecar: dict, distance_by_token: dict):
    anns = sidecar["annotations"]
    reviewed = [a for a in anns
                if a["attributes"].get("review_status", "unchecked") != "unchecked"]
    if not reviewed:
        return None

    per_bin = defaultdict(lambda: {"n": 0, "state_exact": 0,
                                   "tp": 0, "fp": 0, "fn": 0})
    rejected = sum(1 for a in reviewed if a["attributes"]["review_status"] == "rejected")
    manual_added = sum(1 for a in anns if a["attributes"].get("source_type") == "manual")

    for a in reviewed:
        if a["attributes"]["review_status"] not in GT_STATUSES:
            continue
        gt_state = a["attributes"].get("state", "unknown")
        pred_state = ann_state(a)
        b = per_bin[bin_of(distance_by_token.get(a["token"]))]
        b["n"] += 1
        b["state_exact"] += elements_key(parse_state(gt_state)) == \
                            (elements_key(parse_state(pred_state)) or "unknown") or \
                            (gt_state == pred_state == "unknown")
        gt_el, pr_el = element_set(gt_state), element_set(pred_state)
        b["tp"] += len(gt_el & pr_el)
        b["fp"] += len(pr_el - gt_el)
        b["fn"] += len(gt_el - pr_el)

    result = {"reviewed": len(reviewed), "rejected_fp": rejected,
              "manual_added_fn": manual_added, "by_distance": {}}
    for name, b in sorted(per_bin.items(), key=lambda kv: str(kv[0])):
        prec = b["tp"] / max(b["tp"] + b["fp"], 1)
        rec = b["tp"] / max(b["tp"] + b["fn"], 1)
        result["by_distance"][str(name)] = {
            "n": b["n"],
            "state_accuracy": round(b["state_exact"] / max(b["n"], 1), 3),
            "element_precision": round(prec, 3),
            "element_recall": round(rec, 3),
        }
    return result


# -------------------------------------------------------- tidy record ledgers
#
# The report tables below are just *views*. The reusable artifacts are three
# long-format ledgers (one row = one unit of observation), so any later cut
# (pedestrian vs vehicle, per lamp color/shape/arrow, facing, channel, distance,
# state, matched/unmatched, reviewed/not) is a group-by, not new code:
#
#   detections  — unit: one detected box (Tier B annotation). precision side,
#                 state accuracy, unknown rate. dims: channel, signal_kind,
#                 facing, distance_bin, state, matched, review_status, subtype.
#   candidates  — unit: one projected map traffic_light (front/back). recall /
#                 coverage side. dims: distance_bin, facing, detectable, subtype.
#   lamps       — unit: one lamp of one detection (exploded from state tokens).
#                 dims: color, shape, arrow, is_arrow + the detection's dims.
#
# All three share join keys (sample_data_token, way_id) so they can be
# re-joined externally (pandas etc.). Files: build/tl_match/eval_{detections,
# candidates,lamps}.jsonl.


def signal_kind_of(state: str, attr_kind: str | None) -> str:
    if attr_kind:
        return attr_kind
    els = parse_state(state)
    if any(e["shape"] == "ped" for e in els):
        return "pedestrian"
    return "vehicle" if els else "unknown"


def build_detection_records(sidecar: dict, pair_index: dict) -> list[dict]:
    records = []
    for a in sidecar["annotations"]:
        attr = a["attributes"]
        state = ann_state(a)
        box = a["box2d"]
        pair = pair_index.get((a["sample_data_token"], tuple(round(float(v), 1) for v in box)))
        distance = pair["distance_m"] if pair else None
        review = attr.get("review_status", "unchecked")
        gt_state = attr.get("state") if review in GT_STATUSES else None
        records.append({
            "row_id": len(records),
            "det_token": a["token"],
            "sample_token": a["sample_token"],
            "sample_data_token": a["sample_data_token"],
            "channel": a["channel"],
            "timestamp": a.get("timestamp"),
            "det_box": box,
            "det_min_side_px": round(min(box[2] - box[0], box[3] - box[1]), 1),
            "detector_score": float(attr["detector_score"]) if attr.get("detector_score") else None,
            "state": state,
            "signal_kind": signal_kind_of(state, attr.get("signal_kind")),
            "n_lamps": len(parse_state(state)),
            "is_unknown": state == "unknown",
            "way_id": attr.get("map_traffic_light_id") or None,
            "regulatory_element_id": attr.get("regulatory_element_id") or None,
            "subtype": pair.get("map_subtype") if pair else None,
            "facing": attr.get("facing") or None,
            "matched": bool(attr.get("map_traffic_light_id")),
            "iou": pair["iou"] if pair else None,
            "distance_m": distance,
            "distance_bin": bin_of(distance),
            "review_status": review,
            "gt_state": None if gt_state is None else (elements_key(parse_state(gt_state)) or "unknown"),
            "state_correct": None if gt_state is None
                             else (elements_key(parse_state(gt_state)) == state),
        })
    return records


def build_candidate_records(match_report: dict) -> list[dict]:
    records = []
    for frame in match_report["frames"]:
        for c in frame.get("candidates", []):
            facing = c.get("facing") or None
            small = c.get("proj_min_side_px") is not None and c["proj_min_side_px"] < MIN_DETECTABLE_PX
            detectable = facing != "back" and not small
            records.append({
                "row_id": len(records),
                "sample_data_token": frame["sample_data_token"],
                "channel": frame["channel"],
                "way_id": c["way_id"],
                "distance_m": c["distance_m"],
                "distance_bin": bin_of(c["distance_m"]),
                "facing": facing,
                "facing_deg": c.get("facing_deg"),
                "proj_min_side_px": c.get("proj_min_side_px"),
                "too_small": small,
                "detectable": detectable,
                "matched": bool(c["matched"]),
            })
    return records


def build_lamp_records(detection_records: list[dict]) -> list[dict]:
    records = []
    for d in detection_records:
        for e in parse_state(d["state"]):
            records.append({
                "row_id": len(records),
                "det_token": d["det_token"],
                "channel": d["channel"],
                "signal_kind": d["signal_kind"],
                "facing": d["facing"],
                "distance_bin": d["distance_bin"],
                "matched": d["matched"],
                "way_id": d["way_id"],
                "color": e["color"],
                "shape": e["shape"],
                "arrow": e["arrow"],
                "is_arrow": e["shape"] == "arrow",
                "lamp_token": f"{e['color']}-{e['shape']}" + (f"-{e['arrow']}" if e["arrow"] else ""),
            })
    return records


def pivot(records: list[dict], group_by: list[str], metrics: dict) -> list[dict]:
    """Generic group-by over a ledger. `metrics` maps name -> fn(list[row])."""
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r.get(g) for g in group_by)].append(r)
    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        row = dict(zip(group_by, key))
        for name, fn in metrics.items():
            row[name] = fn(rs)
        rows.append(row)
    return rows


def _rate(pred):
    return lambda rs: round(sum(1 for r in rs if pred(r)) / len(rs), 3) if rs else None


def _median(field):
    def fn(rs):
        vals = sorted(r[field] for r in rs if r.get(field) is not None)
        return round(vals[len(vals) // 2], 3) if vals else None
    return fn


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- D: baseline


def run_summary(sidecar: dict):
    anns = sidecar["annotations"]
    states = Counter(ann_state(a) for a in anns)
    matched = sum(1 for a in anns if a["attributes"].get("map_traffic_light_id"))
    return {
        "annotations": len(anns),
        "map_matched_rate": round(matched / max(len(anns), 1), 3),
        "unknown_rate": round(states.get("unknown", 0) / max(len(anns), 1), 3),
        "state_hist_top": dict(states.most_common(8)),
    }


# ------------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--sidecar", default=Path("annotation/traffic_signal_2d_ann.json"), type=Path)
    parser.add_argument("--baseline", default=None, type=Path,
                        help="Another Tier B sidecar to compare against (e.g. fp32 backup).")
    parser.add_argument("--output", default=Path("build/tl_match/eval_report.json"), type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()

    sidecar = json.loads((root / args.sidecar).read_text())
    match_report = json.loads((root / "build/tl_match/match_report.json").read_text())
    timeseries = json.loads((root / "annotation/traffic_signal_re_timeseries.json").read_text())

    # index match_report pairs by (sample_data_token, rounded box) for the join
    box_key = lambda t, b: (t, tuple(round(float(v), 1) for v in b))
    pair_index = {}
    for frame in match_report["frames"]:
        for pair in frame["pairs"]:
            pair_index[box_key(frame["sample_data_token"], pair["detection_box"])] = pair
    distance_by_token = {a["token"]: (pair_index.get(box_key(a["sample_data_token"], a["box2d"])) or {}).get("distance_m")
                         for a in sidecar["annotations"]}

    # tidy ledgers (the reusable analysis artifacts) -----------------------
    det_records = build_detection_records(sidecar, pair_index)
    cand_records = build_candidate_records(match_report)
    lamp_records = build_lamp_records(det_records)
    out = root / args.output
    write_jsonl(out.with_name("eval_detections.jsonl"), det_records)
    write_jsonl(out.with_name("eval_candidates.jsonl"), cand_records)
    write_jsonl(out.with_name("eval_lamps.jsonl"), lamp_records)

    # default report views, all derived from the ledgers via pivot ---------
    det_metrics = {"n": len, "matched_rate": _rate(lambda r: r["matched"]),
                   "unknown_rate": _rate(lambda r: r["is_unknown"]),
                   "iou_median": _median("iou")}
    cov_metrics = {"candidates": len,
                   "detectable": lambda rs: sum(r["detectable"] for r in rs),
                   "coverage": lambda rs: (round(sum(r["matched"] for r in rs if r["detectable"])
                                                 / max(sum(r["detectable"] for r in rs), 1), 3))}
    lamp_metrics = {"n": len, "matched_rate": _rate(lambda r: r["matched"])}

    profile, unmatched_dets = detection_profile(match_report, distance_by_token)
    report = {
        "schema_version": "tlr_eval/v2",
        "inputs": {"sidecar": str(args.sidecar), "run": run_summary(sidecar)},
        "ledgers": {"detections": len(det_records), "candidates": len(cand_records),
                    "lamps": len(lamp_records),
                    "files": ["eval_detections.jsonl", "eval_candidates.jsonl", "eval_lamps.jsonl"]},
        "detection_profile_by_distance": profile,
        "unmatched_detections": unmatched_dets,
        "cuts": {
            "detections_by_signal_kind": pivot(det_records, ["signal_kind"], det_metrics),
            "detections_by_facing": pivot(det_records, ["facing"], det_metrics),
            "detections_by_channel": pivot(det_records, ["channel"], det_metrics),
            "coverage_by_signal_kind_distance": pivot(cand_records, ["distance_bin"], cov_metrics),
            "lamps_by_color_shape": pivot(lamp_records, ["color", "shape"], lamp_metrics),
            "lamps_by_arrow_dir": pivot([r for r in lamp_records if r["is_arrow"]],
                                        ["arrow"], lamp_metrics),
        },
        "stability": stability_metrics(timeseries),
        "gt": gt_metrics(sidecar, distance_by_token),
    }
    # when GT exists, state accuracy is sliceable on the same ledger dimensions
    gt_rows = [r for r in det_records if r["state_correct"] is not None]
    if gt_rows and report["gt"]:
        acc = {"n": len, "state_accuracy": _rate(lambda r: r["state_correct"])}
        report["gt"]["accuracy_by_signal_kind"] = pivot(gt_rows, ["signal_kind"], acc)
        report["gt"]["accuracy_by_distance"] = pivot(gt_rows, ["distance_bin"], acc)
        report["gt"]["accuracy_by_facing"] = pivot(gt_rows, ["facing"], acc)
    if args.baseline:
        report["baseline"] = {"path": str(args.baseline),
                              "run": run_summary(json.loads((root / args.baseline).read_text()))}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    md = [f"# TLR eval report", "",
          f"sidecar: `{args.sidecar}` — {report['inputs']['run']['annotations']} anns, "
          f"map matched {report['inputs']['run']['map_matched_rate']:.1%}, "
          f"unknown {report['inputs']['run']['unknown_rate']:.1%}", "",
          "## Detection profile by distance", "",
          "(coverage = matched / front-facing detectable candidates; back-facing, "
          f"sub-{MIN_DETECTABLE_PX:.0f}px and edge-on candidates are excluded from the denominator)", "",
          "| bin | candidates | front detectable | coverage | back (matched) | too small | matched dets | IoU med | unknown rate |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, b in report["detection_profile_by_distance"].items():
        md.append(f"| {name} | {b['map_candidates']} | {b['detectable_candidates']} | "
                  f"{b['candidate_coverage']} | {b['back_candidates']} ({b['back_matched']}) | "
                  f"{b['too_small_candidates']} | "
                  f"{b['matched_detections']} | {b['matched_iou_median']} | {b['unknown_rate']} |")
    def cut_table(title, rows, cols):
        block = ["", f"## {title}", "", "| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in rows:
            block.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return block

    md += ["", "_Cuts below are pivot views of the tidy ledgers "
           "(`eval_detections.jsonl` / `eval_candidates.jsonl` / `eval_lamps.jsonl`); "
           "slice any other way with a group-by on those files._"]
    md += cut_table("Detections by signal kind", report["cuts"]["detections_by_signal_kind"],
                    ["signal_kind", "n", "matched_rate", "unknown_rate", "iou_median"])
    md += cut_table("Detections by facing", report["cuts"]["detections_by_facing"],
                    ["facing", "n", "matched_rate", "unknown_rate", "iou_median"])
    md += cut_table("Detections by channel", report["cuts"]["detections_by_channel"],
                    ["channel", "n", "matched_rate", "unknown_rate", "iou_median"])
    md += cut_table("Lamps by color × shape", report["cuts"]["lamps_by_color_shape"],
                    ["color", "shape", "n", "matched_rate"])
    md += cut_table("Arrow lamps by direction", report["cuts"]["lamps_by_arrow_dir"],
                    ["arrow", "n", "matched_rate"])

    st = report["stability"]["overall"]
    md += ["", "## Stability", "",
           f"- head groups: {st['head_groups']}, observations: {st['observations']}",
           f"- single-frame flips: {st['single_frame_flips']}",
           f"- mean unknown fraction: {st['mean_unknown_frac']}"]
    if report["gt"]:
        md += ["", "## GT metrics (reviewed)", "",
               f"- reviewed: {report['gt']['reviewed']}, rejected(FP): {report['gt']['rejected_fp']}, "
               f"manual added(FN): {report['gt']['manual_added_fn']}",
               "| bin | n | state acc | elem P | elem R |", "|---|---:|---:|---:|---:|"]
        for name, b in report["gt"]["by_distance"].items():
            md.append(f"| {name} | {b['n']} | {b['state_accuracy']} | "
                      f"{b['element_precision']} | {b['element_recall']} |")
    else:
        md += ["", "_GT metrics inactive: no reviewed annotations "
               "(review_status all unchecked)._"]
    if args.baseline:
        md += ["", "## Baseline comparison", "",
               f"baseline `{args.baseline}`: {json.dumps(report['baseline']['run'], ensure_ascii=False)}"]
    (out.with_suffix(".md")).write_text("\n".join(md) + "\n")

    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.md')}")
    print(json.dumps({"run": report["inputs"]["run"],
                      "stability": st,
                      "gt_active": report["gt"] is not None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
