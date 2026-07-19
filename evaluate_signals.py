#!/usr/bin/env python3
"""L6 evaluation: metrics over Tier B + RE time series (pure analysis, no inference).

Positioning (2026-07-19): GT is always human-made; this layer is the metrics
engine. It always computes GT-free metrics, and the GT-dependent block
activates automatically when the Tier B sidecar contains reviewed annotations
(`review_status` in {accepted, fixed, rejected}).

Metric blocks:
  A. detection profile by distance bin (from build/tl_match/match_report.json):
     map-candidate coverage (matched vs undetected), matched IoU, unknown rate
  B. temporal stability (from annotation/traffic_signal_re_timeseries.json):
     state flips per observation, single-frame flips, unknown fraction
  C. GT metrics (only when reviewed): prediction = `raw_state`,
     GT = reviewed `state` of accepted/fixed boxes; exact-state accuracy and
     element-level precision/recall, by distance bin; FP rate (rejected),
     FN count (manual boxes)
  D. optional --baseline <sidecar.json>: run-to-run comparison (e.g. fp32 vs
     int8 labels) on matched rate / unknown rate / state histogram

Outputs: build/tl_match/eval_report.json (+ eval_report.md summary).
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


def detection_profile(match_report: dict, sidecar_by_token: dict):
    bins = defaultdict(lambda: {
        "map_candidates": 0, "candidates_matched": 0,
        "matched_iou": [], "unknown_matched": 0, "matched": 0})
    unmatched_detections = 0
    for frame in match_report["frames"]:
        for cand in frame.get("candidates", []):
            b = bins[bin_of(cand["distance_m"])]
            b["map_candidates"] += 1
            b["candidates_matched"] += bool(cand["matched"])
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
            "candidate_coverage": round(b["candidates_matched"] / b["map_candidates"], 3)
                                  if b["map_candidates"] else None,
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

    # distance per Tier B token, via match_report pairs (same box in same frame)
    distance_by_token = {}
    box_key = lambda t, b: (t, tuple(round(float(v), 1) for v in b))
    pair_distance = {}
    for frame in match_report["frames"]:
        for pair in frame["pairs"]:
            pair_distance[box_key(frame["sample_data_token"], pair["detection_box"])] = pair["distance_m"]
    for a in sidecar["annotations"]:
        distance_by_token[a["token"]] = pair_distance.get(
            box_key(a["sample_data_token"], a["box2d"]))

    profile, unmatched_dets = detection_profile(match_report, distance_by_token)
    report = {
        "schema_version": "tlr_eval/v1",
        "inputs": {"sidecar": str(args.sidecar), "run": run_summary(sidecar)},
        "detection_profile_by_distance": profile,
        "unmatched_detections": unmatched_dets,
        "stability": stability_metrics(timeseries),
        "gt": gt_metrics(sidecar, distance_by_token),
    }
    if args.baseline:
        report["baseline"] = {"path": str(args.baseline),
                              "run": run_summary(json.loads((root / args.baseline).read_text()))}

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    md = [f"# TLR eval report", "",
          f"sidecar: `{args.sidecar}` — {report['inputs']['run']['annotations']} anns, "
          f"map matched {report['inputs']['run']['map_matched_rate']:.1%}, "
          f"unknown {report['inputs']['run']['unknown_rate']:.1%}", "",
          "## Detection profile by distance", "",
          "| bin | map candidates | coverage | matched dets | IoU med | unknown rate |",
          "|---|---:|---:|---:|---:|---:|"]
    for name, b in report["detection_profile_by_distance"].items():
        md.append(f"| {name} | {b['map_candidates']} | {b['candidate_coverage']} | "
                  f"{b['matched_detections']} | {b['matched_iou_median']} | {b['unknown_rate']} |")
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
