"""GT-free, map-free comparison of Tier A runs ("naive output comparison").

`compare_runs.py` already compares label directories, but only against a
lanelet2 map inside a T4 dataset (`candidate_coverage`). For an arbitrary image
folder, a video or a bag there is no map and no GT, so this module answers the
question that *is* answerable there:

- what each configuration outputs (counts, score/size distributions, states),
- how much two configurations agree, and precisely where they disagree,
- how steady each one is over time (a video/bag's cheapest quality signal),
- optionally, how each one scores against the consensus of the others.

This is L6: it reads Tier A and never re-runs inference. Consensus is a
convenience, not a ground truth — configurations that share a detector share its
errors, so a majority vote among them is correlated by construction. The report
says so, in the report.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tlr_autolabel.core.state_tokens import parse_state
from tlr_autolabel.eval.l1_vs_t4 import canonical_state, div, iou, match_frame, signal_kind

REPORT_SCHEMA = "tlr_compare_naive/v1"
DEFAULT_IOU_THR = 0.5
STABILITY_IOU_THR = 0.3


# --------------------------------------------------------------------- loading


@dataclass
class Run:
    name: str
    labels_dir: str
    frames: dict = field(default_factory=dict)   # frame_key -> record
    meta: dict = field(default_factory=dict)

    @property
    def detections(self) -> int:
        return sum(len(f["signals"]) for f in self.frames.values())


def load_run(labels_dir, name=None) -> Run:
    """Read one Tier A label directory.

    The join key is the label file's path relative to the directory (i.e. the
    frame id the source assigned). `sample_data_token` is kept for cross-checking
    but is not required — a video or a bag has none."""
    root = Path(labels_dir)
    if not root.is_dir():
        raise SystemExit(f"not a label directory: {labels_dir}")
    run = Run(name=name or root.name, labels_dir=str(root))
    for path in sorted(root.rglob("*.json")):
        if path.name in ("frames.json", "run_manifest.json") or \
                path.name.endswith(".viz.json"):
            continue
        payload = json.loads(path.read_text())
        if "signals" not in payload:
            continue
        key = str(path.relative_to(root).with_suffix(""))
        signals = []
        for i, s in enumerate(payload["signals"]):
            box = [float(v) for v in s["box_xyxy"]]
            state = canonical_state(s.get("state") or "unknown")
            signals.append({
                "id": s.get("signal_id") or f"{key}-{i:02d}",
                "box": box,
                "state": state,
                "signal_kind": signal_kind(state),
                "score": s.get("detector_score"),
                "min_side": min(box[2] - box[0], box[3] - box[1]),
            })
        run.frames[key] = {
            "frame_key": key,
            "channel": payload.get("channel"),
            "frame_index": payload.get("frame_index"),
            "timestamp_us": payload.get("timestamp_us"),
            "sample_data_token": payload.get("sample_data_token"),
            "image_realpath": payload.get("image_realpath"),
            "image": payload.get("image"),
            "width": payload.get("width"),
            "height": payload.get("height"),
            "timing_ms": payload.get("timing_ms"),
            "signals": signals,
        }
        if not run.meta:
            run.meta = payload.get("meta") or {}
    if not run.frames:
        raise SystemExit(f"no tlr_autolabel/v1 label files under {labels_dir}")
    return run


def load_runs(specs) -> list:
    """`specs` are `name=path` strings (or bare paths, named after the dir)."""
    runs = []
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = os.path.basename(os.path.normpath(spec)), spec
        runs.append(load_run(path, name=name))
    return runs


def load_runs_from_manifest(manifest_path) -> list:
    """Load every combo of a `run_manifest.json` written by the matrix runner."""
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest_path).parent
    runs = []
    for combo in manifest.get("combos", []):
        runs.append(load_run(root / combo["labels_dir"], name=combo["name"]))
    if not runs:
        raise SystemExit(f"{manifest_path}: no combos to compare")
    return runs


# --------------------------------------------------------------------- helpers


def _percentiles(values, ps=(10, 50, 90)):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {f"p{p}": None for p in ps}
    out = {}
    for p in ps:
        idx = min(len(vals) - 1, max(0, int(round((p / 100.0) * (len(vals) - 1)))))
        out[f"p{p}"] = round(vals[idx], 4)
    return out


def _elements(state):
    return {(e["color"], e["shape"], e["arrow"]) for e in parse_state(state)}


def frames_in_order(run: Run):
    """Frames grouped by channel, ordered the way they were captured."""
    by_channel = defaultdict(list)
    for rec in run.frames.values():
        by_channel[rec.get("channel")].append(rec)
    for channel, recs in by_channel.items():
        recs.sort(key=lambda r: (r.get("frame_index") if r.get("frame_index") is not None
                                 else 0, r["frame_key"]))
        yield channel, recs


# --------------------------------------------------------------------- metrics


def summarize_run(run: Run, iou_thr=STABILITY_IOU_THR) -> dict:
    scores, sides, states = [], [], Counter()
    totals, detectors, classifiers = [], [], []
    for rec in run.frames.values():
        for s in rec["signals"]:
            scores.append(s["score"])
            sides.append(s["min_side"])
            states[s["state"]] += 1
        timing = rec.get("timing_ms") or {}
        totals.append(timing.get("total"))
        detectors.append(timing.get("detector"))
        classifiers.append(timing.get("classifier"))
    n_frames = len(run.frames)
    n_det = run.detections
    return {
        "frames": n_frames,
        "detections": n_det,
        "detections_per_frame": div(n_det, n_frames),
        "empty_frames": sum(1 for f in run.frames.values() if not f["signals"]),
        "unknown_rate": div(states.get("unknown", 0), n_det),
        "state_counts": [{"state": k, "n": v} for k, v in states.most_common(15)],
        "detector_score": _percentiles(scores),
        "box_min_side_px": _percentiles(sides),
        "timing_ms": {
            "total": _percentiles(totals, ps=(50, 90, 99)),
            "detector": _percentiles(detectors, ps=(50,)),
            "classifier": _percentiles(classifiers, ps=(50,)),
        },
        "stability": stability(run, iou_thr=iou_thr),
        "meta": {k: run.meta.get(k) for k in (
            "preset", "detector", "detector_type", "detector_sha256", "classifier",
            "classifier_type", "classifier_sha256", "tiles", "det_score_thr",
            "det_nms_thr", "cls_score_thr", "crop_pad", "min_box")},
    }


def stability(run: Run, iou_thr=STABILITY_IOU_THR) -> dict:
    """Frame-to-frame steadiness, without a map or GT.

    A detection that survives into the next frame is linked by IoU; how often
    that link exists at all is `frame_to_frame_match_rate`, and how often the
    state changes across a link is `state_flip_rate`. A configuration that
    flickers is worse for downstream review even when its per-frame counts look
    identical to a steadier one."""
    links = flips = 0
    pair_matched = pair_possible = 0
    for _channel, recs in frames_in_order(run):
        for prev, cur in zip(recs, recs[1:]):
            pairs, _un_prev, _un_cur = match_frame(prev["signals"], cur["signals"], iou_thr)
            pair_matched += len(pairs)
            pair_possible += max(len(prev["signals"]), len(cur["signals"]))
            for pi, ci, _v in pairs:
                links += 1
                if prev["signals"][pi]["state"] != cur["signals"][ci]["state"]:
                    flips += 1
    return {
        "frame_pairs_compared": pair_possible,
        "frame_to_frame_match_rate": div(pair_matched, pair_possible),
        "tracked_links": links,
        "state_flip_rate": div(flips, links),
    }


def compare_pair(ref: Run, other: Run, iou_thr=DEFAULT_IOU_THR) -> dict:
    """How much `other` agrees with `ref`, and where it does not."""
    common = sorted(set(ref.frames) & set(other.frames))
    matched = only_ref = only_other = 0
    state_same = state_known_n = state_known_same = 0
    ious, only_ref_scores, only_other_scores = [], [], []
    elem_tp = elem_ref_only = elem_other_only = 0
    confusion = Counter()
    per_frame = []
    for key in common:
        a = ref.frames[key]["signals"]
        b = other.frames[key]["signals"]
        pairs, a_un, b_un = match_frame(a, b, iou_thr)
        matched += len(pairs)
        only_ref += len(a_un)
        only_other += len(b_un)
        frame_mismatch = len(a_un) + len(b_un)
        for ai, bi, value in pairs:
            ious.append(value)
            sa, sb = a[ai]["state"], b[bi]["state"]
            same = sa == sb
            state_same += same
            if not (sa == "unknown" and sb == "unknown"):
                state_known_n += 1
                state_known_same += same
            if not same:
                confusion[(sa, sb)] += 1
                frame_mismatch += 1
            ea, eb = _elements(sa), _elements(sb)
            elem_tp += len(ea & eb)
            elem_ref_only += len(ea - eb)
            elem_other_only += len(eb - ea)
        for i in a_un:
            only_ref_scores.append(a[i]["score"])
        for i in b_un:
            only_other_scores.append(b[i]["score"])
        if frame_mismatch:
            per_frame.append({"frame_key": key, "disagreements": frame_mismatch,
                              "only_ref": len(a_un), "only_other": len(b_un)})
    union = matched + only_ref + only_other
    per_frame.sort(key=lambda r: (-r["disagreements"], r["frame_key"]))
    return {
        "reference": ref.name,
        "candidate": other.name,
        "frames_compared": len(common),
        "frames_only_in_reference": len(set(ref.frames) - set(other.frames)),
        "frames_only_in_candidate": len(set(other.frames) - set(ref.frames)),
        "matched": matched,
        "only_in_reference": only_ref,
        "only_in_candidate": only_other,
        "box_agreement": div(matched, union),
        "matched_iou": _percentiles(ious, ps=(50, 90)),
        "state_agreement": div(state_same, matched),
        "state_agreement_excluding_both_unknown": div(state_known_same, state_known_n),
        "element_agreement": {
            "shared": elem_tp,
            "reference_only": elem_ref_only,
            "candidate_only": elem_other_only,
        },
        "only_in_reference_score": _percentiles(only_ref_scores, ps=(50,)),
        "only_in_candidate_score": _percentiles(only_other_scores, ps=(50,)),
        "top_state_disagreements": [
            {"reference": a, "candidate": b, "n": n}
            for (a, b), n in confusion.most_common(15)],
        "worst_frames": per_frame[:25],
    }


def consensus(runs, iou_thr=DEFAULT_IOU_THR, min_support=None) -> dict:
    """Majority-vote pseudo reference across >= 3 configurations.

    Correlated by construction (configurations sharing a detector share its
    misses), so this ranks configurations *relative to the pack*, not against
    reality. Useful to spot the odd one out; not evidence of accuracy."""
    if len(runs) < 3:
        return {"available": False,
                "reason": "consensus needs at least 3 configurations"}
    support_needed = min_support or (len(runs) // 2 + 1)
    all_keys = sorted(set().union(*(set(r.frames) for r in runs)))
    stats = {r.name: {"matched": 0, "state_ok": 0, "detections": 0} for r in runs}
    clusters_total = 0
    for key in all_keys:
        items = []
        for run in runs:
            for s in run.frames.get(key, {}).get("signals", []):
                items.append((run.name, s))
                stats[run.name]["detections"] += 1
        items.sort(key=lambda it: -(it[1]["score"] or 0.0))
        used = [False] * len(items)
        for i, (name_i, sig_i) in enumerate(items):
            if used[i]:
                continue
            cluster = [(name_i, sig_i)]
            used[i] = True
            seen = {name_i}
            for j in range(i + 1, len(items)):
                name_j, sig_j = items[j]
                if used[j] or name_j in seen:
                    continue
                if iou(sig_i["box"], sig_j["box"]) >= iou_thr:
                    used[j] = True
                    seen.add(name_j)
                    cluster.append((name_j, sig_j))
            if len(cluster) < support_needed:
                continue
            clusters_total += 1
            votes = Counter(s["state"] for _n, s in cluster)
            top = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
            for name, sig in cluster:
                stats[name]["matched"] += 1
                stats[name]["state_ok"] += (sig["state"] == top)
    per_run = {}
    for run in runs:
        st = stats[run.name]
        per_run[run.name] = {
            "consensus_recall": div(st["matched"], clusters_total),
            "consensus_precision": div(st["matched"], st["detections"]),
            "state_agreement_with_consensus": div(st["state_ok"], st["matched"]),
        }
    return {"available": True, "min_support": support_needed,
            "consensus_detections": clusters_total, "per_run": per_run,
            "caveat": "configurations that share a detector share its errors; "
                      "a majority vote among them is correlated by construction"}


def shared_models(runs) -> list:
    """Groups of configurations running the identical detector file (by sha256
    when recorded, else by name) — i.e. the ones whose errors correlate."""
    groups = defaultdict(list)
    for run in runs:
        key = run.meta.get("detector_sha256") or run.meta.get("detector")
        groups[key].append(run.name)
    return [{"detector": k, "configs": v} for k, v in groups.items() if len(v) > 1]


def compare(runs, reference=None, iou_thr=DEFAULT_IOU_THR, with_consensus=True) -> dict:
    """Full comparison report over >= 1 Tier A runs."""
    if not runs:
        raise SystemExit("nothing to compare")
    names = [r.name for r in runs]
    ref_name = reference or names[0]
    if ref_name not in names:
        raise SystemExit(f"reference {ref_name!r} is not one of: {', '.join(names)}")
    ref = next(r for r in runs if r.name == ref_name)
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "iou_threshold": iou_thr,
        "reference": ref_name,
        "runs": {r.name: summarize_run(r) for r in runs},
        "pairwise": [compare_pair(ref, r, iou_thr) for r in runs if r.name != ref_name],
        "shared_detectors": shared_models(runs),
    }
    if with_consensus:
        report["consensus"] = consensus(runs, iou_thr=iou_thr)
    frame_sets = {r.name: set(r.frames) for r in runs}
    common = set.intersection(*frame_sets.values()) if frame_sets else set()
    report["frames"] = {
        "common": len(common),
        "per_run": {n: len(s) for n, s in frame_sets.items()},
        "warning": ("configurations did not see the same frames — compare only the "
                    "common ones" if any(len(s) != len(common) for s in frame_sets.values())
                    else None),
    }
    return report


# ---------------------------------------------------------------------- report


def _fmt(value, nd=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{round(value, nd)}"
    return str(value)


def write_markdown(report: dict, path) -> str:
    runs = report["runs"]
    names = list(runs)
    lines = [
        "# Naive output comparison (GT-free, map-free)",
        "",
        f"reference: `{report['reference']}` | IoU threshold: "
        f"{report['iou_threshold']} | common frames: {report['frames']['common']}",
        "",
        "What this can and cannot say: it measures **what the configurations "
        "output and where they differ**, not which one is right. Without GT, a "
        "higher detection count is not better and a disagreement is not an error.",
        "",
        "## Per configuration",
        "",
        "| config | frames | dets | dets/frame | unknown | score p50 | "
        "box p50 (px) | ms p50 | f2f match | state flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in names:
        r = runs[n]
        st = r["stability"]
        lines.append(
            f"| {n} | {r['frames']} | {r['detections']} | "
            f"{_fmt(r['detections_per_frame'])} | {_fmt(r['unknown_rate'])} | "
            f"{_fmt(r['detector_score']['p50'])} | {_fmt(r['box_min_side_px']['p50'])} | "
            f"{_fmt(r['timing_ms']['total']['p50'])} | "
            f"{_fmt(st['frame_to_frame_match_rate'])} | {_fmt(st['state_flip_rate'])} |")

    lines += ["", "## Models", "",
              "| config | preset | detector | type | tiles | score thr | classifier |",
              "|---|---|---|---|---|---:|---|"]
    for n in names:
        m = runs[n]["meta"]
        lines.append(f"| {n} | {_fmt(m.get('preset'))} | {_fmt(m.get('detector'))} | "
                     f"{_fmt(m.get('detector_type'))} | {_fmt(m.get('tiles'))} | "
                     f"{_fmt(m.get('det_score_thr'))} | {_fmt(m.get('classifier'))} |")

    if report["pairwise"]:
        lines += ["", f"## Agreement with `{report['reference']}`", "",
                  "| config | matched | only in ref | only in cand | box agreement | "
                  "IoU p50 | state agreement | state agr. (excl. both-unknown) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for p in report["pairwise"]:
            lines.append(
                f"| {p['candidate']} | {p['matched']} | {p['only_in_reference']} | "
                f"{p['only_in_candidate']} | {_fmt(p['box_agreement'])} | "
                f"{_fmt(p['matched_iou']['p50'])} | {_fmt(p['state_agreement'])} | "
                f"{_fmt(p['state_agreement_excluding_both_unknown'])} |")
        lines += ["", "Median detector score of the disagreeing boxes — a low value "
                      "means the difference is marginal detections, a high value means "
                      "the configurations genuinely see different things:", "",
                  "| config | only-in-ref score p50 | only-in-cand score p50 |",
                  "|---|---:|---:|"]
        for p in report["pairwise"]:
            lines.append(f"| {p['candidate']} | "
                         f"{_fmt(p['only_in_reference_score']['p50'])} | "
                         f"{_fmt(p['only_in_candidate_score']['p50'])} |")
        for p in report["pairwise"]:
            if not p["top_state_disagreements"]:
                continue
            lines += ["", f"### `{report['reference']}` vs `{p['candidate']}`: "
                          "state disagreements on matched boxes", "",
                      "| reference | candidate | n |", "|---|---|---:|"]
            for row in p["top_state_disagreements"]:
                lines.append(f"| {row['reference']} | {row['candidate']} | {row['n']} |")

    cons = report.get("consensus") or {}
    if cons.get("available"):
        lines += ["", "## Versus consensus (pseudo reference)", "",
                  f"majority vote of {len(names)} configurations, support >= "
                  f"{cons['min_support']}, {cons['consensus_detections']} consensus "
                  "detections", "",
                  f"> {cons['caveat']}", "",
                  "| config | consensus recall | consensus precision | state agreement |",
                  "|---|---:|---:|---:|"]
        for n in names:
            c = cons["per_run"][n]
            lines.append(f"| {n} | {_fmt(c['consensus_recall'])} | "
                         f"{_fmt(c['consensus_precision'])} | "
                         f"{_fmt(c['state_agreement_with_consensus'])} |")
    elif cons:
        lines += ["", f"## Versus consensus — skipped: {cons.get('reason')}"]

    if report["shared_detectors"]:
        lines += ["", "## Correlated configurations", "",
                  "These run the identical detector, so their errors are not "
                  "independent (relevant when reading the consensus rows):", ""]
        for group in report["shared_detectors"]:
            lines.append(f"- `{group['detector']}`: {', '.join(group['configs'])}")

    if report["frames"].get("warning"):
        lines += ["", f"> **{report['frames']['warning']}**",
                  "", "| config | frames |", "|---|---:|"]
        for n, v in report["frames"]["per_run"].items():
            lines.append(f"| {n} | {v} |")

    text = "\n".join(lines) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return text


def worst_frames(report: dict, limit=20) -> list:
    """Frame keys ranked by total disagreement across all pairs — what to look
    at first, and what the grid renderer draws."""
    totals = Counter()
    for pair in report["pairwise"]:
        for row in pair["worst_frames"]:
            totals[row["frame_key"]] += row["disagreements"]
    return [k for k, _n in totals.most_common(limit)]
