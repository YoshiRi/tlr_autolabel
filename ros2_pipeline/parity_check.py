#!/usr/bin/env python3
"""L5 parity check: compare the live ROS2 pipeline output against the offline
(tlr_autolabel.py int8) output on the SAME frames.

Both sides are `tlr_autolabel/v1` per-frame JSON (same schema). For each frame
present in both directories:
  1. match signals by box IoU (greedy, --iou threshold)
  2. BBOX parity: matched / offline-only (miss) / ros2-only (extra) / IoU stats
  3. STATE parity: on IoU-matched pairs, compare canonical order-independent
     state (via state_tokens) — state can only be right where the box matched,
     so state agreement is reported over matched pairs.

Acceptance (STATUS/PLAN L5): the live int8 graph agrees with the offline int8
pipeline. Use --mode to focus on detection (bbox) or classification (state).

Usage:
  python3 parity_check.py <offline_dir> <ros2_dir> [--iou 0.5] [--mode both] [--out report.json]
"""
import argparse
import glob
import json
import os
import sys

# reuse the project's canonical state parsing (single source of truth)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tlr_autolabel.core.state_tokens import parse_state, elements_key  # noqa: E402


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def state_key(sig):
    """Canonical order-independent state string for a v1 signal ('' if no state).
    v1 uses `state`; older sidecars used `signal` -- accept either."""
    return elements_key(parse_state(sig.get("state", sig.get("signal", ""))))


def greedy_match(off, ros, thr):
    """Greedy IoU matching. Returns (pairs, off_only_idx, ros_only_idx).
    pairs: list of (i_off, j_ros, iou)."""
    cand = []
    for i, a in enumerate(off):
        for j, b in enumerate(ros):
            v = iou(a["box_xyxy"], b["box_xyxy"])
            if v >= thr:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    used_o, used_r, pairs = set(), set(), []
    for v, i, j in cand:
        if i in used_o or j in used_r:
            continue
        used_o.add(i); used_r.add(j); pairs.append((i, j, v))
    off_only = [i for i in range(len(off)) if i not in used_o]
    ros_only = [j for j in range(len(ros)) if j not in used_r]
    return pairs, off_only, ros_only


def load_dir(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        name = os.path.splitext(os.path.basename(f))[0]
        if name.startswith("_"):
            continue
        try:
            out[name] = json.load(open(f))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("offline_dir")
    ap.add_argument("ros2_dir")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold to call two boxes the same")
    ap.add_argument("--mode", choices=["bbox", "state", "both"], default="both")
    ap.add_argument("--out", default=None, help="write a JSON report")
    ap.add_argument("--show-mismatch", type=int, default=15, help="print up to N state mismatches")
    args = ap.parse_args()

    off_all, ros_all = load_dir(args.offline_dir), load_dir(args.ros2_dir)
    common = sorted(set(off_all) & set(ros_all))
    only_off = sorted(set(off_all) - set(ros_all))
    only_ros = sorted(set(ros_all) - set(off_all))

    # bbox aggregates
    n_off = n_ros = n_match = 0
    iou_sum = 0.0
    # state aggregates (over matched pairs)
    n_state_eval = n_state_agree = 0
    mismatches = []

    per_frame = []
    for name in common:
        off = off_all[name]["signals"]
        ros = ros_all[name]["signals"]
        pairs, off_only, ros_only = greedy_match(off, ros, args.iou)
        n_off += len(off); n_ros += len(ros); n_match += len(pairs)
        iou_sum += sum(v for _, _, v in pairs)
        fs_agree = fs_eval = 0
        for i, j, v in pairs:
            ks, kr = state_key(off[i]), state_key(ros[j])
            fs_eval += 1; n_state_eval += 1
            if ks == kr:
                fs_agree += 1; n_state_agree += 1
            else:
                if len(mismatches) < 10_000:
                    mismatches.append({"frame": name, "iou": round(v, 3),
                                       "offline": ks or "(none)", "ros2": kr or "(none)",
                                       "box_offline": off[i]["box_xyxy"], "box_ros2": ros[j]["box_xyxy"]})
        per_frame.append({"frame": name, "offline_signals": len(off), "ros2_signals": len(ros),
                          "matched": len(pairs), "offline_only": len(off_only),
                          "ros2_only": len(ros_only), "state_agree": fs_agree, "state_eval": fs_eval})

    # ---- report ----
    prec = n_match / n_ros if n_ros else 0.0     # of ros2 boxes, how many matched offline
    rec = n_match / n_off if n_off else 0.0      # of offline boxes, how many matched ros2
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    state_acc = n_state_agree / n_state_eval if n_state_eval else 0.0
    report = {
        "iou_threshold": args.iou,
        "frames": {"common": len(common), "offline_only": only_off, "ros2_only": only_ros},
        "bbox": {"offline_boxes": n_off, "ros2_boxes": n_ros, "matched": n_match,
                 "offline_only(miss)": n_off - n_match, "ros2_only(extra)": n_ros - n_match,
                 "mean_iou_matched": round(iou_sum / n_match, 4) if n_match else 0.0,
                 "recall(vs offline)": round(rec, 4), "precision(vs offline)": round(prec, 4),
                 "f1": round(f1, 4)},
        "state": {"pairs_evaluated": n_state_eval, "agree": n_state_agree,
                  "agreement": round(state_acc, 4)},
    }

    print("=" * 60)
    print(f"L5 PARITY: offline={args.offline_dir}  ros2={args.ros2_dir}")
    print(f"frames: common={len(common)} offline_only={len(only_off)} ros2_only={len(only_ros)}")
    if args.mode in ("bbox", "both"):
        b = report["bbox"]
        print("\n[BBOX]")
        print(f"  offline={b['offline_boxes']} ros2={b['ros2_boxes']} matched={b['matched']} "
              f"(IoU>= {args.iou})")
        print(f"  miss(offline-only)={b['offline_only(miss)']} extra(ros2-only)={b['ros2_only(extra)']}")
        print(f"  mean IoU (matched)={b['mean_iou_matched']}")
        print(f"  recall={b['recall(vs offline)']}  precision={b['precision(vs offline)']}  f1={b['f1']}")
    if args.mode in ("state", "both"):
        s = report["state"]
        print("\n[STATE] (on IoU-matched pairs)")
        print(f"  pairs={s['pairs_evaluated']} agree={s['agree']} agreement={s['agreement']}")
        for m in mismatches[:args.show_mismatch]:
            print(f"    {m['frame']}: offline='{m['offline']}' vs ros2='{m['ros2']}' (IoU {m['iou']})")
        if len(mismatches) > args.show_mismatch:
            print(f"    ... {len(mismatches) - args.show_mismatch} more mismatches")
    print("=" * 60)

    if args.out:
        report["per_frame"] = per_frame
        report["state_mismatches"] = mismatches
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
