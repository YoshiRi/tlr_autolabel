#!/usr/bin/env python3
"""Compare detector configs on GT-free, map-referenced metrics.

Each config is a tlr_autolabel label directory (one L1 run's per-frame JSON).
For each, this runs the L3 matcher against the same lanelet2 map and reads back
the detector-recall proxy that is comparable across configs WITHOUT human GT:

  candidate_coverage = matched / front-facing detectable map candidates,
                       by distance bin  (higher = detector found more real
                       signals the map says are visible)

plus detection count, map-match rate, unknown rate, matched-IoU median, and
unmatched detections (a false-positive-ish proxy). map-fill / interpolation do
not affect candidate matching, so coverage reflects the detector alone.

Usage:
  python3 compare_runs.py --dataset-root <ds> \
      name1=/path/to/labels1 name2=/path/to/labels2 ...

Writes build/tl_match/compare_report.md (+ .json) and prints the table.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST_BINS = ["0-30m", "30-60m", "60-100m", "100-150m", ">=150m"]


def run_match(dataset_root: Path, labels: Path, out_report: Path):
    subprocess.run(
        [sys.executable, str(HERE / "match_traffic_lights.py"),
         "--dataset-root", str(dataset_root),
         "--autolabel-dir", str(labels),
         "--output", str(out_report.with_name(out_report.stem + "_sidecar.json")),
         "--report", str(out_report)],
        check=True, stdout=subprocess.DEVNULL)


def profile(report_path: Path):
    """coverage/unknown/iou by distance bin + overall counts from a match_report."""
    sys.path.insert(0, str(HERE))
    from evaluate_signals import detection_profile  # reuse the canonical logic
    rep = json.loads(report_path.read_text())
    prof, unmatched = detection_profile(rep, {})
    n_det = sum(f["n_detections"] for f in rep["frames"])
    matched = sum(1 for f in rep["frames"] for p in f["pairs"]
                  if p["map_traffic_light_id"] is not None)
    return {"profile": prof, "unmatched_detections": unmatched,
            "detections": n_det, "matched_detections": matched,
            "map_match_rate": round(matched / max(n_det, 1), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, type=Path)
    ap.add_argument("configs", nargs="+", help="name=/path/to/label_dir ...")
    ap.add_argument("--output", default=Path("build/tl_match/compare_report.json"), type=Path)
    args = ap.parse_args()
    root = args.dataset_root.resolve()

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for spec in args.configs:
            name, _, path = spec.partition("=")
            if not path:
                raise SystemExit(f"expected name=path, got {spec!r}")
            rep = Path(tmp) / f"{name}_report.json"
            print(f">> matching {name} ({path})", flush=True)
            run_match(root, Path(path), rep)
            results[name] = profile(rep)

    names = list(results)
    lines = ["# Detector config comparison", "",
             "GT-free, map-referenced. `coverage` = matched / front-facing "
             "detectable map candidates (detector recall proxy). "
             "`unmatched` = detections with no map match (FP-ish).", ""]
    # summary row
    lines += ["## Summary", "",
              "| config | detections | map-match | unmatched | " +
              "overall coverage |",
              "|---|---:|---:|---:|---:|"]
    for n in names:
        r = results[n]
        cov_all = [b for b in r["profile"].values() if b["candidate_coverage"] is not None]
        # overall coverage weighted by detectable candidates
        num = sum(b["detectable_candidates"] * b["candidate_coverage"] for b in cov_all)
        den = sum(b["detectable_candidates"] for b in cov_all)
        lines.append(f"| {n} | {r['detections']} | {r['map_match_rate']} | "
                     f"{r['unmatched_detections']} | {round(num/max(den,1),3)} |")
    # coverage by distance
    lines += ["", "## candidate_coverage by distance", "",
              "| bin | " + " | ".join(names) + " |",
              "|---|" + "|".join(["---:"] * len(names)) + "|"]
    for b in DIST_BINS:
        row = [b]
        for n in names:
            pb = results[n]["profile"].get(b)
            row.append(str(pb["candidate_coverage"]) if pb else "-")
        lines.append("| " + " | ".join(row) + " |")
    # unknown rate by distance
    lines += ["", "## unknown_rate by distance (matched detections)", "",
              "| bin | " + " | ".join(names) + " |",
              "|---|" + "|".join(["---:"] * len(names)) + "|"]
    for b in DIST_BINS:
        row = [b]
        for n in names:
            pb = results[n]["profile"].get(b)
            row.append(str(pb["unknown_rate"]) if pb else "-")
        lines.append("| " + " | ".join(row) + " |")

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
