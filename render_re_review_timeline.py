#!/usr/bin/env python3
"""Render an editable RE timeline review HTML.

The page is static: it does not need a backend and does not write files by
itself. Reviewers click state segments, mark them accepted/fixed/rejected, and
export a traffic_signal_re_review/v1 JSON file for apply_re_review.py.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CROP_CHANNELS = "CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT,CAM_FRONT_FAR"


def id_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def sorted_ids(values) -> list[str]:
    return sorted((str(v) for v in values), key=id_sort_key)


def signal_group_id(member_ways: list[str]) -> str:
    return "ways:" + ",".join(sorted_ids(member_ways))


def split_ids(value: str) -> set[str]:
    return {v.strip() for v in (value or "").split(",") if v.strip()}


def parse_float(value, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def rel_href(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path, base_dir)).as_posix()


def parse_channels(value: str) -> set[str] | None:
    if value.strip().lower() in {"all", "*"}:
        return None
    return {v.strip() for v in value.split(",") if v.strip()}


def segment_observations(observations: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for obs in sorted(observations, key=lambda o: o.get("timestamp") or 0):
        state = obs.get("state", "unknown") or "unknown"
        if segments and segments[-1]["state"] == state:
            cur = segments[-1]
            cur["end_sample_token"] = obs["sample_token"]
            cur["end_timestamp"] = obs.get("timestamp")
            cur["sample_tokens"].append(obs["sample_token"])
            cur["n_frames"] += 1
            cur["flags"] = sorted(set(cur["flags"] + obs.get("flags", [])))
            cur["confidence_sum"] += float(obs.get("confidence") or 0.0)
            continue
        segments.append(
            {
                "start_sample_token": obs["sample_token"],
                "end_sample_token": obs["sample_token"],
                "sample_tokens": [obs["sample_token"]],
                "start_timestamp": obs.get("timestamp"),
                "end_timestamp": obs.get("timestamp"),
                "state": state,
                "n_frames": 1,
                "flags": sorted(obs.get("flags", [])),
                "confidence_sum": float(obs.get("confidence") or 0.0),
                "head_states": obs.get("head_states", {}),
            }
        )
    for segment in segments:
        segment["confidence"] = round(
            segment.pop("confidence_sum") / max(segment["n_frames"], 1), 3
        )
    return segments


def build_rows(series: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for item in series:
        grouped[tuple(sorted_ids(item["member_ways"]))].append(item)

    rows = []
    for member_ways, members in sorted(grouped.items(), key=lambda kv: kv[0]):
        representative = max(members, key=lambda s: s.get("n_observations", 0))
        rel_ids = sorted_ids(s["regulatory_element_id"] for s in members)
        segments = segment_observations(representative.get("observations", []))
        rows.append(
            {
                "signal_group_id": signal_group_id(list(member_ways)),
                "member_ways": list(member_ways),
                "regulatory_element_ids": rel_ids,
                "label": "ways " + ",".join(member_ways),
                "sublabel": "RE " + ",".join(rel_ids),
                "segments": segments,
            }
        )
    return rows


def annotation_matches_group(ann: dict, row: dict) -> bool:
    attrs = ann.get("attributes") or {}
    way_id = attrs.get("map_traffic_light_id", "")
    if way_id and way_id in set(row["member_ways"]):
        return True
    ann_re_ids = split_ids(attrs.get("regulatory_element_id", ""))
    return bool(ann_re_ids & set(row["regulatory_element_ids"]))


def annotation_rank(ann: dict, segment: dict) -> float:
    attrs = ann.get("attributes") or {}
    box = ann.get("box2d") or ann.get("bbox") or [0, 0, 0, 0]
    w = max(float(box[2]) - float(box[0]), 0.0)
    h = max(float(box[3]) - float(box[1]), 0.0)
    min_side_score = min(min(w, h) / 80.0, 1.0) if w and h else 0.0
    detector_score = parse_float(attrs.get("detector_score"), 0.0)
    visibility = attrs.get("visibility", "")
    visibility_score = {
        "clear": 1.0,
        "full": 1.0,
        "partial_occluded": 0.55,
        "heavy_occluded": 0.2,
    }.get(visibility, 0.45)
    source_weight = {
        "auto": 1.0,
        "manual": 0.95,
        "cvat": 0.95,
        "projected_map": 0.75,
        "interpolated": 0.6,
        "map_presence": 0.45,
    }.get(attrs.get("source_type", ""), 0.75)
    sample_tokens = segment.get("sample_tokens") or []
    if ann.get("sample_token") in sample_tokens:
        if len(sample_tokens) == 1:
            centrality = 1.0
        else:
            index = sample_tokens.index(ann["sample_token"])
            center_index = (len(sample_tokens) - 1) / 2
            centrality = 1.0 - min(abs(index - center_index) / max(center_index, 1.0), 1.0)
    else:
        start_ts = int(segment["start_timestamp"])
        end_ts = int(segment["end_timestamp"])
        duration = max(end_ts - start_ts, 1)
        center = (start_ts + end_ts) / 2
        centrality = 1.0 - min(abs(int(ann.get("timestamp") or center) - center) / duration, 1.0)
    return round(
        source_weight
        * (
            0.45 * min_side_score
            + 0.35 * detector_score
            + 0.10 * visibility_score
            + 0.10 * centrality
        ),
        4,
    )


def select_crop_candidates(annotations: list[dict], segment: dict, limit: int) -> list[dict]:
    sample_tokens = set(segment.get("sample_tokens") or [])
    def in_segment(ann: dict) -> bool:
        if sample_tokens:
            return ann.get("sample_token") in sample_tokens
        return segment["start_timestamp"] <= int(ann.get("timestamp") or 0) <= segment["end_timestamp"]

    ranked = sorted(
        (
            (annotation_rank(ann, segment), ann)
            for ann in annotations
            if in_segment(ann)
        ),
        key=lambda item: (-item[0], item[1].get("channel", ""), item[1].get("timestamp", 0)),
    )
    selected: list[tuple[float, dict]] = []
    used_channels: set[str] = set()
    seen_tokens: set[str] = set()
    for score, ann in ranked:
        token = ann.get("token", "")
        if token in seen_tokens:
            continue
        channel = ann.get("channel", "")
        if channel in used_channels:
            continue
        selected.append((score, ann))
        seen_tokens.add(token)
        used_channels.add(channel)
        if len(selected) >= limit:
            return selected
    for score, ann in ranked:
        token = ann.get("token", "")
        if token in seen_tokens:
            continue
        selected.append((score, ann))
        seen_tokens.add(token)
        if len(selected) >= limit:
            break
    return selected


def write_crop(root: Path, ann: dict, crop_path: Path, margin_ratio: float) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    image_path = root / ann["filename"]
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in ann["box2d"]]
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    margin = max(bw, bh) * margin_ratio
    cx0 = max(int(x0 - margin), 0)
    cy0 = max(int(y0 - margin), 0)
    cx1 = min(int(x1 + margin), width)
    cy1 = min(int(y1 + margin), height)
    if cx1 <= cx0 or cy1 <= cy0:
        return False
    crop = image[cy0:cy1, cx0:cx1].copy()
    scale = min(max(220.0 / max(min(crop.shape[:2]), 1), 1.0), 5.0)
    if scale > 1.01:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    rx0 = int(round((x0 - cx0) * scale))
    ry0 = int(round((y0 - cy0) * scale))
    rx1 = int(round((x1 - cx0) * scale))
    ry1 = int(round((y1 - cy0) * scale))
    thickness = max(2, int(round(scale * 1.5)))
    cv2.rectangle(crop, (rx0, ry0), (rx1, ry1), (0, 255, 255), thickness)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(crop_path), crop))


def attach_crop_candidates(
    root: Path,
    rows: list[dict],
    sidecar_path: Path,
    assets_dir: Path,
    output_dir: Path,
    limit: int,
    margin_ratio: float,
    crop_channels: set[str] | None,
) -> int:
    if limit <= 0 or not sidecar_path.exists():
        return 0
    sidecar = json.loads(sidecar_path.read_text())
    annotations = sidecar.get("annotations", [])
    total = 0
    for row in rows:
        group_annotations = [
            ann for ann in annotations
            if annotation_matches_group(ann, row) and ann.get("box2d") and ann.get("filename")
            and (crop_channels is None or ann.get("channel") in crop_channels)
        ]
        for segment_index, segment in enumerate(row["segments"]):
            candidates = []
            for rank, ann in select_crop_candidates(group_annotations, segment, limit):
                key = hashlib.sha1(
                    f"{row['signal_group_id']}:{segment_index}:{ann.get('token')}".encode()
                ).hexdigest()[:16]
                crop_path = assets_dir / f"{key}.jpg"
                if not crop_path.exists() and not write_crop(root, ann, crop_path, margin_ratio):
                    continue
                box = [round(float(v), 1) for v in ann["box2d"]]
                attrs = ann.get("attributes") or {}
                area = round(max(box[2] - box[0], 0) * max(box[3] - box[1], 0), 1)
                candidates.append(
                    {
                        "src": rel_href(crop_path, output_dir),
                        "full_image": rel_href(root / ann["filename"], output_dir),
                        "channel": ann.get("channel", ""),
                        "filename": ann.get("filename", ""),
                        "sample_token": ann.get("sample_token", ""),
                        "timestamp": ann.get("timestamp"),
                        "state": attrs.get("state", ""),
                        "raw_state": attrs.get("raw_state", ""),
                        "source_type": attrs.get("source_type", ""),
                        "visibility": attrs.get("visibility", ""),
                        "detector_score": attrs.get("detector_score", ""),
                        "rank_score": rank,
                        "box_area": area,
                        "box2d": box,
                    }
                )
            segment["candidates"] = candidates
            total += len(candidates)
    return total


def hide_segments_without_crops(rows: list[dict]) -> tuple[list[dict], int]:
    filtered_rows = []
    hidden_segments = 0
    for row in rows:
        kept_segments = [segment for segment in row["segments"] if segment.get("candidates")]
        hidden_segments += len(row["segments"]) - len(kept_segments)
        if kept_segments:
            filtered_rows.append({**row, "segments": kept_segments})
    return filtered_rows, hidden_segments


def load_optional_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument(
        "--input",
        default=Path("annotation/traffic_signal_re_timeseries.json"),
        type=Path,
    )
    parser.add_argument(
        "--review",
        default=None,
        type=Path,
        help="optional existing traffic_signal_re_review/v1 JSON to pre-load",
    )
    parser.add_argument(
        "--sidecar",
        default=Path("annotation/traffic_signal_2d_ann.json"),
        type=Path,
        help="Tier B sidecar used to generate representative crop candidates",
    )
    parser.add_argument(
        "--assets-dir",
        default=Path("build/tl_match/re_review_assets"),
        type=Path,
        help="directory for generated crop images",
    )
    parser.add_argument(
        "--crop-candidates",
        default=6,
        type=int,
        help="maximum crop candidates embedded per timeline segment",
    )
    parser.add_argument(
        "--crop-margin",
        default=1.2,
        type=float,
        help="crop margin as a multiple of the bbox longer side",
    )
    parser.add_argument(
        "--crop-channels",
        default=DEFAULT_CROP_CHANNELS,
        help="comma-separated camera channels used for crop candidates; use 'all' for debug",
    )
    parser.add_argument(
        "--show-empty-crop-segments",
        action="store_true",
        help="show timeline segments that have no crop candidates after channel filtering",
    )
    parser.add_argument(
        "--output",
        default=Path("build/tl_match/re_review_timeline.html"),
        type=Path,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    input_path = args.input if args.input.is_absolute() else root / args.input
    review_path = None
    if args.review is not None:
        review_path = args.review if args.review.is_absolute() else root / args.review
    output_path = args.output if args.output.is_absolute() else root / args.output
    sidecar_path = args.sidecar if args.sidecar.is_absolute() else root / args.sidecar
    assets_dir = args.assets_dir if args.assets_dir.is_absolute() else root / args.assets_dir

    timeseries = json.loads(input_path.read_text())
    rows = build_rows(timeseries.get("series", []))
    n_crops = attach_crop_candidates(
        root,
        rows,
        sidecar_path,
        assets_dir,
        output_path.parent,
        args.crop_candidates,
        args.crop_margin,
        parse_channels(args.crop_channels),
    )
    hidden_segments = 0
    if not args.show_empty_crop_segments:
        rows, hidden_segments = hide_segments_without_crops(rows)

    t_values = [
        s["start_timestamp"]
        for row in rows
        for s in row["segments"]
        if s.get("start_timestamp") is not None
    ]
    t_values += [
        s["end_timestamp"]
        for row in rows
        for s in row["segments"]
        if s.get("end_timestamp") is not None
    ]
    if not t_values:
        raise SystemExit(
            "no timeline segments with crop candidates. "
            "Use --show-empty-crop-segments or --crop-channels all for debugging."
        )
    flag_counts = Counter(
        flag for row in rows for seg in row["segments"] for flag in seg.get("flags", [])
    )

    data_json = json.dumps(
        {
            "rows": rows,
            "t0": min(t_values),
            "t1": max(t_values),
            "source_timeseries": str(args.input),
            "crop_channels": args.crop_channels,
            "hidden_empty_crop_segments": hidden_segments,
        },
        ensure_ascii=False,
    )
    review_json = json.dumps(load_optional_json(review_path), ensure_ascii=False)
    flag_summary = ", ".join(f"{k}: {v}" for k, v in flag_counts.most_common())
    subtitle = (
        f"{len(rows)} signal groups &middot; "
        f"{sum(len(r['segments']) for r in rows)} segments &middot; "
        f"{n_crops} crop candidates &middot; "
        f"crop channels: {html.escape(args.crop_channels)} &middot; "
        + (f"hidden no-evidence segments: {hidden_segments} &middot; " if hidden_segments else "")
        +
        f"flags: {html.escape(flag_summary or 'none')}"
    )

    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TLR RE timeline review</title>
<style>
  :root { --bg:#f8f8f6; --panel:#ffffff; --ink:#202428; --muted:#626a73;
    --line:#d8dde2; --red:#c7352b; --amber:#d49716; --green:#27824a;
    --unknown:#9aa3ad; --accent:#2457c5; --bad:#9f2d20; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:13px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
  header { position:sticky; top:0; z-index:5; padding:12px 16px;
    background:var(--panel); border-bottom:1px solid var(--line); }
  h1 { margin:0 0 2px; font-size:17px; font-weight:650; }
  .sub { color:var(--muted); font-size:12px; }
  main { display:grid; grid-template-columns:minmax(0,1fr) 430px; gap:0; }
  #chart { overflow:auto; padding:16px; }
  .row { display:flex; align-items:center; min-width:max-content; margin-bottom:8px; }
  .rowlabel { flex:0 0 230px; padding-right:12px; font-size:12px; }
  .rowlabel b { display:block; font-weight:650; }
  .rowlabel span { color:var(--muted); font-size:11px; }
  .strip { position:relative; height:38px; border-left:1px solid var(--line); }
  .seg { position:absolute; top:7px; height:24px; border-radius:3px;
    box-sizing:border-box; border:1px solid rgba(0,0,0,.16); cursor:pointer;
    overflow:hidden; white-space:nowrap; text-overflow:clip; color:#fff;
    padding:3px 5px; font-size:11px; line-height:17px; }
  .seg:hover { outline:2px solid var(--ink); z-index:3; }
  .seg.selected { outline:3px solid var(--accent); z-index:4; }
  .seg.decided { box-shadow:inset 0 -4px 0 rgba(255,255,255,.75); }
  .seg.flagged::before { content:""; position:absolute; left:0; right:0; top:0;
    height:3px; background:#111; opacity:.8; }
  aside { height:calc(100vh - 58px); overflow:auto; position:sticky; top:58px;
    border-left:1px solid var(--line); background:var(--panel); padding:14px; }
  label { display:block; margin:10px 0 4px; color:var(--muted); font-size:12px; }
  input, select, textarea, button { font:inherit; box-sizing:border-box; }
  input, select, textarea { width:100%; border:1px solid var(--line);
    border-radius:4px; padding:7px; background:#fff; color:var(--ink); }
  textarea { min-height:72px; resize:vertical; }
  button { border:1px solid var(--line); border-radius:4px; background:#fff;
    color:var(--ink); padding:7px 9px; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.danger { color:var(--bad); }
  .buttons { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; }
  .hint, .mono { color:var(--muted); font-size:12px; }
  .mono { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; word-break:break-all; }
  .cropPanel { border:1px solid var(--line); border-radius:4px; background:#fafafa;
    min-height:120px; padding:8px; }
  .cropPanel img { display:block; width:100%; max-height:340px; object-fit:contain;
    background:#111; border-radius:3px; }
  .candidateTabs { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }
  .candidateTabs button { padding:4px 7px; font-size:12px; }
  .candidateTabs button.active { border-color:var(--accent); color:var(--accent); }
  .candidateMeta { margin-top:7px; color:var(--muted); font-size:12px; }
  #decisionList { margin-top:12px; border-top:1px solid var(--line); padding-top:10px; }
  .decision { border:1px solid var(--line); border-radius:4px; padding:7px;
    margin-bottom:7px; background:#fafafa; }
  .decision b { display:block; font-size:12px; }
</style></head><body>
<header>
  <h1>Traffic signal RE timeline review</h1>
  <div class="sub">__SUBTITLE__</div>
</header>
<main>
  <section id="chart"></section>
  <aside>
    <div class="hint">Click a segment, edit the state/status, then add it to the review JSON.</div>
    <label>Signal group</label>
    <div id="selectedGroup" class="mono">none</div>
    <label>Interval</label>
    <div id="selectedInterval" class="mono">none</div>
    <label>Image evidence</label>
    <div id="cropPanel" class="cropPanel">
      <div class="hint">Select a segment to show representative crops.</div>
    </div>
    <label for="stateInput">State</label>
    <input id="stateInput" placeholder="red-circle">
    <label for="statusInput">Review status</label>
    <select id="statusInput">
      <option value="accepted">accepted</option>
      <option value="fixed">fixed</option>
      <option value="rejected">rejected</option>
      <option value="unchecked">unchecked</option>
    </select>
    <label for="noteInput">Note</label>
    <textarea id="noteInput"></textarea>
    <div class="buttons">
      <button class="primary" id="addDecision">Add/update</button>
      <button class="danger" id="clearDecision">Clear</button>
    </div>
    <div class="buttons">
      <button id="exportReview">Export JSON</button>
      <label style="margin:0"><input id="importReview" type="file" accept="application/json" style="display:none">
        <button id="importButton" type="button">Import JSON</button></label>
    </div>
    <div id="decisionList"></div>
  </aside>
</main>
<script>
const DATA = __DATA__;
const INITIAL_REVIEW = __REVIEW__;
const PX_PER_S = 42;
const MIN_SEG_W = 5;
let selected = null;
let decisions = new Map();
let evidenceIndex = 0;

function keyFor(groupId, seg) {
  return `${groupId}|${seg.start_sample_token}|${seg.end_sample_token}`;
}

function stateStyle(state) {
  const hasRed = state.includes('red-');
  const hasAmber = state.includes('amber-');
  const hasGreen = state.includes('green-');
  const colors = [];
  if (hasRed) colors.push('var(--red)');
  if (hasAmber) colors.push('var(--amber)');
  if (hasGreen) colors.push('var(--green)');
  if (!colors.length) return 'background:var(--unknown)';
  if (colors.length === 1) return `background:${colors[0]}`;
  const stops = colors.map((c, i) => {
    const a = Math.round((i / colors.length) * 100);
    const b = Math.round(((i + 1) / colors.length) * 100);
    return `${c} ${a}% ${b}%`;
  }).join(',');
  return `background:linear-gradient(90deg,${stops})`;
}

function loadReview(payload) {
  decisions = new Map();
  for (const group of payload.groups || []) {
    const gid = group.signal_group_id;
    for (const d of group.decisions || group.segments || []) {
      const k = `${gid}|${d.start_sample_token}|${d.end_sample_token || d.start_sample_token}`;
      decisions.set(k, {...d, signal_group_id: gid});
    }
  }
}

function renderChart() {
  const chart = document.getElementById('chart');
  chart.innerHTML = '';
  const span = Math.max((DATA.t1 - DATA.t0) / 1e6, 1);
  const width = Math.ceil(span * PX_PER_S) + 80;
  for (const row of DATA.rows) {
    const rowEl = document.createElement('div');
    rowEl.className = 'row';
    rowEl.innerHTML = `<div class="rowlabel"><b>${row.label}</b><span>${row.sublabel}</span></div>`;
    const strip = document.createElement('div');
    strip.className = 'strip';
    strip.style.width = width + 'px';
    for (const seg of row.segments) {
      const left = ((seg.start_timestamp - DATA.t0) / 1e6) * PX_PER_S;
      const dur = Math.max((seg.end_timestamp - seg.start_timestamp) / 1e6, 0.1);
      const segEl = document.createElement('div');
      segEl.className = 'seg';
      if (seg.flags && seg.flags.length) segEl.classList.add('flagged');
      if (decisions.has(keyFor(row.signal_group_id, seg))) segEl.classList.add('decided');
      segEl.style.left = left + 'px';
      segEl.style.width = Math.max(dur * PX_PER_S, MIN_SEG_W) + 'px';
      segEl.setAttribute('style', segEl.getAttribute('style') + ';' + stateStyle(seg.state));
      segEl.title = `${seg.state}\\nframes=${seg.n_frames} conf=${seg.confidence}\\n${(seg.flags || []).join(', ')}`;
      if (parseFloat(segEl.style.width) > 70) segEl.textContent = seg.state;
      segEl.addEventListener('click', () => selectSegment(row, seg, segEl));
      strip.appendChild(segEl);
    }
    rowEl.appendChild(strip);
    chart.appendChild(rowEl);
  }
}

function selectSegment(row, seg, el) {
  document.querySelectorAll('.seg.selected').forEach(n => n.classList.remove('selected'));
  el.classList.add('selected');
  selected = {row, seg};
  const existing = decisions.get(keyFor(row.signal_group_id, seg));
  document.getElementById('selectedGroup').textContent = row.signal_group_id;
  document.getElementById('selectedInterval').textContent =
    `${seg.start_sample_token} .. ${seg.end_sample_token}`;
  document.getElementById('stateInput').value = existing?.state || seg.state;
  document.getElementById('statusInput').value = existing?.review_status || 'accepted';
  document.getElementById('noteInput').value = existing?.note || '';
  evidenceIndex = 0;
  renderEvidence(seg);
}

function renderEvidence(seg) {
  const panel = document.getElementById('cropPanel');
  const candidates = seg.candidates || [];
  panel.innerHTML = '';
    if (!candidates.length) {
      const empty = document.createElement('div');
      empty.className = 'hint';
      empty.textContent = `No crop candidates in channels: ${DATA.crop_channels}.`;
      panel.appendChild(empty);
      return;
    }
  evidenceIndex = Math.min(evidenceIndex, candidates.length - 1);
  const tabs = document.createElement('div');
  tabs.className = 'candidateTabs';
  candidates.forEach((cand, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = idx === evidenceIndex ? 'active' : '';
    btn.textContent = `${idx + 1} ${cand.channel || 'camera'}`;
    btn.title = `rank=${cand.rank_score} score=${cand.detector_score || '-'} area=${cand.box_area}`;
    btn.addEventListener('click', () => {
      evidenceIndex = idx;
      renderEvidence(seg);
    });
    tabs.appendChild(btn);
  });
  panel.appendChild(tabs);

  const cand = candidates[evidenceIndex];
  const link = document.createElement('a');
  link.href = cand.full_image;
  link.target = '_blank';
  const img = document.createElement('img');
  img.src = cand.src;
  img.alt = `${cand.channel} crop`;
  link.appendChild(img);
  panel.appendChild(link);

  const seconds = ((cand.timestamp - DATA.t0) / 1e6).toFixed(2);
  const meta = document.createElement('div');
  meta.className = 'candidateMeta';
  meta.textContent =
    `${cand.channel} t=${seconds}s rank=${cand.rank_score} ` +
    `det=${cand.detector_score || '-'} area=${cand.box_area} ` +
    `src=${cand.source_type || '-'} vis=${cand.visibility || '-'} ` +
    `state=${cand.state || '-'} raw=${cand.raw_state || '-'}`;
  panel.appendChild(meta);

  const path = document.createElement('div');
  path.className = 'mono';
  path.textContent = cand.filename;
  panel.appendChild(path);
}

function addDecision() {
  if (!selected) return;
  const {row, seg} = selected;
  const decision = {
    start_sample_token: seg.start_sample_token,
    end_sample_token: seg.end_sample_token,
    start_timestamp: seg.start_timestamp,
    end_timestamp: seg.end_timestamp,
    state: document.getElementById('stateInput').value.trim() || 'unknown',
    review_status: document.getElementById('statusInput').value,
    source: 'manual_timeline_review',
    n_frames: seg.n_frames,
    flags: seg.flags || [],
    note: document.getElementById('noteInput').value,
    signal_group_id: row.signal_group_id,
  };
  decisions.set(keyFor(row.signal_group_id, seg), decision);
  renderDecisionList();
  renderChart();
}

function clearDecision() {
  if (!selected) return;
  decisions.delete(keyFor(selected.row.signal_group_id, selected.seg));
  renderDecisionList();
  renderChart();
}

function buildPayload() {
  const byGroup = new Map();
  for (const row of DATA.rows) {
    byGroup.set(row.signal_group_id, {
      signal_group_id: row.signal_group_id,
      member_ways: row.member_ways,
      regulatory_element_ids: row.regulatory_element_ids,
      decisions: [],
    });
  }
  for (const decision of decisions.values()) {
    const group = byGroup.get(decision.signal_group_id);
    if (!group) continue;
    const clean = {...decision};
    delete clean.signal_group_id;
    group.decisions.push(clean);
  }
  return {
    schema_version: 'traffic_signal_re_review/v1',
    source_timeseries: DATA.source_timeseries,
    created_at: new Date().toISOString(),
    groups: [...byGroup.values()].filter(g => g.decisions.length),
  };
}

function renderDecisionList() {
  const list = document.getElementById('decisionList');
  const payload = buildPayload();
  const n = payload.groups.reduce((acc, g) => acc + g.decisions.length, 0);
  list.innerHTML = `<div class="hint">${n} decisions staged</div>`;
  for (const group of payload.groups) {
    for (const d of group.decisions) {
      const item = document.createElement('div');
      item.className = 'decision';
      item.innerHTML = `<b>${group.signal_group_id}</b>${d.review_status}: ${d.state}<br>` +
        `<span class="mono">${d.start_sample_token} .. ${d.end_sample_token}</span>`;
      list.appendChild(item);
    }
  }
}

function exportReview() {
  const blob = new Blob([JSON.stringify(buildPayload(), null, 2) + '\\n'], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'traffic_signal_re_review.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

document.getElementById('addDecision').addEventListener('click', addDecision);
document.getElementById('clearDecision').addEventListener('click', clearDecision);
document.getElementById('exportReview').addEventListener('click', exportReview);
document.getElementById('importButton').addEventListener('click', () => {
  document.getElementById('importReview').click();
});
document.getElementById('importReview').addEventListener('change', ev => {
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    loadReview(JSON.parse(reader.result));
    renderDecisionList();
    renderChart();
  };
  reader.readAsText(file);
});

loadReview(INITIAL_REVIEW || {});
renderDecisionList();
renderChart();
</script></body></html>
"""
    page = (
        page.replace("__SUBTITLE__", subtitle)
        .replace("__DATA__", data_json)
        .replace("__REVIEW__", review_json)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    print(f"wrote {output_path} ({len(rows)} groups)")


if __name__ == "__main__":
    main()
