#!/usr/bin/env python3
"""Render a top-down map view of signal classification over the ego path.

The frame view answers "what is in this image". This one answers "where was
the ego, which mapped signals were around it, and how was each one classified
at that moment" -- the view needed to reason about *why* map matching
succeeded or failed, since the failure modes (geometry_mismatch, beyond_gate)
are geometric.

Read-only: it stages nothing and writes nothing back.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from tlr_autolabel.map.geo import MgrsGridError, local_to_latlon
from tlr_autolabel.map.lanelet2 import (
    load_lanelet2_context,
    load_lanelet2_traffic_lights,
)
from tlr_autolabel.review.re_apply import load_reviewed_sidecar
from tlr_autolabel.review.re_review_timeline import (
    DEFAULT_CROP_CHANNELS,
    companion_links,
    parse_float,
    resolve_crop_channels,
)


def quaternion_yaw(rotation) -> float:
    """Heading in the map plane, from a T4 [w, x, y, z] quaternion."""
    if not rotation or len(rotation) < 4:
        return 0.0
    w, x, y, z = (float(v) for v in rotation[:4])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def load_ego_by_sample_data(dataset_root: Path) -> dict[str, dict]:
    """sample_data_token -> ego pose, the link annotations already carry."""
    ego_by_token = {
        row["token"]: row
        for row in json.loads((dataset_root / "annotation/ego_pose.json").read_text())
    }
    out: dict[str, dict] = {}
    for row in json.loads((dataset_root / "annotation/sample_data.json").read_text()):
        pose = ego_by_token.get(row.get("ego_pose_token"))
        if not pose:
            continue
        translation = pose.get("translation") or [0.0, 0.0, 0.0]
        out[row["token"]] = {
            "x": float(translation[0]),
            "y": float(translation[1]),
            "yaw": quaternion_yaw(pose.get("rotation")),
            "t": pose.get("timestamp"),
        }
    return out


def light_positions(traffic_lights: dict, regulatory_by_way: dict) -> dict[str, dict]:
    """Flatten the lanelet2 traffic lights into plottable 2D records."""
    out: dict[str, dict] = {}
    for way_id, light in traffic_lights.items():
        corners = light["corners"]
        cx = float(corners[:, 0].mean())
        cy = float(corners[:, 1].mean())
        axis = light.get("facing_axis")
        out[way_id] = {
            "way": way_id,
            "x": round(cx, 2),
            "y": round(cy, 2),
            "z": round(float(corners[:, 2].mean()), 2),
            "subtype": light.get("subtype", "") or "",
            "re": sorted(regulatory_by_way.get(way_id, [])),
            "fx": round(float(axis[0]), 3) if axis is not None else None,
            "fy": round(float(axis[1]), 3) if axis is not None else None,
        }
    return out


def build_steps(
    annotations: list[dict],
    ego_by_sample_data: dict[str, dict],
    channels: set[str] | None,
) -> list[dict]:
    """One entry per captured frame: where the ego was, and what each box in
    that frame was classified as."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        channel = ann.get("channel") or ""
        if channels is not None and channel not in channels:
            continue
        sd_token = ann.get("sample_data_token")
        if sd_token and sd_token in ego_by_sample_data:
            grouped[sd_token].append(ann)

    steps = []
    for sd_token, anns in grouped.items():
        pose = ego_by_sample_data[sd_token]
        obs = []
        for ann in anns:
            attrs = ann.get("attributes") or {}
            way = attrs.get("map_traffic_light_id", "") or ""
            obs.append({
                "way": way,
                "cand": attrs.get("map_candidate_id", "") or "",
                "state": attrs.get("state", "") or "unknown",
                "kind": attrs.get("signal_kind", "") or "",
                "reason": attrs.get("unmatched_reason", "") or "",
                "score": round(parse_float(attrs.get("detector_score"), 0.0), 3),
                "matched": bool(way or attrs.get("regulatory_element_id")),
            })
        obs.sort(key=lambda o: (not o["matched"], o["way"] or o["cand"]))
        steps.append({
            "t": anns[0].get("timestamp"),
            "channel": anns[0].get("channel") or "",
            "x": round(pose["x"], 2),
            "y": round(pose["y"], 2),
            "yaw": round(pose["yaw"], 4),
            "obs": obs,
        })
    steps.sort(key=lambda s: (s["channel"], s["t"] or 0))
    for index, step in enumerate(steps):
        step["i"] = index
    return steps


def relevant_lights(lights: dict[str, dict], steps: list[dict], radius: float) -> dict:
    """Signals worth drawing: everything the run referenced (matched or merely
    proposed), plus anything within `radius` of the ego path for context.

    The map spans kilometres while a run covers a few hundred metres, so
    plotting every way would make the interesting area a dot.
    """
    referenced = set()
    for step in steps:
        for obs in step["obs"]:
            if obs["way"]:
                referenced.add(obs["way"])
            if obs["cand"]:
                referenced.add(obs["cand"])

    path = [(s["x"], s["y"]) for s in steps]
    keep = {}
    for way_id, light in lights.items():
        if way_id in referenced:
            keep[way_id] = {**light, "referenced": True}
            continue
        for px, py in path:
            if math.hypot(light["x"] - px, light["y"] - py) <= radius:
                keep[way_id] = {**light, "referenced": False}
                break
    return keep


def load_mgrs_grid(projector_path: Path) -> str | None:
    """The map's MGRS grid, or None when the map is not MGRS-projected."""
    if not projector_path.exists():
        return None
    info = yaml.safe_load(projector_path.read_text()) or {}
    if str(info.get("projector_type", "")).upper() != "MGRS":
        return None
    return info.get("mgrs_grid") or None


def attach_latlon(steps: list[dict], lights: dict, grid: str | None) -> str | None:
    """Add WGS84 coordinates so the page can link out to Google Maps.

    Returns the grid actually used, or None if the conversion is unavailable --
    the views stay usable without it, they just lose the external links.
    """
    if not grid:
        return None
    try:
        for step in steps:
            lat, lon = local_to_latlon(step["x"], step["y"], grid)
            step["lat"], step["lon"] = round(lat, 7), round(lon, 7)
        for light in lights.values():
            lat, lon = local_to_latlon(light["x"], light["y"], grid)
            light["lat"], light["lon"] = round(lat, 7), round(lon, 7)
    except (MgrsGridError, ValueError) as exc:
        print(f"skipping geo links: {exc}", flush=True)
        return None
    return grid


def view_bounds(steps: list[dict], lights: dict, margin: float) -> tuple:
    """Axis-aligned box covering the ego path and every drawn signal."""
    xs = [s["x"] for s in steps] + [l["x"] for l in lights.values()]
    ys = [s["y"] for s in steps] + [l["y"] for l in lights.values()]
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def clip_polyline(points, bounds) -> bool:
    """True when any vertex falls inside `bounds`.

    Vertex-level rather than exact clipping: road geometry here is dense
    enough that a segment crossing the box always has a vertex near it, and
    the alternative is embedding the whole city.
    """
    minx, miny, maxx, maxy = bounds
    return any(minx <= x <= maxx and miny <= y <= maxy for x, y in points)


def round_points(points, ndigits: int = 2) -> list:
    return [[round(x, ndigits), round(y, ndigits)] for x, y in points]


def local_road_context(lanelets, ways, bounds) -> tuple[list, list]:
    """Only the road geometry near the run: the full map is kilometres wide
    and would dominate the payload."""
    kept_lanelets = [
        {
            "id": lane["id"],
            "subtype": lane["subtype"],
            "left": round_points(lane["left"]),
            "right": round_points(lane["right"]),
        }
        for lane in lanelets
        if clip_polyline(lane["left"], bounds) or clip_polyline(lane["right"], bounds)
    ]
    kept_ways = [
        {"id": way["id"], "type": way["type"], "points": round_points(way["points"])}
        for way in ways
        if clip_polyline(way["points"], bounds)
    ]
    return kept_lanelets, kept_ways


def summarize(steps: list[dict], lights: dict) -> dict:
    per_way = defaultdict(Counter)
    per_cand = defaultdict(Counter)
    reasons = Counter()
    n_obs = 0
    n_unmatched = 0
    for step in steps:
        for obs in step["obs"]:
            n_obs += 1
            if obs["matched"]:
                per_way[obs["way"]][obs["state"]] += 1
            else:
                n_unmatched += 1
                reasons[obs["reason"] or "(none)"] += 1
                if obs["cand"]:
                    per_cand[obs["cand"]][obs["state"]] += 1
    return {
        "n_steps": len(steps),
        "n_observations": n_obs,
        "n_unmatched": n_unmatched,
        "n_lights_drawn": len(lights),
        "matched_ways": {k: dict(v) for k, v in sorted(per_way.items())},
        "candidate_ways": {k: dict(v) for k, v in sorted(per_cand.items())},
        "unmatched_reasons": dict(reasons.most_common()),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument(
        "--sidecar",
        default=Path("annotation/traffic_signal_2d_ann.json"),
        type=Path,
    )
    parser.add_argument(
        "--map",
        default=Path("map/lanelet2_map.osm"),
        type=Path,
        help="lanelet2 map providing traffic-light geometry",
    )
    parser.add_argument(
        "--map-projector-info",
        default=Path("map/map_projector_info.yaml"),
        type=Path,
        help="map projector info, used to turn map coordinates into lat/lon",
    )
    parser.add_argument(
        "--context-radius",
        default=120.0,
        type=float,
        help="draw unreferenced map signals within this distance of the ego path",
    )
    parser.add_argument(
        "--review",
        default=None,
        type=Path,
        help="apply this traffic_signal_re_review/v1 file before rendering, to "
             "confirm what the corrections produced",
    )
    parser.add_argument("--crop-channels", default=DEFAULT_CROP_CHANNELS)
    parser.add_argument(
        "--output",
        default=Path("build/tl_match/re_map_view.html"),
        type=Path,
    )
    parser.add_argument(
        "--frame-view",
        default=Path("build/tl_match/re_frame_view.html"),
        type=Path,
        help="companion frame view to link to (generated by re_frame_view.py)",
    )
    parser.add_argument(
        "--timeline",
        default=Path("build/tl_match/re_review_timeline.html"),
        type=Path,
        help="companion timeline review to link to",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


def run(args: argparse.Namespace) -> None:
    root = args.dataset_root.resolve()
    sidecar_path = args.sidecar if args.sidecar.is_absolute() else root / args.sidecar
    map_path = args.map if args.map.is_absolute() else root / args.map
    output_path = args.output if args.output.is_absolute() else root / args.output

    for path, what in ((sidecar_path, "sidecar"), (map_path, "lanelet2 map")):
        if not path.exists():
            raise SystemExit(f"{what} not found: {path}")

    review_path = None
    if args.review is not None:
        review_path = args.review if args.review.is_absolute() else root / args.review
    sidecar, apply_summary = load_reviewed_sidecar(
        sidecar_path, root, review_path, required=args.review is not None
    )
    annotations = sidecar.get("annotations", [])
    channels = resolve_crop_channels(
        args.crop_channels,
        {a.get("channel") for a in annotations if a.get("channel")},
    )
    ego_by_sample_data = load_ego_by_sample_data(root)
    steps = build_steps(annotations, ego_by_sample_data, channels)
    if not steps:
        raise SystemExit(
            "no frames with both annotations and an ego pose.\n"
            f"  annotations: {len(annotations)}  "
            f"sample_data with ego_pose: {len(ego_by_sample_data)}"
        )

    traffic_lights, regulatory_by_way = load_lanelet2_traffic_lights(map_path)
    lights = relevant_lights(
        light_positions(traffic_lights, regulatory_by_way), steps, args.context_radius
    )
    lanelets, context_ways = load_lanelet2_context(map_path)
    bounds = view_bounds(steps, lights, args.context_radius)
    lanelets, context_ways = local_road_context(lanelets, context_ways, bounds)
    projector_path = (
        args.map_projector_info if args.map_projector_info.is_absolute()
        else root / args.map_projector_info
    )
    grid = attach_latlon(steps, lights, load_mgrs_grid(projector_path))
    stats = summarize(steps, lights)

    subtitle = (
        f"{stats['n_steps']} frames &middot; "
        f"{stats['n_observations']} boxes &middot; "
        f"{stats['n_unmatched']} unmatched &middot; "
        f"{len(stats['matched_ways'])} ways matched / "
        f"{stats['n_lights_drawn']} drawn of {len(traffic_lights)} in map &middot; "
        f"{len(lanelets)} lanelets"
        + (
            f" &middot; <b>reviewed</b> ({apply_summary['applied_annotations']} "
            f"state, {apply_summary['applied_visibility_annotations']} visibility, "
            f"{apply_summary['applied_roi_annotations']} ROI applied)"
            if apply_summary else ""
        )
    )

    links = companion_links(
        root, output_path.parent, frame_view=args.frame_view, timeline=args.timeline
    )
    page = (
        PAGE_TEMPLATE.replace("__SUBTITLE__", subtitle)
        .replace("__LINKS__", json.dumps(links, ensure_ascii=False))
        .replace("__GEO__", json.dumps({"grid": grid}, ensure_ascii=False))
        .replace("__STEPS__", json.dumps(steps, ensure_ascii=False))
        .replace("__LIGHTS__", json.dumps(list(lights.values()), ensure_ascii=False))
        .replace("__LANELETS__", json.dumps(lanelets, ensure_ascii=False))
        .replace("__WAYS__", json.dumps(context_ways, ensure_ascii=False))
        .replace("__STATS__", json.dumps(stats, ensure_ascii=False))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    print(
        f"wrote {output_path} ({stats['n_steps']} frames, "
        f"{stats['n_lights_drawn']} signals drawn, "
        f"{len(stats['matched_ways'])} ways matched, "
        f"{stats['n_unmatched']} unmatched boxes, "
        f"{len(lanelets)} lanelets + {len(context_ways)} context ways)"
    )


PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TLR map view</title>
<style>
  :root { --bg:#f8f8f6; --panel:#ffffff; --ink:#202428; --muted:#626a73;
    --line:#d8dde2; --red:#c7352b; --amber:#d49716; --green:#27824a;
    --unknown:#9aa3ad; --accent:#2457c5; --bad:#9f2d20; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:13px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
  header { position:sticky; top:0; z-index:5; padding:10px 16px;
    background:var(--panel); border-bottom:1px solid var(--line); }
  h1 { margin:0 0 2px; font-size:16px; font-weight:650; }
  .sub { color:var(--muted); font-size:12px; }
  .toolbar { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:9px; }
  .toolbar .grow { flex:1; min-width:200px; }
  button, select { font:inherit; border:1px solid var(--line); border-radius:4px;
    background:#fff; color:var(--ink); padding:5px 9px; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  input[type=range] { width:100%; }
  label.chk { display:inline-flex; align-items:center; gap:5px; color:var(--muted);
    font-size:12px; cursor:pointer; }
  .viewlink { color:var(--accent); text-decoration:none; font-size:12px;
    border:1px solid var(--line); border-radius:4px; padding:5px 9px; }
  .viewlink:hover { border-color:var(--accent); }
  .viewlink.geo { color:#1a7f4b; }
  .geolinks { margin-top:4px; display:flex; gap:6px; }
  .geolinks a { color:#1a7f4b; text-decoration:none; font-size:11px;
    border:1px solid var(--line); border-radius:3px; padding:1px 5px; }
  .geolinks a:hover { border-color:#1a7f4b; }
  main { display:grid; grid-template-columns:minmax(0,1fr) 380px; }
  #mapWrap { padding:14px 16px; }
  canvas { display:block; width:100%; background:#fff; border:1px solid var(--line);
    border-radius:5px; touch-action:none; cursor:grab; }
  canvas.dragging { cursor:grabbing; }
  aside { border-left:1px solid var(--line); background:var(--panel); padding:14px;
    height:calc(100vh - 96px); overflow:auto; position:sticky; top:96px; }
  h2 { margin:0 0 6px; font-size:13px; text-transform:uppercase;
    letter-spacing:.04em; color:var(--muted); }
  .obs { border:1px solid var(--line); border-left-width:4px; border-radius:4px;
    padding:6px 8px; margin-bottom:6px; font-size:12px; }
  .obs.unmatched { border-left-color:var(--bad); background:#fdf7f6; }
  .obs .st { font-weight:700; padding:1px 5px; border-radius:3px; color:#fff; }
  .obs .m { color:var(--muted); font-size:11px; display:block; margin-top:3px; }
  .frameInfo { font-size:12px; color:var(--muted); margin-bottom:8px; }
  .frameInfo b { color:var(--ink); }
  .legend { margin-top:10px; color:var(--muted); font-size:11px; line-height:1.8; }
  .legend span { margin-right:12px; white-space:nowrap; }
  .swatch { display:inline-block; width:10px; height:10px; border-radius:2px;
    vertical-align:-1px; margin-right:3px; }
</style>
</head><body>
<header>
  <h1>Traffic signal map view</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="toolbar">
    <select id="channelSel"></select>
    <button id="prevBtn" type="button">&larr;</button>
    <button id="nextBtn" type="button">&rarr;</button>
    <button id="playBtn" type="button">Play</button>
    <span class="grow"><input id="slider" type="range" min="0" value="0"></span>
    <label class="chk"><input type="checkbox" id="showRoad" checked> road</label>
    <label class="chk"><input type="checkbox" id="showRays" checked> rays to observed</label>
    <label class="chk"><input type="checkbox" id="showFacing" checked> facing</label>
    <button id="resetView" type="button">Reset view</button>
    <a id="egoMaps" class="viewlink geo" href="#" target="_blank" rel="noopener"
       style="display:none">Ego on Google Maps &#8599;</a>
    <a id="frameLink" class="viewlink" href="#">Frame view &rarr;</a>
    <a id="timelineLink" class="viewlink" href="#">Timeline review &rarr;</a>
  </div>
</header>
<main>
  <div id="mapWrap">
    <div class="frameInfo" id="frameInfo"></div>
    <canvas id="map" height="720"></canvas>
    <div class="legend">
      <span><i class="swatch" style="background:var(--red)"></i>red</span>
      <span><i class="swatch" style="background:var(--amber)"></i>amber</span>
      <span><i class="swatch" style="background:var(--green)"></i>green</span>
      <span><i class="swatch" style="background:var(--unknown)"></i>unknown / not seen now</span>
      <span>&#9679; matched way &nbsp; &#9633; candidate-only &nbsp; &#183; map context</span>
      <br>
      <span><i class="swatch" style="background:#eceff2;border:1px solid #dadfe4"></i>road</span>
      <span><i class="swatch" style="background:#e4ecf6;border:1px solid #b9cbe4"></i>crosswalk</span>
      <span><i class="swatch" style="background:#eef3ec;border:1px solid #cfdcca"></i>pedestrian lane</span>
      <span>dashed purple = intersection area</span>
      <span>grey line = stop line</span>
      <span>drag to pan, wheel to zoom, &larr;/&rarr; step, space play</span>
    </div>
  </div>
  <aside>
    <h2>At this frame</h2>
    <div id="obsList"></div>
  </aside>
</main>
<script>
const STEPS = __STEPS__;
const LIGHTS = __LIGHTS__;
const LANELETS = __LANELETS__;
const CONTEXT_WAYS = __WAYS__;
const STATS = __STATS__;
const LINKS = __LINKS__;
const GEO = __GEO__;
const LANE_STYLE = {
  road:           {fill: '#eceff2', stroke: '#dadfe4'},
  road_shoulder:  {fill: '#f2f4f6', stroke: '#e3e7ea'},
  crosswalk:      {fill: '#e4ecf6', stroke: '#b9cbe4'},
  pedestrian_lane:{fill: '#eef3ec', stroke: '#cfdcca'},
  walkway:        {fill: '#eef3ec', stroke: '#cfdcca'},
};
const lightByWay = new Map(LIGHTS.map(l => [l.way, l]));
let visible = [], pos = 0, playTimer = null;
let view = null;            // {cx, cy, scale} in map metres
const canvas = document.getElementById('map');

function stateColor(state) {
  const s = state || '';
  if (s.includes('red')) return '#c7352b';
  if (s.includes('amber') || s.includes('yellow')) return '#d49716';
  if (s.includes('green')) return '#27824a';
  return '#9aa3ad';
}

function channels() { return [...new Set(STEPS.map(s => s.channel))].sort(); }

function rebuildVisible() {
  const ch = document.getElementById('channelSel').value;
  visible = STEPS.filter(s => s.channel === ch).map(s => s.i);
  const slider = document.getElementById('slider');
  slider.max = Math.max(visible.length - 1, 0);
  pos = Math.min(pos, Math.max(visible.length - 1, 0));
  slider.value = pos;
}

function currentStep() {
  return visible.length ? STEPS[visible[Math.min(pos, visible.length - 1)]] : null;
}

// Fit the ego path plus every drawn signal, so nothing referenced sits
// off-screen at the default zoom.
function defaultView() {
  const xs = [], ys = [];
  for (const i of visible) { xs.push(STEPS[i].x); ys.push(STEPS[i].y); }
  for (const l of LIGHTS) { xs.push(l.x); ys.push(l.y); }
  if (!xs.length) return {cx: 0, cy: 0, scale: 1};
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 15;
  const w = Math.max(maxX - minX, 1) + pad * 2;
  const h = Math.max(maxY - minY, 1) + pad * 2;
  const scale = Math.min(canvas.width / w, canvas.height / h);
  return {cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, scale};
}

// Map metres -> canvas pixels. y is flipped so north is up.
function toCanvas(x, y) {
  return [
    canvas.width / 2 + (x - view.cx) * view.scale,
    canvas.height / 2 - (y - view.cy) * view.scale,
  ];
}

function fromCanvas(px, py) {
  return [
    view.cx + (px - canvas.width / 2) / view.scale,
    view.cy - (py - canvas.height / 2) / view.scale,
  ];
}

function syncCanvasSize() {
  const w = Math.max(Math.round(canvas.clientWidth), 320);
  if (canvas.width !== w) { canvas.width = w; return true; }
  return false;
}

function draw() {
  syncCanvasSize();
  const ctx = canvas.getContext('2d');
  const step = currentStep();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!view) view = defaultView();
  if (!step) return;

  // State of each way at this frame, and which candidates were proposed.
  const stateByWay = new Map();
  const candSeen = new Map();
  for (const o of step.obs) {
    if (o.matched && o.way) stateByWay.set(o.way, o.state);
    else if (o.cand) candSeen.set(o.cand, o);
  }

  if (document.getElementById('showRoad').checked) drawRoad(ctx);
  drawPath(ctx);
  for (const l of LIGHTS) drawLight(ctx, l, stateByWay, candSeen);
  if (document.getElementById('showRays').checked) {
    drawRays(ctx, step, stateByWay, candSeen);
  }
  drawEgo(ctx, step);
  drawScaleBar(ctx);
}

// Road first, as a flat background layer: it exists to make the signal
// geometry legible, so it must never compete with the markers on top.
function drawRoad(ctx) {
  for (const lane of LANELETS) {
    const style = LANE_STYLE[lane.subtype];
    if (!style) continue;
    ctx.beginPath();
    lane.left.forEach(([x, y], n) => {
      const [px, py] = toCanvas(x, y);
      n ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    for (let n = lane.right.length - 1; n >= 0; n--) {
      const [px, py] = toCanvas(lane.right[n][0], lane.right[n][1]);
      ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = style.fill;
    ctx.fill();
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  for (const way of CONTEXT_WAYS) {
    ctx.beginPath();
    way.points.forEach(([x, y], n) => {
      const [px, py] = toCanvas(x, y);
      n ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    if (way.type === 'intersection_area') {
      ctx.closePath();
      ctx.strokeStyle = '#c9b8d8';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (way.type === 'crosswalk_polygon') {
      ctx.closePath();
      ctx.strokeStyle = '#9fb4d0';
      ctx.lineWidth = 1;
      ctx.stroke();
    } else {
      ctx.strokeStyle = '#8f98a1';   // stop_line
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

function drawPath(ctx) {
  ctx.strokeStyle = '#c3c9cf';
  ctx.lineWidth = 2;
  ctx.beginPath();
  visible.forEach((idx, n) => {
    const [px, py] = toCanvas(STEPS[idx].x, STEPS[idx].y);
    n ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  });
  ctx.stroke();
}

function drawLight(ctx, light, stateByWay, candSeen) {
  const [px, py] = toCanvas(light.x, light.y);
  const seen = stateByWay.has(light.way);
  const cand = candSeen.get(light.way);
  const color = seen ? stateColor(stateByWay.get(light.way))
              : (cand ? stateColor(cand.state) : '#c3c9cf');

  if (document.getElementById('showFacing').checked && light.fx !== null) {
    // Which way the housing faces: the usual cause of geometry_mismatch.
    ctx.strokeStyle = seen || cand ? color : '#e2e6ea';
    ctx.lineWidth = seen ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + light.fx * 22, py - light.fy * 22);
    ctx.stroke();
  }

  ctx.beginPath();
  if (seen) {
    ctx.fillStyle = color;
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#202428';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  } else if (cand) {
    // Proposed by the matcher but rejected: hollow square, so a run of these
    // along the path reads as "the matcher kept pointing here and failing".
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([3, 2]);
    ctx.strokeRect(px - 6, py - 6, 12, 12);
    ctx.setLineDash([]);
  } else {
    ctx.fillStyle = light.referenced ? '#9aa3ad' : '#dfe4e8';
    ctx.arc(px, py, light.referenced ? 3.5 : 2, 0, Math.PI * 2);
    ctx.fill();
  }

  if (seen || cand) {
    ctx.fillStyle = '#202428';
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(light.way, px + 10, py - 8);
  }
}

function drawRays(ctx, step, stateByWay, candSeen) {
  const [ex, ey] = toCanvas(step.x, step.y);
  for (const [way, state] of stateByWay) {
    const l = lightByWay.get(way);
    if (!l) continue;
    const [px, py] = toCanvas(l.x, l.y);
    ctx.strokeStyle = stateColor(state);
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(px, py); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  for (const [way, obs] of candSeen) {
    const l = lightByWay.get(way);
    if (!l) continue;
    const [px, py] = toCanvas(l.x, l.y);
    ctx.strokeStyle = '#9f2d20';
    ctx.globalAlpha = 0.4;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(px, py); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }
}

function drawEgo(ctx, step) {
  const [px, py] = toCanvas(step.x, step.y);
  const len = 18;
  ctx.strokeStyle = '#2457c5';
  ctx.fillStyle = '#2457c5';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px + Math.cos(step.yaw) * len, py - Math.sin(step.yaw) * len);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(px, py, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawScaleBar(ctx) {
  const target = 80;
  let metres = Math.pow(10, Math.floor(Math.log10(target / view.scale)));
  while (metres * view.scale < 40) metres *= 2;
  const px = metres * view.scale;
  const x = 14, y = canvas.height - 16;
  ctx.strokeStyle = '#626a73';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y); ctx.lineTo(x + px, y);
  ctx.moveTo(x, y - 4); ctx.lineTo(x, y + 4);
  ctx.moveTo(x + px, y - 4); ctx.lineTo(x + px, y + 4);
  ctx.stroke();
  ctx.fillStyle = '#626a73';
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillText(metres + ' m', x + px + 6, y + 4);
}

// Companion views follow the frame on screen, so switching view keeps the
// moment rather than restarting at frame 0.
function mapsUrl(lat, lon, zoom) {
  return `https://www.google.com/maps/@${lat},${lon},${zoom || 19}z`;
}

function panoUrl(lat, lon) {
  // Street View answers "is there physically a signal here", which the map
  // alone cannot -- the reason a signal may be missing from lanelet2.
  return `https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${lat},${lon}`;
}

function geoLinks(lat, lon, label) {
  if (lat === undefined || lat === null) return '';
  return `<span class="geolinks"><a href="${mapsUrl(lat, lon)}" target="_blank"`
    + ` rel="noopener" title="${escapeHtml(label)} on Google Maps">map &#8599;</a>`
    + `<a href="${panoUrl(lat, lon)}" target="_blank" rel="noopener"`
    + ` title="${escapeHtml(label)} in Street View">street view &#8599;</a></span>`;
}

function syncViewLinks() {
  const step = currentStep();
  const q = step ? `#ch=${encodeURIComponent(step.channel)}&t=${step.t}` : '';
  document.getElementById('frameLink').href = LINKS.frame_view + q;
  document.getElementById('timelineLink').href = LINKS.timeline + q;
  const ego = document.getElementById('egoMaps');
  if (step && step.lat !== undefined) {
    ego.style.display = '';
    ego.href = mapsUrl(step.lat, step.lon, 18);
  } else {
    ego.style.display = 'none';
  }
}

function renderPanel() {
  syncViewLinks();
  const host = document.getElementById('obsList');
  const info = document.getElementById('frameInfo');
  const step = currentStep();
  if (!step) { host.innerHTML = ''; info.textContent = ''; return; }
  const nUnmatched = step.obs.filter(o => !o.matched).length;
  info.innerHTML = `<b>${pos + 1} / ${visible.length}</b> &middot; ${step.channel}`
    + ` &middot; ego (${step.x.toFixed(1)}, ${step.y.toFixed(1)})`
    + ` yaw ${(step.yaw * 180 / Math.PI).toFixed(1)}&deg;`
    + ` &middot; ${step.obs.length} box(es)`
    + (nUnmatched ? ` &middot; <b>${nUnmatched} unmatched</b>` : '')
    + (step.lat !== undefined
        ? ` &middot; ${step.lat.toFixed(6)}, ${step.lon.toFixed(6)}`
          + (GEO.grid ? ` (${escapeHtml(GEO.grid)})` : '')
        : '');

  host.innerHTML = step.obs.map(o => {
    const light = lightByWay.get(o.way || o.cand);
    const re = light && light.re.length ? light.re.join(', ') : '-';
    const dist = light
      ? Math.hypot(light.x - step.x, light.y - step.y).toFixed(1) + ' m'
      : 'n/a';
    const head = o.matched
      ? `way ${escapeHtml(o.way)}`
      : `candidate ${escapeHtml(o.cand || '-')}`;
    const why = o.matched ? '' : `<br>rejected: ${escapeHtml(o.reason || 'n/a')}`;
    const geo = light ? geoLinks(light.lat, light.lon, `way ${o.way || o.cand}`) : '';
    return `<div class="obs ${o.matched ? '' : 'unmatched'}">`
      + `<span class="st" style="background:${stateColor(o.state)}">`
      + `${escapeHtml(o.state)}</span> ${head}`
      + `<span class="m">RE ${escapeHtml(re)} &middot; ${dist} &middot; `
      + `kind=${escapeHtml(o.kind || '-')} score=${o.score}${why}</span>${geo}</div>`;
  }).join('') || '<div class="sub">no boxes in this frame</div>';
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text == null ? '' : String(text);
  return d.innerHTML;
}

function show(newPos) {
  if (!visible.length) return;
  pos = (newPos + visible.length) % visible.length;
  document.getElementById('slider').value = pos;
  draw();
  renderPanel();
}

function setPlaying(on) {
  const btn = document.getElementById('playBtn');
  if (on && !playTimer) {
    playTimer = setInterval(() => show(pos + 1), 300);
    btn.textContent = 'Pause'; btn.classList.add('primary');
  } else if (!on && playTimer) {
    clearInterval(playTimer); playTimer = null;
    btn.textContent = 'Play'; btn.classList.remove('primary');
  }
}

const channelSel = document.getElementById('channelSel');
for (const ch of channels()) {
  const opt = document.createElement('option');
  opt.value = ch; opt.textContent = ch;
  channelSel.appendChild(opt);
}
channelSel.addEventListener('change', () => { rebuildVisible(); view = null; show(0); });
document.getElementById('prevBtn').addEventListener('click', () => show(pos - 1));
document.getElementById('nextBtn').addEventListener('click', () => show(pos + 1));
document.getElementById('playBtn').addEventListener('click', () => setPlaying(!playTimer));
document.getElementById('slider').addEventListener('input', ev => {
  setPlaying(false); show(parseInt(ev.target.value, 10));
});
document.getElementById('showRoad').addEventListener('change', draw);
document.getElementById('showRays').addEventListener('change', draw);
document.getElementById('showFacing').addEventListener('change', draw);
document.getElementById('resetView').addEventListener('click', () => {
  view = defaultView(); draw();
});
document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT') return;
  if (ev.key === 'ArrowLeft') { setPlaying(false); show(pos - 1); }
  else if (ev.key === 'ArrowRight') { setPlaying(false); show(pos + 1); }
  else if (ev.key === ' ') { ev.preventDefault(); setPlaying(!playTimer); }
});

canvas.addEventListener('wheel', ev => {
  ev.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const px = (ev.clientX - rect.left) * (canvas.width / rect.width);
  const py = (ev.clientY - rect.top) * (canvas.height / rect.height);
  const [mx, my] = fromCanvas(px, py);
  view.scale *= Math.exp(-ev.deltaY * 0.001);
  // Keep the point under the cursor fixed while zooming.
  const [nx, ny] = fromCanvas(px, py);
  view.cx += mx - nx; view.cy += my - ny;
  draw();
}, {passive: false});

let drag = null;
canvas.addEventListener('pointerdown', ev => {
  canvas.setPointerCapture(ev.pointerId);
  canvas.classList.add('dragging');
  drag = {px: ev.clientX, py: ev.clientY, cx: view.cx, cy: view.cy};
});
canvas.addEventListener('pointermove', ev => {
  if (!drag) return;
  const rect = canvas.getBoundingClientRect();
  const k = canvas.width / rect.width;
  view.cx = drag.cx - (ev.clientX - drag.px) * k / view.scale;
  view.cy = drag.cy + (ev.clientY - drag.py) * k / view.scale;
  draw();
});
const endDrag = ev => {
  if (!drag) return;
  canvas.releasePointerCapture(ev.pointerId);
  canvas.classList.remove('dragging');
  drag = null;
};
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);
window.addEventListener('resize', () => { draw(); });

// Deep link from the timeline review / frame view: #ch=&t= selects the frame
// captured at that timestamp, so the three views stay on the same moment.
function applyHash() {
  const q = new URLSearchParams(location.hash.replace(/^#/, ''));
  const ch = q.get('ch'), t = q.get('t');
  let target = null;
  if (t) target = STEPS.find(s => String(s.t) === t && (!ch || s.channel === ch));
  else if (ch) target = STEPS.find(s => s.channel === ch);
  if (!target) return false;
  document.getElementById('channelSel').value = target.channel;
  rebuildVisible();
  const at = visible.indexOf(target.i);
  if (at < 0) return false;
  pos = at;
  document.getElementById('slider').value = pos;
  view = null;
  show(pos);
  return true;
}

window.addEventListener('hashchange', () => { setPlaying(false); applyHash(); });

rebuildVisible();
if (!applyHash()) show(0);
</script></body></html>
"""


if __name__ == "__main__":
    main()
