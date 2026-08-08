#!/usr/bin/env python3
"""Render a per-frame review view of the A' 2D sidecar.

The timeline review answers "what state did this signal group hold over
time". This view answers the complementary question: "on this image, what
was detected and how was each box classified" -- including boxes with no
map/regulatory-element match, which the timeline cannot show at all.

Read-only: it stages nothing and writes nothing back. Open the generated
HTML directly; no server is needed.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from tlr_autolabel.review.re_review_timeline import (
    DEFAULT_CROP_CHANNELS,
    companion_links,
    parse_float,
    rel_href,
    resolve_crop_channels,
)


def annotation_view(ann: dict) -> dict:
    """Flatten one sidecar annotation into what the page needs.

    Keys are short because every annotation is embedded in the HTML.
    """
    attrs = ann.get("attributes") or {}
    way = attrs.get("map_traffic_light_id", "") or ""
    re_id = attrs.get("regulatory_element_id", "") or ""
    return {
        "token": ann.get("token", ""),
        "box": [round(float(v), 1) for v in (ann.get("box2d") or [0, 0, 0, 0])],
        "state": attrs.get("state", "") or "unknown",
        "raw_state": attrs.get("raw_state", "") or "",
        "kind": attrs.get("signal_kind", "") or "",
        "vis": attrs.get("visibility", "") or "",
        "status": attrs.get("review_status", "") or "",
        "score": round(parse_float(attrs.get("detector_score"), 0.0), 3),
        "src_type": attrs.get("source_type", "") or "",
        "way": way,
        "re": re_id,
        "cand": attrs.get("map_candidate_id", "") or "",
        "reason": attrs.get("unmatched_reason", "") or "",
        # Matched == the timeline review can reach it. Everything else is
        # only reviewable here, which is the reason this view exists.
        "matched": bool(way or re_id),
    }


def build_frames(
    annotations: list[dict],
    root: Path,
    output_dir: Path,
    channels: set[str] | None,
) -> list[dict]:
    """One entry per image that carries at least one annotation, in capture
    order within each channel."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    meta: dict[tuple, dict] = {}
    for ann in annotations:
        channel = ann.get("channel") or ""
        filename = ann.get("filename") or ""
        if not filename:
            continue
        if channels is not None and channel not in channels:
            continue
        key = (channel, filename)
        grouped[key].append(ann)
        if key not in meta:
            meta[key] = {
                "channel": channel,
                "filename": filename,
                "src": rel_href(root / filename, output_dir),
                "sample_token": ann.get("sample_token", ""),
                "timestamp": ann.get("timestamp"),
            }

    frames = []
    for key, anns in grouped.items():
        entry = dict(meta[key])
        # Stable left-to-right ordering so the crop strip does not reshuffle
        # between neighbouring frames.
        entry["anns"] = [
            annotation_view(a)
            for a in sorted(anns, key=lambda a: (a.get("box2d") or [0])[0])
        ]
        frames.append(entry)

    frames.sort(key=lambda f: (f["channel"], f["timestamp"] or 0))
    for index, frame in enumerate(frames):
        frame["i"] = index
    return frames


def summarize(frames: list[dict]) -> dict:
    states = Counter()
    kinds = Counter()
    reasons = Counter()
    n_anns = 0
    n_unmatched = 0
    for frame in frames:
        for ann in frame["anns"]:
            n_anns += 1
            states[ann["state"]] += 1
            if not ann["matched"]:
                n_unmatched += 1
                kinds[ann["kind"] or "unknown"] += 1
                reasons[ann["reason"] or "(none)"] += 1
    return {
        "n_frames": len(frames),
        "n_annotations": n_anns,
        "n_unmatched": n_unmatched,
        "states": dict(states.most_common()),
        "unmatched_kinds": dict(kinds.most_common()),
        "unmatched_reasons": dict(reasons.most_common()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument(
        "--sidecar",
        default=Path("annotation/traffic_signal_2d_ann.json"),
        type=Path,
        help="A' 2D sidecar to visualise",
    )
    parser.add_argument(
        "--crop-channels",
        default=DEFAULT_CROP_CHANNELS,
        help="camera channels to include: 'auto' (default), 'all', or a list",
    )
    parser.add_argument(
        "--output",
        default=Path("build/tl_match/re_frame_view.html"),
        type=Path,
    )
    parser.add_argument(
        "--map-view",
        default=Path("build/tl_match/re_map_view.html"),
        type=Path,
        help="companion map view to link to (generated by re_map_view.py)",
    )
    parser.add_argument(
        "--timeline",
        default=Path("build/tl_match/re_review_timeline.html"),
        type=Path,
        help="companion timeline review to link to",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    sidecar_path = args.sidecar if args.sidecar.is_absolute() else root / args.sidecar
    output_path = args.output if args.output.is_absolute() else root / args.output

    if not sidecar_path.exists():
        raise SystemExit(f"sidecar not found: {sidecar_path}")
    annotations = json.loads(sidecar_path.read_text()).get("annotations", [])
    if not annotations:
        raise SystemExit(f"no annotations in {sidecar_path}")

    channels = resolve_crop_channels(
        args.crop_channels,
        {a.get("channel") for a in annotations if a.get("channel")},
    )
    frames = build_frames(annotations, root, output_path.parent, channels)
    if not frames:
        available = sorted({a.get("channel") for a in annotations if a.get("channel")})
        raise SystemExit(
            "no frames to show.\n"
            f"  --crop-channels {args.crop_channels} resolved to: "
            f"{'all' if channels is None else ', '.join(sorted(channels)) or 'none'}\n"
            f"  channels in {sidecar_path.name}: {', '.join(available) or 'none'}"
        )

    stats = summarize(frames)
    channel_label = "all" if channels is None else ",".join(sorted(channels))
    subtitle = (
        f"{stats['n_frames']} frames &middot; "
        f"{stats['n_annotations']} boxes &middot; "
        f"{stats['n_unmatched']} unmatched to a map RE &middot; "
        f"channels: {html.escape(channel_label)}"
    )

    links = companion_links(
        root, output_path.parent, map_view=args.map_view, timeline=args.timeline
    )
    page = PAGE_TEMPLATE
    page = (
        page.replace("__SUBTITLE__", subtitle)
        .replace("__FRAMES__", json.dumps(frames, ensure_ascii=False))
        .replace("__STATS__", json.dumps(stats, ensure_ascii=False))
        .replace("__LINKS__", json.dumps(links, ensure_ascii=False))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page)
    print(
        f"wrote {output_path} "
        f"({stats['n_frames']} frames, {stats['n_annotations']} boxes, "
        f"{stats['n_unmatched']} unmatched)"
    )


PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TLR frame review</title>
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
  .toolbar { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
    margin-top:9px; }
  .toolbar .grow { flex:1; min-width:180px; }
  button, select { font:inherit; border:1px solid var(--line); border-radius:4px;
    background:#fff; color:var(--ink); padding:5px 9px; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  button:disabled { opacity:.45; cursor:default; }
  input[type=range] { width:100%; }
  label.chk { display:inline-flex; align-items:center; gap:5px; color:var(--muted);
    font-size:12px; cursor:pointer; }
  .viewlink { color:var(--accent); text-decoration:none; font-size:12px;
    border:1px solid var(--line); border-radius:4px; padding:5px 9px; }
  .viewlink:hover { border-color:var(--accent); }
  main { padding:14px 16px 28px; }
  .frameInfo { font-size:12px; color:var(--muted); margin-bottom:8px; }
  .frameInfo b { color:var(--ink); }
  .imageWrap { position:relative; display:block; background:#111;
    border:1px solid var(--line); border-radius:5px; overflow:hidden;
    line-height:0; }
  .imageWrap img { display:block; width:100%; height:auto; }
  .boxOverlay { position:absolute; border:2px solid var(--unknown);
    box-sizing:border-box; border-radius:2px; pointer-events:none; }
  .boxOverlay.unmatched { border-style:dashed; }
  .boxOverlay.sel { box-shadow:0 0 0 2px #fff, 0 0 0 4px var(--accent); }
  .boxTag { position:absolute; transform:translateY(-100%);
    font:600 11px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
    color:#fff; padding:1px 4px; border-radius:2px 2px 0 0; white-space:nowrap;
    pointer-events:none; }
  .crops { display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
  .crop { border:1px solid var(--line); border-radius:5px; background:var(--panel);
    padding:8px; width:190px; cursor:pointer; }
  .crop.sel { border-color:var(--accent); box-shadow:0 0 0 2px rgba(36,87,197,.18); }
  .crop.unmatched { border-left:4px solid var(--bad); }
  .crop canvas { display:block; width:100%; aspect-ratio:1/1; background:#111;
    border-radius:3px; }
  .crop .state { margin-top:6px; font-weight:700; font-size:13px;
    padding:2px 6px; border-radius:3px; color:#fff; display:inline-block; }
  .crop .meta { margin-top:5px; color:var(--muted); font-size:11px;
    line-height:1.5; word-break:break-all; }
  .crop .meta .warn { color:var(--bad); font-weight:600; }
  .legend { margin-top:14px; color:var(--muted); font-size:11px; }
  .legend span { margin-right:12px; }
  .swatch { display:inline-block; width:10px; height:10px; border-radius:2px;
    vertical-align:-1px; margin-right:3px; }
  .empty { color:var(--muted); padding:20px 0; }
</style>
</head><body>
<header>
  <h1>Traffic signal frame review</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="toolbar">
    <select id="channelSel"></select>
    <button id="prevBtn" type="button">&larr; Prev</button>
    <button id="nextBtn" type="button">Next &rarr;</button>
    <button id="playBtn" type="button">Play</button>
    <span class="grow"><input id="slider" type="range" min="0" value="0"></span>
    <label class="chk"><input type="checkbox" id="onlyUnmatched"> frames with unmatched only</label>
    <a id="mapLink" class="viewlink" href="#">Map view &rarr;</a>
    <a id="timelineLink" class="viewlink" href="#">Timeline review &rarr;</a>
  </div>
</header>
<main>
  <div class="frameInfo" id="frameInfo"></div>
  <div class="imageWrap" id="imageWrap"><img id="frameImg" alt=""></div>
  <div class="crops" id="crops"></div>
  <div class="legend">
    <span><i class="swatch" style="background:var(--red)"></i>red</span>
    <span><i class="swatch" style="background:var(--amber)"></i>amber</span>
    <span><i class="swatch" style="background:var(--green)"></i>green</span>
    <span><i class="swatch" style="background:var(--unknown)"></i>unknown</span>
    <span>solid border = matched to a map RE</span>
    <span>dashed border = unmatched (timeline review cannot reach it)</span>
    <span>&larr;/&rarr; step, space play/pause</span>
  </div>
</main>
<script>
const FRAMES = __FRAMES__;
const STATS = __STATS__;
const LINKS = __LINKS__;
const CROP_PAD = 2.6;      // zoom window as a multiple of the box longer side
const CROP_MIN = 48;       // never zoom tighter than this many source pixels
let visible = [];          // frame indices currently reachable by nav
let pos = 0;               // position within `visible`
let selected = null;       // selected annotation token
let playTimer = null;
const img = document.getElementById('frameImg');

function stateColor(state) {
  const s = state || '';
  if (s.includes('red')) return 'var(--red)';
  if (s.includes('amber') || s.includes('yellow')) return 'var(--amber)';
  if (s.includes('green')) return 'var(--green)';
  return 'var(--unknown)';
}

function channels() {
  return [...new Set(FRAMES.map(f => f.channel))].sort();
}

function rebuildVisible(keepFrame) {
  const channel = document.getElementById('channelSel').value;
  const onlyUnmatched = document.getElementById('onlyUnmatched').checked;
  visible = FRAMES
    .filter(f => f.channel === channel)
    .filter(f => !onlyUnmatched || f.anns.some(a => !a.matched))
    .map(f => f.i);
  const slider = document.getElementById('slider');
  slider.max = Math.max(visible.length - 1, 0);
  // Keep the reviewer near where they were when a filter changes.
  const at = keepFrame == null ? -1 : visible.indexOf(keepFrame);
  pos = at >= 0 ? at : Math.min(pos, Math.max(visible.length - 1, 0));
  slider.value = pos;
}

function currentFrame() {
  if (!visible.length) return null;
  return FRAMES[visible[Math.min(pos, visible.length - 1)]];
}

function show(newPos) {
  if (!visible.length) return;
  pos = (newPos + visible.length) % visible.length;
  document.getElementById('slider').value = pos;
  selected = null;
  render();
}

// Companion views follow whatever frame is on screen, so switching view keeps
// the moment rather than restarting at frame 0.
function syncViewLinks() {
  const frame = currentFrame();
  const q = frame ? `#ch=${encodeURIComponent(frame.channel)}&t=${frame.timestamp}` : '';
  document.getElementById('mapLink').href = LINKS.map_view + q;
  document.getElementById('timelineLink').href = LINKS.timeline + q;
}

function render() {
  const frame = currentFrame();
  const info = document.getElementById('frameInfo');
  const crops = document.getElementById('crops');
  syncViewLinks();
  if (!frame) {
    info.textContent = '';
    crops.innerHTML = '<div class="empty">No frames match the current filter.</div>';
    img.removeAttribute('src');
    document.querySelectorAll('.boxOverlay, .boxTag').forEach(n => n.remove());
    return;
  }
  const nUnmatched = frame.anns.filter(a => !a.matched).length;
  info.innerHTML =
    `<b>${pos + 1} / ${visible.length}</b> &middot; ${escapeHtml(frame.filename)}`
    + ` &middot; ${frame.anns.length} box(es)`
    + (nUnmatched ? ` &middot; <b>${nUnmatched} unmatched</b>` : '')
    + ` &middot; t=${frame.timestamp}`;
  if (img.getAttribute('src') !== frame.src) {
    img.src = frame.src;
  } else {
    drawFrame();
  }
  preload(pos + 1);
}

function preload(nextPos) {
  if (!visible.length) return;
  const frame = FRAMES[visible[(nextPos + visible.length) % visible.length]];
  if (frame) new Image().src = frame.src;
}

img.addEventListener('load', drawFrame);

function drawFrame() {
  drawOverlays();
  drawCrops();
}

function drawOverlays() {
  const wrap = document.getElementById('imageWrap');
  wrap.querySelectorAll('.boxOverlay, .boxTag').forEach(n => n.remove());
  const frame = currentFrame();
  if (!frame || !img.naturalWidth) return;
  const W = img.naturalWidth, H = img.naturalHeight;
  for (const ann of frame.anns) {
    const [x0, y0, x1, y1] = ann.box;
    const color = stateColor(ann.state);
    const box = document.createElement('div');
    box.className = 'boxOverlay' + (ann.matched ? '' : ' unmatched')
      + (selected === ann.token ? ' sel' : '');
    box.style.left = (x0 / W * 100) + '%';
    box.style.top = (y0 / H * 100) + '%';
    box.style.width = ((x1 - x0) / W * 100) + '%';
    box.style.height = ((y1 - y0) / H * 100) + '%';
    box.style.borderColor = color;
    wrap.appendChild(box);

    const tag = document.createElement('div');
    tag.className = 'boxTag';
    tag.textContent = ann.state;
    tag.style.left = (x0 / W * 100) + '%';
    tag.style.top = (y0 / H * 100) + '%';
    tag.style.background = color;
    wrap.appendChild(tag);
  }
}

function drawCrops() {
  const host = document.getElementById('crops');
  host.innerHTML = '';
  const frame = currentFrame();
  if (!frame || !img.naturalWidth) return;
  for (const ann of frame.anns) {
    host.appendChild(cropCard(ann));
  }
}

function cropCard(ann) {
  const card = document.createElement('div');
  card.className = 'crop' + (ann.matched ? '' : ' unmatched')
    + (selected === ann.token ? ' sel' : '');
  const canvas = document.createElement('canvas');
  canvas.width = 180;
  canvas.height = 180;
  card.appendChild(canvas);
  drawCrop(canvas, ann.box);

  const state = document.createElement('div');
  state.className = 'state';
  state.textContent = ann.state || 'unknown';
  state.style.background = stateColor(ann.state);
  card.appendChild(state);

  const meta = document.createElement('div');
  meta.className = 'meta';
  const rows = [`kind=${ann.kind || '-'} score=${ann.score}`,
                `vis=${ann.vis || '-'} status=${ann.status || '-'}`];
  if (ann.raw_state && ann.raw_state !== ann.state) rows.push(`raw=${ann.raw_state}`);
  if (ann.matched) {
    rows.push(`way=${ann.way || '-'} re=${ann.re || '-'}`);
  } else {
    rows.push(`<span class="warn">unmatched: ${escapeHtml(ann.reason || 'n/a')}</span>`);
    rows.push(`candidate=${escapeHtml(ann.cand || '-')}`);
  }
  meta.innerHTML = rows.join('<br>');
  card.appendChild(meta);

  card.addEventListener('click', () => {
    selected = selected === ann.token ? null : ann.token;
    drawOverlays();
    drawCrops();
  });
  return card;
}

function drawCrop(canvas, box) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const W = img.naturalWidth, H = img.naturalHeight;
  const [x0, y0, x1, y1] = box;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  let size = Math.max(Math.max(x1 - x0, y1 - y0) * CROP_PAD, CROP_MIN);
  size = Math.min(size, Math.max(W, H));
  let sx = cx - size / 2, sy = cy - size / 2;
  sx = Math.max(0, Math.min(sx, W - size));
  sy = Math.max(0, Math.min(sy, H - size));
  ctx.imageSmoothingEnabled = false;   // keep small signals crisp when upscaled
  ctx.drawImage(img, sx, sy, size, size, 0, 0, canvas.width, canvas.height);
  // Redraw the box in crop space so the classified region is unambiguous.
  const k = canvas.width / size;
  ctx.strokeStyle = '#ffdd33';
  ctx.lineWidth = 2;
  ctx.strokeRect((x0 - sx) * k, (y0 - sy) * k, (x1 - x0) * k, (y1 - y0) * k);
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text == null ? '' : String(text);
  return d.innerHTML;
}

function setPlaying(on) {
  const btn = document.getElementById('playBtn');
  if (on && !playTimer) {
    playTimer = setInterval(() => show(pos + 1), 400);
    btn.textContent = 'Pause';
    btn.classList.add('primary');
  } else if (!on && playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    btn.textContent = 'Play';
    btn.classList.remove('primary');
  }
}

const channelSel = document.getElementById('channelSel');
for (const ch of channels()) {
  const opt = document.createElement('option');
  opt.value = ch;
  opt.textContent = ch;
  channelSel.appendChild(opt);
}
channelSel.addEventListener('change', () => { rebuildVisible(null); show(0); });
document.getElementById('onlyUnmatched').addEventListener('change', () => {
  const frame = currentFrame();
  rebuildVisible(frame ? frame.i : null);
  show(pos);
});
document.getElementById('prevBtn').addEventListener('click', () => show(pos - 1));
document.getElementById('nextBtn').addEventListener('click', () => show(pos + 1));
document.getElementById('playBtn').addEventListener('click', () => setPlaying(!playTimer));
document.getElementById('slider').addEventListener('input', ev => {
  setPlaying(false);
  show(parseInt(ev.target.value, 10));
});
document.addEventListener('keydown', ev => {
  if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT') return;
  if (ev.key === 'ArrowLeft') { setPlaying(false); show(pos - 1); }
  else if (ev.key === 'ArrowRight') { setPlaying(false); show(pos + 1); }
  else if (ev.key === ' ') { ev.preventDefault(); setPlaying(!playTimer); }
});
window.addEventListener('resize', drawOverlays);

// Deep link from the timeline review: #token=<annotation> jumps to the frame
// holding that box and selects it; #ch=&t= addresses a frame directly.
function applyHash() {
  const q = new URLSearchParams(location.hash.replace(/^#/, ''));
  const token = q.get('token'), ch = q.get('ch'), t = q.get('t');
  let target = null;
  if (token) {
    target = FRAMES.find(f => f.anns.some(a => a.token === token));
  } else if (t) {
    target = FRAMES.find(f => String(f.timestamp) === t && (!ch || f.channel === ch));
  } else if (ch) {
    target = FRAMES.find(f => f.channel === ch);
  }
  if (!target) return false;
  // A filtered-out target would otherwise land the reviewer somewhere else.
  if (!FRAMES.filter(f => f.channel === target.channel)
             .some(f => f.i === target.i && (!document.getElementById('onlyUnmatched').checked
                                             || f.anns.some(a => !a.matched)))) {
    document.getElementById('onlyUnmatched').checked = false;
  }
  document.getElementById('channelSel').value = target.channel;
  rebuildVisible(null);
  const at = visible.indexOf(target.i);
  if (at < 0) return false;
  pos = at;
  document.getElementById('slider').value = pos;
  selected = token || null;
  render();
  return true;
}

window.addEventListener('hashchange', () => { setPlaying(false); applyHash(); });

rebuildVisible(null);
if (!applyHash()) show(0);
</script></body></html>
"""


if __name__ == "__main__":
    main()
