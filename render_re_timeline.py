#!/usr/bin/env python3
"""Render annotation/traffic_signal_re_timeseries.json as an interactive HTML timeline.

One row per physical head group (regulatory elements sharing the same member
ways are folded together), x = time. Each observation cell stacks its signal
elements into fixed slots (red top, yellow middle, green bottom, like a real
head); arrows and pedestrian lamps carry a glyph. Verification flags are drawn
as markers above the cell and detailed in the hover tooltip.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

GLYPH = {"up": "↑", "left": "←", "right": "→", "down": "↓",
         "up_left": "↖", "up_right": "↗", "down_left": "↙",
         "down_right": "↘"}

DISAGREEMENT_FLAGS = ("cross_head_state_disagreement", "cross_camera_state_disagreement")


def cell_payload(obs) -> dict:
    """Compact per-observation dict embedded into the page."""
    slots = []
    for e in obs["elements"]:
        glyph = ""
        if e["shape"] == "arrow":
            glyph = GLYPH.get(e["arrow"] or "", "?")
        elif e["shape"] == "ped":
            glyph = "P"
        slots.append({"color": e["color"], "glyph": glyph})
    flags = obs["flags"]
    return {
        "t": obs["timestamp"],
        "state": obs["state"],
        "slots": slots,
        "heads": obs["head_states"],
        "conf": obs.get("confidence", 0),
        "flags": flags,
        "disagree": any(f in DISAGREEMENT_FLAGS for f in flags),
        "suspect": any(f.startswith(("arrow_", "color_", "ped_on", "single_frame"))
                       for f in flags),
    }


def build_rows(series: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for s in series:
        grouped[tuple(s["member_ways"])].append(s)
    rows = []
    for ways, members in sorted(grouped.items(), key=lambda kv: -members_obs(kv[1])):
        rep = max(members, key=lambda s: s["n_observations"])
        rows.append({
            "label": "TL " + "+".join(ways),
            "sublabel": "RE " + ", ".join(m["regulatory_element_id"] for m in members),
            "cells": [cell_payload(o) for o in rep["observations"]],
        })
    return rows


def members_obs(members) -> int:
    return max(m["n_observations"] for m in members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=".", type=Path)
    parser.add_argument("--input", default=Path("annotation/traffic_signal_re_timeseries.json"), type=Path)
    parser.add_argument("--output", default=Path("build/tl_match/re_timeline.html"), type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()

    payload = json.loads((root / args.input).read_text())
    rows = build_rows(payload["series"])
    t_values = [c["t"] for r in rows for c in r["cells"] if c["t"]]
    t0, t1 = min(t_values), max(t_values)
    flag_counts = Counter(f for r in rows for c in r["cells"] for f in c["flags"])

    data_json = json.dumps({"rows": rows, "t0": t0, "t1": t1}, ensure_ascii=False)
    flag_summary = ", ".join(f"{k}: {v}" for k, v in flag_counts.most_common())

    page = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traffic light regulatory element timeline</title>
<style>
  :root {
    --red: #C43025; --yellow: #D69B00; --green: #1E6E35;
    --unknown: #B9C0C7; --surface: #fcfcfb; --panel: #f1f0ee;
    --ink: #1f2328; --ink-2: #57606a; --line: #d9dcdf; --mark: #1f2328;
  }
  @media (prefers-color-scheme: dark) { :root {
    --red: #C13A30; --yellow: #B98A2B; --green: #2C7B49;
    --unknown: #4a5157; --surface: #1a1a19; --panel: #242423;
    --ink: #e8eaed; --ink-2: #9aa4ae; --line: #3a3f44; --mark: #e8eaed;
  } }
  :root[data-theme="dark"] { --red:#C13A30; --yellow:#B98A2B; --green:#2C7B49;
    --unknown:#4a5157; --surface:#1a1a19; --panel:#242423;
    --ink:#e8eaed; --ink-2:#9aa4ae; --line:#3a3f44; --mark:#e8eaed; }
  :root[data-theme="light"] { --red:#C43025; --yellow:#D69B00; --green:#1E6E35;
    --unknown:#B9C0C7; --surface:#fcfcfb; --panel:#f1f0ee;
    --ink:#1f2328; --ink-2:#57606a; --line:#d9dcdf; --mark:#1f2328; }
  body { background: var(--surface); color: var(--ink);
         font: 14px/1.5 system-ui, sans-serif; margin: 24px; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: var(--ink-2); font-size: 12px; margin-bottom: 16px; }
  .legend { display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
            background: var(--panel); border-radius: 8px; padding: 8px 14px;
            font-size: 12px; margin-bottom: 18px; }
  .legend .chip { display: inline-block; width: 14px; height: 10px;
                  border-radius: 3px; vertical-align: -1px; margin-right: 5px; }
  .legend .tick { display: inline-block; width: 10px; height: 3px;
                  background: var(--mark); vertical-align: 2px; margin-right: 5px; }
  .legend .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%;
                 background: var(--mark); vertical-align: 1px; margin-right: 5px; }
  .chart { overflow-x: auto; }
  .row { display: flex; align-items: center; margin-bottom: 10px; }
  .rowlabel { flex: 0 0 210px; font-size: 12px; }
  .rowlabel b { display: block; }
  .rowlabel span { color: var(--ink-2); font-size: 11px; }
  .strip { position: relative; height: 46px; flex: 0 0 auto;
           border-left: 1px solid var(--line); }
  .cell { position: absolute; top: 12px; bottom: 4px; border-radius: 2px;
          overflow: hidden; display: flex; flex-direction: column; cursor: default; }
  .band { flex: 1; display: flex; align-items: center; justify-content: center;
          color: #fff; font-size: 9px; line-height: 1; margin-bottom: 1px; }
  .band:last-child { margin-bottom: 0; }
  .band.arrow { background-image: repeating-linear-gradient(45deg,
      rgba(255,255,255,.55) 0 1px, transparent 1px 3px); }
  .band.ped { background-image: repeating-linear-gradient(135deg,
      rgba(255,255,255,.55) 0 1px, transparent 1px 3px); }
  .cell:hover { outline: 2px solid var(--mark); z-index: 3; }
  .m-tick { position: absolute; top: 4px; height: 3px; background: var(--mark); }
  .m-dot { position: absolute; top: 3px; width: 5px; height: 5px;
           border-radius: 50%; background: var(--mark); }
  .axis { position: relative; height: 20px; margin-left: 210px;
          color: var(--ink-2); font-size: 11px; }
  .axis span { position: absolute; transform: translateX(-50%); }
  .axis .gridline { position: absolute; top: -8px; width: 1px; height: 8px;
                    background: var(--line); }
  #tip { position: fixed; display: none; background: var(--panel);
         color: var(--ink); border: 1px solid var(--line); border-radius: 6px;
         padding: 8px 10px; font-size: 12px; pointer-events: none; z-index: 10;
         max-width: 340px; box-shadow: 0 2px 10px rgba(0,0,0,.25); }
  #tip .t2 { color: var(--ink-2); }
</style></head><body>
<h1>Regulatory element signal timeline</h1>
<div class="sub">__SUB__</div>
<div class="legend">
  <span><span class="chip" style="background:var(--red)"></span>red</span>
  <span><span class="chip" style="background:var(--yellow)"></span>yellow</span>
  <span><span class="chip" style="background:var(--green)"></span>green (solid circle)</span>
  <span><span class="chip" style="background-color:var(--green);background-image:repeating-linear-gradient(45deg, rgba(255,255,255,.55) 0 1px, transparent 1px 3px)"></span>arrow (striped; direction in tooltip)</span>
  <span><span class="chip" style="background-color:var(--green);background-image:repeating-linear-gradient(135deg, rgba(255,255,255,.55) 0 1px, transparent 1px 3px)"></span>pedestrian (cross-striped)</span>
  <span><span class="chip" style="background:var(--unknown)"></span>unknown / no state</span>
  <span><span class="tick"></span>cross-camera/head disagreement</span>
  <span><span class="dot"></span>map-bulb mismatch or single-frame flip</span>
  <span class="t2">gaps = regulatory element not observed</span>
</div>
<div class="chart" id="chart"></div>
<div class="axis" id="axis"></div>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const PX_PER_S = 40, CELL_W = 3;
const span = (DATA.t1 - DATA.t0) / 1e6;
const width = Math.ceil(span * PX_PER_S) + 20;
const chart = document.getElementById('chart');
const colorVar = {red: 'var(--red)', yellow: 'var(--yellow)', amber: 'var(--yellow)', green: 'var(--green)'};
const slotOrder = {red: 0, yellow: 1, amber: 1, green: 2};

for (const row of DATA.rows) {
  const rowEl = document.createElement('div');
  rowEl.className = 'row';
  rowEl.innerHTML = `<div class="rowlabel"><b>${row.label}</b><span>${row.sublabel}</span></div>`;
  const strip = document.createElement('div');
  strip.className = 'strip';
  strip.style.width = width + 'px';
  for (const c of row.cells) {
    const x = ((c.t - DATA.t0) / 1e6) * PX_PER_S;
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.style.left = x + 'px';
    cell.style.width = CELL_W + 'px';
    const slots = [...c.slots].sort((a, b) => (slotOrder[a.color] ?? 9) - (slotOrder[b.color] ?? 9));
    if (!slots.length) {
      cell.innerHTML = `<div class="band" style="background:var(--unknown)"></div>`;
    } else {
      cell.innerHTML = slots.map(s => {
        const cls = s.glyph === 'P' ? 'band ped' : (s.glyph ? 'band arrow' : 'band');
        return `<div class="${cls}" style="background-color:${colorVar[s.color] || 'var(--unknown)'}">` +
          `${CELL_W >= 10 ? s.glyph : ''}</div>`;
      }).join('');
    }
    cell.dataset.tip = JSON.stringify(c);
    strip.appendChild(cell);
    if (c.disagree) {
      const m = document.createElement('div');
      m.className = 'm-tick'; m.style.left = x + 'px'; m.style.width = CELL_W + 'px';
      strip.appendChild(m);
    } else if (c.suspect) {
      const m = document.createElement('div');
      m.className = 'm-dot'; m.style.left = (x + CELL_W / 2 - 2.5) + 'px';
      strip.appendChild(m);
    }
  }
  rowEl.appendChild(strip);
  chart.appendChild(rowEl);
}

const axis = document.getElementById('axis');
axis.style.width = width + 'px';
for (let s = 0; s <= span; s += 5) {
  const g = document.createElement('div');
  g.className = 'gridline'; g.style.left = (s * PX_PER_S) + 'px';
  axis.appendChild(g);
  const lab = document.createElement('span');
  lab.style.left = (s * PX_PER_S) + 'px';
  lab.textContent = s + 's';
  axis.appendChild(lab);
}

const tip = document.getElementById('tip');
chart.addEventListener('mousemove', (ev) => {
  const cell = ev.target.closest('.cell');
  if (!cell) { tip.style.display = 'none'; return; }
  const c = JSON.parse(cell.dataset.tip);
  const t = ((c.t - DATA.t0) / 1e6).toFixed(1);
  const heads = Object.entries(c.heads).map(([w, s]) => `TL ${w}: ${s}`).join('<br>');
  tip.innerHTML = `<b>${c.state}</b> <span class="t2">t=${t}s conf=${c.conf}</span>` +
    `<br>${heads}` +
    (c.flags.length ? `<br><span class="t2">flags: ${c.flags.join(', ')}</span>` : '');
  tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX + 14, innerWidth - 360) + 'px';
  tip.style.top = (ev.clientY + 14) + 'px';
});
chart.addEventListener('mouseleave', () => tip.style.display = 'none');
</script></body></html>
"""
    n_obs = sum(len(r["cells"]) for r in rows)
    sub = (f"{len(rows)} head groups &middot; {n_obs} fused observations &middot; "
           f"flags &mdash; {html.escape(flag_summary)}")
    page = page.replace("__SUB__", sub).replace("__DATA__", data_json)

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out} ({len(rows)} rows, {n_obs} observations)")


if __name__ == "__main__":
    main()
