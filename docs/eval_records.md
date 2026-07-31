# L6 evaluation ledgers (`tlr_eval/v2`)

`evaluate_signals.py` emits three **tidy long-format ledgers** as the reusable
analysis artifacts. One row = one unit of observation; every slicing dimension
is a column, so any post-hoc cut (pedestrian vs vehicle, per lamp
color/shape/arrow, facing, channel, distance, matched/unmatched, reviewed/not)
is a group-by — no new evaluation code. The `eval_report.{json,md}` tables are
just default `pivot()` views over these files.

Files (under `build/tl_match/`): `eval_detections.jsonl`,
`eval_candidates.jsonl`, `eval_lamps.jsonl` (JSON Lines: one object per row).

Why three tables and not one: precision-side questions ("of detections, how
many matched / were unknown") and recall-side questions ("of map signals, how
many were detected") have different denominators, so they get different units.
Per-lamp questions need a row per lamp. All three carry shared join keys
(`sample_data_token`, `way_id`, `det_token`) for external re-joining (pandas,
duckdb, etc.).

## `eval_detections.jsonl` — unit: one detected/reviewed A' box

Precision side, state/unknown rates, and (when reviewed) state accuracy.

| column | meaning |
|---|---|
| `row_id` | row index within the file |
| `det_token` | A' annotation token (join key) |
| `sample_token` / `sample_data_token` | T4 frame identity (join keys) |
| `channel` | camera channel |
| `timestamp` | frame timestamp (µs) |
| `det_box` | `[x0,y0,x1,y1]` original px |
| `det_min_side_px` | shorter side of the detected box |
| `detector_score` | detector confidence, or null |
| `state` | canonical detector state (sorted lamp tokens) |
| `signal_kind` | `vehicle` / `pedestrian` / `unknown` (attr, else derived) |
| `n_lamps` | lamp count parsed from `state` |
| `is_unknown` | `state == "unknown"` |
| `way_id` | matched lanelet2 traffic_light way, or null |
| `regulatory_element_id` | comma-joined relation ids, or null |
| `subtype` | map way subtype (`red_yellow_green`, `red_green`, …) |
| `facing` | `front` / `back` / null (of the matched map head) |
| `matched` | whether a map way was associated |
| `iou` | IoU with the projected map box, or null |
| `distance_m` / `distance_bin` | ego→head distance and its bin |
| `review_status` | `unchecked` / `accepted` / `fixed` / `rejected` |
| `source_type` | A' annotation source: `auto` / `tracked` / `propagated` / `interpolated` / `map_presence` / etc. |
| `temporal_source` | evaluation grouping for temporal output: `observed` / `propagated` / `map_presence` / null |
| `track_id` | L3 temporal track id, or null |
| `tracking_status` / `tracking_lost_frames` | temporal track state and lost-frame count, or null |
| `gt_state` | reviewed canonical state (only for accepted/fixed) |
| `state_correct` | `gt_state == state` (null until reviewed) |

## `eval_candidates.jsonl` — unit: one projected map traffic_light

Recall / coverage side: every front/back map candidate projected into a frame
(edge-on already dropped at projection).

| column | meaning |
|---|---|
| `row_id` | row index |
| `sample_data_token` / `channel` | frame identity |
| `way_id` | lanelet2 traffic_light way (join key) |
| `distance_m` / `distance_bin` | ego→head distance and bin |
| `facing` | `front` / `back` / null |
| `facing_deg` | signed angle, face-normal vs sight line |
| `proj_min_side_px` | shorter side of the projected box |
| `too_small` | projected shorter side < 8 px (detector min-box) |
| `detectable` | front-facing and not too small (coverage denominator) |
| `matched` | whether a detection was associated to it |

Coverage / recall = `matched` over `detectable` rows.

## `eval_lamps.jsonl` — unit: one lamp of one detection

Per-lamp (灯火ごと) analysis; exploded from each detection's `state` tokens.

| column | meaning |
|---|---|
| `row_id` | row index |
| `det_token` / `way_id` | join keys back to the detection |
| `channel` / `signal_kind` / `facing` / `distance_bin` / `matched` | inherited detection dims |
| `color` | `red` / `amber` / `green` |
| `shape` | `circle` / `arrow` / `ped` / `u_turn` / `number` / `cross` |
| `arrow` | 8-way direction (arrow lamps), else null |
| `is_arrow` | `shape == "arrow"` |
| `lamp_token` | `{color}-{shape}[-{arrow}]` |

## Cutting the data

The report already includes: detections by signal_kind / facing / channel,
lamps by color×shape and arrow direction, coverage by distance. For anything
else, group by on the jsonl directly, e.g. pedestrian coverage by distance:

```python
import json, collections
rows = [json.loads(l) for l in open("build/tl_match/eval_lamps.jsonl")]
g = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r["shape"] == "ped":
        g[r["distance_bin"]][0] += 1
        g[r["distance_bin"]][1] += r["matched"]
```

The in-code helper `pivot(records, group_by, metrics)` does the same for the
default views and is reusable for new cuts.

## Notes / findings surfaced by the cuts (2026-07-19, pre-GT)

- **Pedestrian heads project with low IoU** (median ~0.12) although they match
  at ~97%: the lanelet2 `red_green` (pedestrian) way geometry and the detected
  ped box align in position but not extent. IoU is a weak signal for ped heads;
  the association is still correct. Worth a dedicated ped projection check.
- **`back`-facing matched detections are ~98% `unknown`**: consistent with the
  detector firing on the housing's back. Colored state on a `back` head is
  flagged `colored_state_on_back_face` by the aggregator (misassociation).
- These are GT-free observations; GT (`review_status`) activates state accuracy
  and element P/R, which are then sliceable on the same columns.
