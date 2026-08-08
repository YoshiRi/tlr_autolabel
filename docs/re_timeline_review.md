# RE timeline review contract (L4.5)

Purpose: make traffic-light state review operate at the natural unit: a
physical signal group over time. CVAT remains the geometry editor. This layer
does not replace CVAT; it removes the need to repeat the same state attribute
on every bbox/frame/camera.

Confirmed on 2026-07-21 with the existing c1af6a38 outputs, without rerunning
L1 inference.

## Responsibility split

| layer | Owns | Does not own |
|---|---|---|
| CVAT / A' import | `box2d` creation, false-positive rejection, per-frame `visibility` fixes, `map_traffic_light_id`, local bbox additions | Bulk state propagation across frames |
| RE timeline review | `state`/`review_status` over a physical signal group/time interval, `visibility` over a physical-signal × **camera channel** / time interval, and single-frame `box2d` corrections spotted while reviewing | Geometry authoring/deletion, raw detector provenance, map matching |
| L6 evaluation | Reads reviewed A' sidecar / derived GT | Generates or edits GT |

Visibility is a bulk-editable exception to "CVAT owns visibility": a passing
vehicle occluding one signal for a stretch of frames is exactly the kind of
repetitive per-frame edit the timeline review exists to avoid. Because
occlusion is camera-view-dependent (one camera can be blocked while another
sees the same physical signal fine), visibility segments are tracked and
applied **per camera channel**, not per physical-signal group like state is.

`box2d` correction is a narrower, single-frame exception: CVAT remains the
primary geometry editor (creating boxes, rejecting false positives). The
timeline review only lets a reviewer nudge an *existing* box that drifted
wrong on a specific frame -- typically noticed while reviewing state/visibility
on the same image -- without leaving the tool. Unlike `state`/`visibility`,
geometry does not propagate across time, so each correction targets one exact
annotation (`annotation_token`), not a time interval.

## Signal group identity

Use the physical head group, not a single regulatory element, as the editable
unit:

```text
signal_group_id = "ways:" + comma-joined lanelet2 traffic_light way ids
```

Reason: one physical traffic light can be referenced by multiple
`regulatory_element` relations. A reviewer should correct that state once and
have it propagate to every relation sharing the same member ways.

## Review sidecar: `traffic_signal_re_review/v1`

Default path:

```text
annotation/traffic_signal_re_review.json
```

Schema:

```jsonc
{
  "schema_version": "traffic_signal_re_review/v1",
  "source_timeseries": "annotation/traffic_signal_re_timeseries.json",
  "created_at": "2026-07-21T00:00:00+00:00",
  "groups": [
    {
      "signal_group_id": "ways:2180,2182",
      "member_ways": ["2180", "2182"],
      "regulatory_element_ids": ["10360", "10362", "10364"],
      "decisions": [
        {
          "start_sample_token": "...",
          "end_sample_token": "...",
          "start_timestamp": 1783325718047571,
          "end_timestamp": 1783325720947572,
          "state": "green-arrow-up,red-circle",
          "review_status": "fixed",
          "source": "manual_timeline_review",
          "n_frames": 3,
          "flags": ["cross_head_state_disagreement"],
          "note": ""
        }
      ],
      "visibility_decisions": {
        "CAM_FRONT": [
          {
            "start_sample_token": "...",
            "end_sample_token": "...",
            "start_timestamp": 1783325718047571,
            "end_timestamp": 1783325718947571,
            "visibility": "occluded",
            "review_status": "fixed",
            "source": "manual_timeline_review",
            "n_frames": 3,
            "note": "truck passing in front of the signal"
          }
        ]
      },
      "roi_decisions": [
        {
          "annotation_token": "...",
          "channel": "CAM_FRONT",
          "sample_token": "...",
          "timestamp": 1783325718047571,
          "box2d": [1234.5, 567.0, 1300.2, 620.4],
          "review_status": "fixed",
          "source": "manual_timeline_review",
          "note": "box drifted onto the pole during occlusion"
        }
      ]
    }
  ]
}
```

`state` uses the canonical README vocabulary. `scripts/apply_re_review.py` validates
tokens with the same parser used by CVAT import. `visibility_decisions` is
optional per group and keyed by camera channel; a group with no visibility
corrections to make simply omits the key (or leaves it empty), so existing
`traffic_signal_re_review/v1` files without it remain valid. `roi_decisions`
is likewise optional per group and is a flat list, not keyed by channel --
each entry already carries the exact `annotation_token` (and `channel` for
display) it corrects, so there is nothing to group by.

## Application rules

`scripts/apply_re_review.py` reads A' and this review sidecar, then writes a new
reviewed A' sidecar. It never edits the input file in place.

For an A' annotation, a `decisions` entry matches when:

- `timestamp` is inside `[start_timestamp, end_timestamp]`, and
- `map_traffic_light_id` is in `member_ways`, or the annotation's
  `regulatory_element_id` intersects `regulatory_element_ids`.

Effect by `review_status`:

| status | Effect |
|---|---|
| `accepted` | Set `state`, derive `signal_kind`, set `review_status=accepted` |
| `fixed` | Set `state`, derive `signal_kind`, set `review_status=fixed` |
| `rejected` | Set only `review_status=rejected` |
| `unchecked` | No effect when applying |

A `visibility_decisions[channel]` entry matches the same way, plus the
annotation's own `channel` must equal the decision's channel. Effect by
`review_status`:

| status | Effect |
|---|---|
| `accepted` / `fixed` | Set `visibility` |
| `rejected` / `unchecked` | No effect when applying |

A `roi_decisions` entry matches an annotation when its `annotation_token`
equals the annotation's own `token` -- an exact identity match, not an
interval/way match, since geometry does not propagate across frames the way
state and visibility do. Effect by `review_status`:

| status | Effect |
|---|---|
| `fixed` | Set `box2d` |
| `accepted` / `rejected` / `unchecked` | No effect when applying |

`accepted` intentionally has no effect: it just lets a reviewer mark a frame
as "geometry checked, already correct" in the UI without writing a spurious
identical `box2d`.

State, visibility, and ROI decisions apply independently -- an annotation can
get its `state` updated by a group decision, its `visibility` updated by a
channel decision, and its `box2d` updated by a token-matched ROI decision, all
in the same `apply_re_review.py` run, or any subset of the three.

The following are preserved unless overridden by the rules above: `box2d`,
`occluded`, `visibility`, `map_traffic_light_id`, `regulatory_element_id`,
`raw_state`, `detector_score`, `source_type`, and `annotation_uid`.

## Workflow

Use existing autolabel/map outputs; do not rerun L1 just for review.

```bash
# 1. Optional static editor for manual review. It also generates representative
#    crop candidates from the A' sidecar.
python3 scripts/render_re_review_timeline.py --dataset-root <dataset>

# 2. Optional JSON template. Default status is unchecked.
python3 scripts/make_re_review_template.py --dataset-root <dataset> \
  --output annotation/traffic_signal_re_review.template.json

# 3. Apply a reviewed JSON to the current A' sidecar
python3 scripts/apply_re_review.py --dataset-root <dataset> \
  --review annotation/traffic_signal_re_review.json \
  --output annotation/traffic_signal_2d_ann.reviewed.json

# 4. Verify that the reviewed sidecar still feeds L3/L6
python3 scripts/aggregate_regulatory_signals.py --dataset-root <dataset> \
  --input annotation/traffic_signal_2d_ann.reviewed.json
python3 scripts/evaluate_signals.py --dataset-root <dataset> \
  --sidecar annotation/traffic_signal_2d_ann.reviewed.json
```

## Smoke result on c1af6a38

Using the current autolabel output as a provisional accepted review:

```bash
python3 scripts/make_re_review_template.py --dataset-root <dataset> \
  --review-status accepted \
  --output build/tl_match/re_review.accept_autolabel.json
python3 scripts/apply_re_review.py --dataset-root <dataset> \
  --review build/tl_match/re_review.accept_autolabel.json \
  --output build/tl_match/traffic_signal_2d_ann.review_smoke.json
```

Observed:

- 24 regulatory-element series folded into 8 physical signal groups.
- 62 review segments were emitted.
- 2376 / 2793 A' sidecar annotations received `review_status=accepted`.
- 417 annotations remained `unchecked` because they did not fall under a
  reviewed RE group interval.
- Re-aggregation of the reviewed sidecar completed successfully.

This smoke output is a plumbing check, not a human-verified GT.

## Representative crop candidates

`scripts/render_re_review_timeline.py` generates crop images under
`build/tl_match/re_review_assets/` by default and embeds candidate metadata into
the static HTML. When a reviewer clicks a timeline segment, the right pane
shows the top crop and buttons to switch among the remaining candidates.

Candidate selection is intentionally heuristic and review-oriented:

- match annotations whose `map_traffic_light_id` is in the signal group's
  `member_ways`, or whose `regulatory_element_id` intersects the group;
- keep annotations whose `sample_token` is inside the selected segment, avoiding
  false misses from the few-millisecond difference between `sample.timestamp`
  and per-camera `sample_data.timestamp`;
- use the traffic-light and front-facing camera channels by default:
  `CAM_TRAFFIC_LIGHT_NEAR,CAM_TRAFFIC_LIGHT_FAR,CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT,CAM_FRONT_FAR`
  -- the `CAM_TRAFFIC_LIGHT_*` pair is what T4 datasets normally annotate
  signals on, the `CAM_FRONT*` entries cover datasets that use the general
  forward cameras instead;
- rank by bbox shorter side, detector confidence, visibility, source type, and
  closeness to the segment center in sample order;
- select one best candidate per camera first, then fill the remaining slots by
  global rank.
- hide segments with no crop candidates after channel filtering by default.

Useful knobs:

```bash
python3 scripts/render_re_review_timeline.py --dataset-root <dataset> \
  --crop-candidates 8 \
  --crop-margin 1.5 \
  --crop-channels CAM_TRAFFIC_LIGHT_NEAR,CAM_TRAFFIC_LIGHT_FAR \
  --assets-dir build/tl_match/re_review_assets
```

The crop contains context around the bbox and draws the bbox in yellow. Clicking
the crop opens the original full image.

`--crop-channels` defaults to **`auto`**: every `CAM_*` channel present in the
sidecar is used, minus explicitly rear-facing ones (`CAM_BACK*`, `CAM_REAR*`) —
a rear camera cannot see a signal the ego drives toward. t4devkit permits any
`CAM_*` name, so nothing is hard-coded; a TLR dataset shipping
`CAM_TRAFFIC_LIGHT_NEAR/FAR` and one shipping `CAM_FRONT*` both work with no
flag. The channels actually used are printed in the page header.

Pass an explicit comma-separated list to narrow it, `--crop-channels all` to
disable filtering entirely (includes rear cameras; useful for map/fusion
debugging), or `--show-empty-crop-segments` to keep no-evidence segments visible
instead of hiding them.

If every segment is still filtered out the tool exits with the channels the
sidecar actually contains alongside what `--crop-channels` resolved to, since a
channel-name mismatch is the usual cause of an empty result.

## ROI (box2d) correction

The pre-baked crop above is a lossy, heuristic JPEG meant for quick evidence
browsing, not pixel-accurate editing. The "Edit ROI on this frame" button
under a crop instead opens a live editor against the *original* full-resolution
image (the same one already linked from the crop), so corrections are exact
in the source image's own pixel space rather than a resized/burned-in copy.

The editor:

- computes a tight zoom window around the current box client-side (no new
  image files are generated) -- this is the "strict crop": always centered on
  the box, always mathematically exact, recomputed on demand rather than
  pre-baked at a fixed heuristic margin;
- lets a reviewer drag any of 8 handles (4 corners + 4 edge midpoints) to
  resize, or drag the box body to move it, with numeric `x0/y0/x1/y1` fields
  as an exact-value fallback;
- zooms with the mouse wheel, anchored on the cursor, so the point under the
  pointer stays put; "Recenter zoom" resets to the tight window around the box;
- steps frame-by-frame ("Prev/Next frame") through every annotated frame for
  that signal group and camera channel across the **whole row**, not only the
  selected segment -- a box that drifts at a segment boundary is normally
  fixed from the neighbouring frame, which sits in the adjacent segment. The
  frame counter marks `[outside segment]` once you step past the segment you
  opened the editor from;
- copies the neighbouring frame's box with "Copy ROI from prev/next frame"
  (its saved fix if it has one, else its detected box), which is the common
  case for a short run of drifted boxes. The copy is staged in the editor
  only -- adjust and press "Save ROI" to keep it;
- is reachable from either a state segment or a visibility segment, since ROI
  problems are usually noticed incidentally while reviewing one of those.

Saved corrections are staged in-browser exactly like state/visibility
decisions and are included in the same exported `traffic_signal_re_review/v1`
JSON, as `roi_decisions`.

## Saving back to the dataset (`--serve`)

Opened over `file://` the page is fully static and cannot write to disk --
browsers forbid it -- so "Export JSON" downloads the review file and you move
it into the dataset yourself.

`--serve` removes that step:

```bash
python3 -m tlr_autolabel.review.re_review_timeline \
  --dataset-root data/<dataset> --serve
# -> open http://127.0.0.1:8765/build/tl_match/re_review_timeline.html
```

It serves the dataset root over loopback and lets the page write back. Two
separate files keep "where I am" apart from "what I approved":

| file | written by | holds |
|---|---|---|
| `annotation/traffic_signal_re_review.draft.json` | auto-save, continuously | work in progress |
| `annotation/traffic_signal_re_review.json` | "Export / commit" only | reviewed output |

Reason: a review session is long and interruptible, but the committed file is
what `apply_re_review.py` and L6 consume. Continuous writes to that file would
mean any half-finished session silently becomes the reviewed output. Splitting
them makes crash recovery free without ever publishing unverified state.

**Auto-save is on by default** and targets the draft only: every staged change
(state, visibility, ROI, and JSON import) lands within about half a second, so
an interrupted session loses nothing. The "auto-save draft on every change"
checkbox turns it off; turning it back on flushes what is already staged.

**"Export / commit"** opens a diff of the staged review against the currently
committed file -- added, changed, and removed entries per decision kind, with
before/after values -- and writes only after you confirm. On success the draft
is deleted, so its presence means "unfinished work", not stale residue.
Comparison ignores derived bookkeeping (`source`, `n_frames`); only `state` /
`visibility` / `box2d`, `review_status`, and `note` count as changes.

On load the page prefers the draft, falling back to the committed file, so
reopening resumes exactly where you stopped. `--review` overrides both.

Other guarantees:

- the commit target is `--review-out`, default
  `annotation/traffic_signal_re_review.json`; the draft is derived from it;
- writes are atomic (temp file + rename); the committed file keeps one
  generation of backup at `<target>.bak`, the draft does not need one;
- every endpoint rejects payloads that are not `traffic_signal_re_review/v1`,
  so a stray request cannot truncate either file;
- bursts of edits coalesce into one write and saves never overlap, so the draft
  cannot end up behind the UI from an out-of-order response;
- the server binds `127.0.0.1` only, since these endpoints write to the dataset.

`--output` must live inside `--dataset-root` under `--serve` so the page and
its images share one document root. Datasets that symlink `data/` and `build/`
to a shared sibling are fine -- the check is on the logical path and the
served files follow the links.

"Export JSON" (browser download) still works in both modes and is the only
option over `file://`.

## Per-frame view (`re_frame_view.py`)

The timeline is organised by signal group over time, so it can only show
annotations that matched a map regulatory element -- `annotation_matches_group()`
requires a non-empty `map_traffic_light_id` or `regulatory_element_id`. On the
odaiba cb7fd5c0 dataset that hides **742 of 1642 boxes (45%)**, 482 of them
pedestrian signals.

`re_frame_view.py` answers the complementary question -- "on this image, what
was detected and how was each box classified" -- and shows every box regardless
of map match:

```bash
python3 -m tlr_autolabel.review.re_frame_view --dataset-root data/<dataset>
# -> build/tl_match/re_frame_view.html
```

Read-only: it stages nothing and writes nothing back, so no server is needed
and `file://` is fine.

Signals occupy roughly 46x23 px in a 2880x1860 image, so a whole-frame view
alone cannot convey state. The page pairs them:

- the full frame with state-coloured boxes -- solid = matched to a map RE,
  dashed = unmatched and therefore unreachable from the timeline review;
- a strip of zoomed crops, one per box, each labelled with its `state` plus
  kind/score/visibility and, when unmatched, the `unmatched_reason` and the
  `map_candidate_id` the matcher rejected.

Crops are drawn client-side from the already-loaded frame, so no extra JPEGs
are generated. Navigation is arrow keys / slider / play, with the next frame
preloaded, and a "frames with unmatched only" filter for auditing match
failures.

### What an unmatched box means downstream

Unmatched does **not** mean dropped. `convert.py` reads
`map_traffic_light_id or None` and exports every annotation to `object_ann`,
pedestrian states included (`db_tlr_state()` maps them to
`crosswalk_red`/`crosswalk_green`). What an unmatched box lacks is the
`traffic_light.json` row tying its instance to a map primitive.

Measured on cb7fd5c0: 1562 `object_ann` (80 dropped by the unrelated
`--drop-source-types` default), of which 969 carry a map link and **593 do
not**. So an unmatched box still ships as a labelled 2D annotation; it simply
cannot say which mapped signal it is -- and no reviewer ever saw it.

## Map view (`re_map_view.py`)

Where the frame view shows *what* a box was classified as, this one shows
*where* it happened: a top-down plot of the ego path with the mapped signals
around it, so the geometric failure modes (`geometry_mismatch`,
`beyond_gate`) can be reasoned about at all.

```bash
python3 -m tlr_autolabel.review.re_map_view --dataset-root data/<dataset>
# -> build/tl_match/re_map_view.html
```

Also read-only, also its own HTML. Ego poses come from
`sample_data.ego_pose_token`, signal geometry from `map/lanelet2_map.osm` via
`tlr_autolabel.map.lanelet2`; both are already in the same MGRS local frame,
so no reprojection is involved.

Per frame it draws:

- the surrounding road, so the signal geometry has context to sit in:
  lanelet polygons (road / road_shoulder / crosswalk / pedestrian_lane) plus
  `intersection_area`, `crosswalk_polygon` and `stop_line` ways, from
  `load_lanelet2_context()`. It is a flat background layer and is toggleable,
  since its only job is to make the markers legible;
- the whole ego path, plus the current pose with its heading;
- **filled circle** = a way matched at this frame, coloured by its state;
- **dashed square** = a way the matcher *proposed* (`map_candidate_id`) and
  then rejected, coloured by the state the detector read anyway;
- small dots = other mapped signals, for context only;
- a short stub per signal showing its `facing_axis`, and optional rays from
  the ego to everything observed or proposed.

The map spans kilometres while a run covers a few hundred metres, so only
signals referenced by the run plus anything within `--context-radius`
(default 120 m) are drawn, and road geometry is clipped to the box those
cover -- 155 lanelets of 2535 on cb7fd5c0.

The side panel lists every box at the current frame with its state, way or
rejected candidate, the regulatory elements of that way, and the ego-to-signal
distance -- which is what makes a rejection legible.

### What it showed on cb7fd5c0

Only 4 ways were ever matched. The same way is often matched *and* rejected,
and the distance bands overlap, so distance alone does not explain the
rejections:

| way 3595 | n | ego distance |
|---|---|---|
| matched | 200 | 46.2 – 46.8 m |
| `candidate_taken` | 122 | 46.2 – 46.6 m |
| `geometry_mismatch` | 259 | 37.0 – 54.8 m |
| `beyond_gate` | 44 | 36.1 – 60.4 m |

`beyond_gate` fires at 36 m while matching succeeds at 46 m, and
`candidate_taken` covers exactly the matched band -- meaning several boxes
competed for way 3595 in those frames and all but one lost. Way 3595 is
rejected roughly four times as often as it is matched.

## Moving between the three views

The timeline corrects annotations; the frame and map views explain them. They
cross-link, and the link carries the frame you are on, so switching view keeps
the moment instead of restarting at frame 0.

| from | to | link carries |
|---|---|---|
| timeline (evidence crop) | frame view | `#token=<annotation_token>` -- opens that frame with the box selected |
| timeline (evidence crop) | map view | `#ch=<channel>&t=<timestamp>` |
| frame view | map view / timeline | `#ch=&t=` of the frame on screen |
| map view | frame view / timeline | `#ch=&t=` of the frame on screen |

The header links follow whichever evidence crop is selected; each crop also
carries its own pair of links, which open in a new tab. A hash naming a frame
hidden by the "unmatched only" filter clears the filter rather than landing
somewhere else.

Targets are resolved by `companion_links()` and default to the conventional
`build/tl_match/` filenames, overridable with `--frame-view` / `--map-view` /
`--timeline`. The three pages are generated independently, so a link can point
at a page not generated yet -- deliberate, since forcing a generation order
would be worse than a link you regenerate into:

```bash
python3 -m tlr_autolabel.review.re_review_timeline --dataset-root data/<ds> --serve
python3 -m tlr_autolabel.review.re_frame_view      --dataset-root data/<ds>
python3 -m tlr_autolabel.review.re_map_view        --dataset-root data/<ds>
```

Verified on cb7fd5c0: all 137 timeline crop candidates and all 900 ROI-editor
frames resolve in the frame view, and all 668 frames resolve in the map view.
