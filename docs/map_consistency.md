# Map / image consistency check

Purpose: turn "the map and the images disagree" from something you notice while
scrolling into a number, so a map replacement can be judged by whether the
number moved.

Companion to the review views in [re_timeline_review.md](re_timeline_review.md),
which made the disagreement *visible*. This makes it *measurable*.

```bash
python3 -m tlr_autolabel.review.re_map_consistency --dataset-root data/<ds>
# --output build/tl_match/map_consistency.json  writes the full report
# --fail-on-finding                             exits 1 when anything is found (CI)
```

## How it decides

The check is deliberately **independent of the matcher's verdict**. It
re-projects every lanelet2 traffic light into each frame with
`project_traffic_lights()` -- the same projection the matcher uses -- and
associates the results with the annotated boxes in image space. A box the
matcher *rejected* still counts as an observation here if it lands where a map
signal projects.

That independence is the point: it separates a matcher that is over-rejecting
from a map that is actually wrong. On cb7fd5c0 way 3595 pairs in 94% of head-on
frames by projection, while the matcher credits only 65% -- so 3595's map entry
is fine and the rejections are the matcher's problem, whereas the 281 unmapped
detections are the map's.

Each frame yields three classes:

| class | meaning |
|---|---|
| `paired` | a projected map signal and a detected box agree |
| `map_only` | a map signal projects as readable, nothing was detected there |
| `image_only` | a box with a readable state, no map signal projects near it |

Association is greedy nearest-centre, closest pairs first, within a radius of
`max(--assoc-min-px, --assoc-scale x projected box's longer side)`. The radius
scales with the projection because map and ego-pose error show up as a larger
pixel offset on a near signal than on a distant one.

Two filters keep the check from crying wolf:

- **`unknown` states are ignored.** They are usually the back of a housing or a
  signal too small to call, and asserting the map is wrong on that basis is
  noise. `--min-detector-score` drops weak detections for the same reason.
- **Only head-on frames count toward observability.** `project_traffic_lights()`
  classifies a steeply angled front face as `front_oblique` and documents that
  its matched rate collapses there (12% at 70-80 degrees). Counting those frames
  manufactures findings: way 3281 is seen only at 67-73 degrees on cb7fd5c0, and
  including them reported it as a phantom map entry when it is simply never
  readable.

Findings are aggregated over the whole run, never per frame. One occluded frame
or one missed detection means nothing; a persistent asymmetry means a map
problem.

## Findings

| kind | raised when | usually means |
|---|---|---|
| `unmapped_signal` | at least `--min-unmapped-boxes` readable detections have no map signal projecting near them | the map is missing entries |
| `signal_never_observed` | a way projects head-on in at least `--min-projected-frames` frames but is never detected | the map holds a signal that is not there |
| `low_observation_rate` | head-on observation rate below `--min-observation-rate` | map position/orientation error, or persistent occlusion |
| `large_projection_offset` | pairs hold, but sit a median `--offset-hint-px` or more from the projection | the map position looks off |

A way raises at most one finding, most severe first, and `unmapped_signal`
ranks above the per-way findings.

## What it reports on cb7fd5c0

```
frames analysed : 668
paired          : 1005
map only        : 114  (projects readable, nothing detected)
image only      : 281  (readable detection, no map signal)

per way:
       way  head-on  oblique  paired   rate  offset px
      3281        0       74       0      -          -
      3289      668        0     665   1.00       50.9
      3591       69        0      51   0.74       47.7
      3595      308        0     289   0.94       50.1

1 finding(s):
  [unmapped_signal] CAM_TRAFFIC_LIGHT_NEAR
      281 boxes with a readable state have no map signal projecting near them
      (pedestrian=278, vehicle=3) -- the map is missing entries
```

Way 3289 pairing in 668 of 668 head-on frames is the control: projection,
calibration and time sync are sound, so the remaining asymmetry is about the
map's contents rather than the geometry pipeline.

`image_only` is 281 on `CAM_TRAFFIC_LIGHT_NEAR` and **0** on
`CAM_TRAFFIC_LIGHT_FAR`, which is what a missing *pedestrian* signal should
look like -- those are only readable close up.

The per-way median offsets (+49.8 / -0.7 / -31.6 px in x, differing in sign)
are not a shared bias, so there is no global calibration correction to make;
they are per-way position discrepancies.

## After replacing the map

Re-run against the new map and compare. `image_only` falling toward zero is the
signal that the missing entries were added; `map_only` and
`signal_never_observed` appearing is the signal that something was added that
is not really there.
