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
| CVAT / A' import | `box2d`, false-positive rejection, `visibility`, `map_traffic_light_id`, local bbox additions | Bulk state propagation across frames |
| RE timeline review | `state` and `review_status` over a physical signal group/time interval | Geometry, raw detector provenance, map matching |
| L6 evaluation | Reads reviewed A' sidecar / derived GT | Generates or edits GT |

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
      ]
    }
  ]
}
```

`state` uses the canonical README vocabulary. `scripts/apply_re_review.py` validates
tokens with the same parser used by CVAT import.

## Application rules

`scripts/apply_re_review.py` reads A' and this review sidecar, then writes a new
reviewed A' sidecar. It never edits the input file in place.

For an A' annotation, a decision matches when:

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

The following are preserved: `box2d`, `occluded`, `visibility`,
`map_traffic_light_id`, `regulatory_element_id`, `raw_state`,
`detector_score`, `source_type`, and `annotation_uid`.

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
- use front-facing camera channels by default:
  `CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT,CAM_FRONT_FAR`;
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
  --crop-channels CAM_FRONT,CAM_FRONT_LEFT,CAM_FRONT_RIGHT \
  --assets-dir build/tl_match/re_review_assets
```

The crop contains context around the bbox and draws the bbox in yellow. Clicking
the crop opens the original full image.

For map/fusion debugging, pass `--crop-channels all` to allow rear cameras too.
Pass `--show-empty-crop-segments` to keep no-evidence segments visible instead
of hiding them.
