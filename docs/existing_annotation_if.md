# Existing Annotation IF Notes

Recorded: 2026-07-21
Updated: 2026-07-27

対象:

```text
/home/yoshiri/.webauto/data/data/annotation_dataset/cb7fd5c0-f833-4110-83d3-c4b9446c24e4/0
```

## Position

既存アノテーションをこのリポジトリへ結合するなら、主な接続点は **L3 adapter** とする。

理由:

- L1 は画像から `tlr_autolabel/v1` を作る推論層であり、既存アノテーションを読む層ではない。
- L2 は L1/L3 出力を standard T4 `object_ann.json` へ落とす export 層であり、T4 既存 annotation の取り込み口ではない。
- L3 は T4 dataset、ego pose、calibration、lanelet2 map、RE を扱い、内部 sidecar の A' `traffic_signal_2d/v2` と `traffic_signal_re/v1` を作る層である。
- L4/L4.5/L6 は A' sidecar / reviewed sidecar / RE sidecar、または standard T4 `object_ann.json` からの adapter 入力を前提にするため、既存 annotation はそこへ入る前に L3 相当の adapter で正規化するのが自然。

## Annotation Kinds

既存 annotation は少なくとも 2 系統に分けて扱う。

- **Evaluator Annotation**: 評価用 GT / レビュー結果として使う annotation。このメモの対象。
- **Training Annotation**: 学習用 annotation。standard T4 `object_ann.json` を本体とし、必要なら map identity を t4devkit 定義の `traffic_light.json` で持つ(RE / group relation は map から解く)。Observed training dataset notes are below.

Evaluator Annotation の目標 IF は、評価モードに応じて以下のどちらか:

- reviewed A' sidecar: `annotation/traffic_signal_2d_ann.reviewed.json`
- または A' sidecar + RE review sidecar: `annotation/traffic_signal_2d_ann.json` + `annotation/traffic_signal_re_review.json`

Training Annotation の目標 IF は:

- **B**: `annotation/object_ann.json` + `annotation/category.json` + `annotation/attribute.json` + optional `annotation/instance.json`
- **B'**: B + `annotation/traffic_light.json`

`traffic_light.json` is intentionally outside `object_ann.json` / `instance.json`
and is a t4dataset annotation type defined in t4devkit:

```json
{
  "token": "relation-001",
  "instance_token": "instance-head-001",
  "traffic_light_linestring_id": "501"
}
```

`instance_token` points to the 2D `instance.json` row. The
`traffic_light_linestring_id` points to the lanelet2 traffic-light linestring.
Regulatory-element and group relationships are already in the map and are
resolved from that linestring. This repository must not encode those map
identifiers in `object_ann`, `instance_token`, or `instance_name`.

### B/B' Instance Identity

Current contract: **B `instance.json` does not carry map id**.

`object_ann.instance_token` points to a standard T4 2D annotation instance used
for temporal grouping/tracking of boxes. In the current converter, this
instance is assigned by per-camera bbox continuity and is map-independent.

Map identity is carried only by B':

```text
annotation/traffic_light.json
instance_token -> traffic_light_linestring_id
```

This is deliberate for now:

- standard B remains a normal T4 `object_ann` dataset with no custom map field
- B' can be absent when there is no map; relation absence for an instance means
  no map match
- one 2D tracked instance is not guaranteed to be the same semantic unit as one
  lanelet2 `traffic_light` way, especially around missed frames, camera changes,
  or manual bbox edits

If a downstream evaluator wants one stable physical signal identity, it should
join through `traffic_light.json`. DLR-oriented formats are derived from that
relationship and the map into Traffic Light ID based rows. Do not infer map id
from `instance_token` or `instance_name`.

Deprecated: `annotation/traffic_light_map_association.json` was a transitional
repo-local association table keyed by `object_ann_token`. It is superseded by
t4devkit-defined `annotation/traffic_light.json` and should not be used by new
tools.

Tier classification rule: B' is B plus `annotation/traffic_light.json`. If that
file is absent, the dataset is B. Temporary/deprecated JSON files do not affect
the B/B' distinction.

## IF Relationship Summary

The common payload across training and evaluation is:

```text
sample_data_token
box / bbox
traffic-light state
occlusion / truncation when available
optional traffic-light map relation
```

The difference is the semantic unit:

| IF | File shape | Unit | State location | Map id | Evaluation geometry? |
|---|---|---|---|---|---|
| A' internal sidecar | `traffic_signal_2d/v2` | one detected or reviewed signal box | `attributes.state` canonical lamp string | `attributes.map_traffic_light_id` | yes, if box is real |
| B training | standard T4 `object_ann.json` | one 2D signal box | `category_token` -> db_tlr category name | no | yes, if box is real |
| B' training | B + `traffic_light.json` | one 2D signal box + optional map relation | same as B | `traffic_light.json` by `instance_token` | yes, if box is real |
| Evaluator state annotation observed here | T4 `object_ann.json` | one frame/channel state label | `category_token` -> `red` / `green` | absent | no; bbox is fixed ROI placeholder |

Therefore, B' is structurally compatible with A' for map identity after an
adapter, but the observed Evaluator Annotation is not automatically B/B':
it uses T4 storage, but its bbox and row cardinality indicate state labels,
not per-signal detection boxes.

## Conversion Rules

Allowed conversions:

- A' -> B: convert canonical `attributes.state` to db_tlr category, copy
  `box2d` to `bbox`, map visibility to occlusion, default truncation if absent.
- A' -> B': same as A' -> B, plus write
  `instance_token -> attributes.map_traffic_light_id` as
  `traffic_light.json.instance_token -> traffic_light_linestring_id` according
  to the t4devkit schema.
- B' -> A': join `object_ann.instance_token` to `traffic_light.json`, convert
  db_tlr category back to a canonical evaluator state where possible, copy
  `bbox` to `box2d`, and expose traffic-light / RE fields as A' review
  attributes. RE fields are resolved from the map via
  `traffic_light_linestring_id`.
- B/B' -> detector evaluation GT: valid only when `bbox` is a real per-signal
  2D box.
- Observed Evaluator Annotation -> RE review: valid if the frame-level state
  can be associated with a single RE / traffic-light group for that timestamp.

Blocked or lossy conversions:

- Observed Evaluator Annotation -> B/B' training: not valid as detector
  training without replacing the fixed ROI with real per-signal boxes.
- Observed Evaluator Annotation -> geometry GT: not valid for IoU / detector
  precision because `[960, 620, 1920, 1240]` is a placeholder ROI.
- B without `traffic_light.json` -> RE evaluation: impossible to make
  deterministic unless the map id is reconstructed by a separate L3 matching
  step.
- B' -> exact A lamp decomposition: lossy when db_tlr category does not encode
  all original lamp details.

## Observed Dataset Shape

この Evaluator dataset は T4 dataset としての基本構造を持つ。

- camera data: `data/CAM_TRAFFIC_LIGHT_FAR/*.jpg`, `data/CAM_TRAFFIC_LIGHT_NEAR/*.jpg`
- frames: 668 `sample_data` rows
- channels: 334 frames each for `CAM_TRAFFIC_LIGHT_FAR` and `CAM_TRAFFIC_LIGHT_NEAR`
- T4 tables present: `sample_data.json`, `sample.json`, `ego_pose.json`, `calibrated_sensor.json`, `sensor.json`, etc.
- lanelet2 map present: `map/lanelet2_map.osm`
- map traffic-light content is usable by repo loaders:
  - `traffic_light` ways: 261
  - `light_bulbs` entries: 261
  - traffic-light regulatory elements: 234
  - ways with regulatory element from `match_traffic_lights.py`: 259

Existing annotation files:

- `annotation/object_ann.json`: 668 rows
- `annotation/category.json`: only `red` and `green`
- `annotation/sample_annotation.json`: empty
- `annotation/attribute.json`: empty
- no repo A' / RE sidecars and no B' traffic-light table:
  - `annotation/traffic_signal_2d_ann.json`: absent
  - `annotation/traffic_signal_2d_ann.reviewed.json`: absent
  - `annotation/traffic_signal_re_timeseries.json`: absent
  - `annotation/traffic_signal_re_review.json`: absent
  - `annotation/traffic_light.json`: absent
  - `build/tl_match/match_report.json`: absent
- no L1 output:
  - `tlr_autolabel/**/*.json`: absent

`object_ann.json` details observed:

- 668 annotations, one per `sample_data`.
- category counts:
  - `red`: 484
  - `green`: 184
- bbox is fixed for all rows:
  - `[960, 620, 1920, 1240]`
- `automatic_annotation`: false

This means the current `object_ann.json` is not a normal per-signal 2D box table for L4/L6 geometry evaluation. It looks like an evaluator-state annotation encoded in T4 `object_ann` form, using a fixed region and category-as-state.

## Current Repo IF Expectations

L3 map matching expects L1 detections:

```text
tlr_autolabel/<CHANNEL>/<frame>.json
schema_version = tlr_autolabel/v1
signals[].box_xyxy
signals[].state
signals[].lamps
sample_data_token
```

L3 output / L4 input expects A' internal sidecar:

```text
annotation/traffic_signal_2d_ann.json
schema_version = traffic_signal_2d/v2
annotations[].sample_token
annotations[].sample_data_token
annotations[].channel
annotations[].filename
annotations[].timestamp
annotations[].box2d
annotations[].attributes.state
annotations[].attributes.map_traffic_light_id
annotations[].attributes.regulatory_element_id
annotations[].attributes.review_status
annotations[].attributes.signal_kind
annotations[].attributes.visibility
annotations[].attributes.raw_state
annotations[].attributes.source_type
```

L4.5 / L6 RE review expects:

```text
annotation/traffic_signal_re_timeseries.json
schema_version = traffic_signal_re/v1
series[].regulatory_element_id
series[].member_ways
series[].segments
series[].observations

annotation/traffic_signal_re_review.json
schema_version = traffic_signal_re_review/v1
groups[].member_ways
groups[].regulatory_element_ids
groups[].decisions[].state
groups[].decisions[].review_status
groups[].decisions[].start_timestamp
groups[].decisions[].end_timestamp
```

## Adapter Requirement

To use this Evaluator Annotation without rerunning the normal L1->L3 route as
the source of GT, add an L3-side adapter that converts T4 evaluator annotation
into repo sidecars.

Minimum adapter responsibilities:

1. Read `object_ann.json`, `category.json`, and T4 frame tables.
2. Interpret category names `red` / `green` as evaluator state.
3. Convert state into canonical vocabulary:
   - `red` -> `red-circle`
   - `green` -> `green-circle`
4. Preserve evaluator source separately from model prediction:
   - reviewed GT state should go to `attributes.state`
   - prediction/provenance should not be invented as `raw_state`
5. Decide how to attach map / RE identity:
   - preferred: derive from map projection / existing L3 candidate timeline using `sample_data_token`, channel, timestamp, and lanelet2 RE visibility.
   - if the fixed bbox is intentionally a state-only placeholder, do not treat it as detection geometry for IoU or detector precision.
6. Emit one of:
   - reviewed A' sidecar for L6, if per-head/per-frame mapping and real geometry are known.
   - `traffic_signal_re_review/v1`, if annotation is really an RE/time interval state label rather than per-box geometry.

Open question for implementation:

- Does one `object_ann.json` row correspond to one physical RE visible in that frame, or to a dataset/frame-level signal state label?
- Where is the RE id / traffic light way id encoded, if anywhere?
- Is the fixed bbox `[960, 620, 1920, 1240]` a deliberate evaluator ROI placeholder?
- If the evaluator state is frame-level, which RE / traffic-light group is the intended target when multiple map signals are visible?

Until these are answered, the safe classification is:

```text
Evaluator Annotation: usable as a GT state source after L3-side adapter work.
Existing evaluator object_ann.json: T4-shaped, but not direct B/B' detector GT.
Training B/B': compatible with evaluator only when boxes are real and map id is
present or reconstructable.
```

## Training Annotation Dataset

Recorded: 2026-07-21

対象:

```text
/home/yoshiri/Downloads/TLRv0.1_JapanTaxi5_odaiba_20220909-20220912_194562de-4f65-4ff2-91bb-69ecc20042ea
```

User note mentioned `object_ann.jspn`; the actual files found are:

```text
annotation/object_ann.json
annotation/object_ann_org.json
```

### Position

This dataset matches the **Training Annotation** side, not the Evaluator Annotation side.

It is a T4 native 2D detection dataset where traffic-light state is encoded as the per-box category name. That matches the AWML/db_tlr-style training IF described in README and `configs/state_vocab/db_tlr.yaml`.

Do not use this dataset as direct L3/L4/L6 evaluator input:

- no lanelet2 map directory was found at `map/lanelet2_map.osm`
- no B' traffic-light table or A' / RE sidecars were found
- no `tlr_autolabel/**/*.json` outputs were found

### Observed Training Dataset Shape

- camera channel: `CAM_TRAFFIC_LIGHT_NEAR`
- `sample_data.json`: 1222 rows
- `sample.json`: 1222 rows
- `sensor.json`: 1 camera sensor
- `calibrated_sensor.json`: 1 camera calibration
- `ego_pose.json`: 33711 rows
- `listdata/all.txt`: 1222 rows
- `listdata/train.txt`: 855 rows
- `listdata/val.txt`: 219 rows
- `listdata/test.txt`: 148 rows

Image presence check:

- sample_data rows: 1222
- image files present under `data/`: 1070
- missing image files: 152
- annotated sample_data whose image file is missing: 140
- listdata missing files:
  - all: 152
  - train: 111
  - val: 26
  - test: 15

This is an important packaging issue for training runs; the annotation schema is usable, but the local folder is not a complete image dataset as-is.

### `object_ann.json`

Current training annotation:

- rows: 3941
- annotated sample_data: 1121
- max boxes per sample_data: 9
- bbox:
  - all rows have bbox
  - unique boxes: 3938
  - width min/median/max: 6 / 29 / 304 px
  - height min/median/max: 4 / 17.5 / 184 px
  - area min/median/max: 55 / 504 / 32706 px^2
- `sample_annotation.json`: empty
- `surface_ann.json`: empty

Category counts:

| category | count |
|---|---:|
| `green` | 2082 |
| `red` | 1317 |
| `crosswalk_red` | 215 |
| `crosswalk_green` | 201 |
| `yellow` | 71 |
| `red_right` | 37 |
| `crosswalk_unknown` | 10 |
| `unknown` | 8 |

The bbox distribution is consistent with real per-signal/per-lamp 2D boxes, unlike the Evaluator Annotation dataset whose bbox was fixed to `[960, 620, 1920, 1240]`.

### `object_ann_org.json`

Original training annotation:

- rows: 3515
- annotated sample_data: 1121
- max boxes per sample_data: 7

Category counts:

| category | count |
|---|---:|
| `green` | 2082 |
| `red` | 1317 |
| `yellow` | 71 |
| `red_right` | 37 |
| `unknown` | 8 |

The current `object_ann.json` adds pedestrian categories compared to `object_ann_org.json`:

- `crosswalk_red`
- `crosswalk_green`
- `crosswalk_unknown`

### Category Vocabulary Compatibility

`annotation/category.json` contains:

```text
red
green
yellow
red_right
unknown
crosswalk_red
crosswalk_green
crosswalk_unknown
```

All category names are present in `configs/state_vocab/db_tlr.yaml` `allowed`.

Therefore:

```text
Training Annotation category vocabulary: compatible with repo db_tlr/AWML adapter vocabulary.
Training Annotation storage shape: compatible with Tier B state-as-category training IF.
Training Annotation becomes B' when t4devkit-defined `traffic_light.json` is added.
Training Annotation is not equivalent to Evaluator A'/RE IF without a conversion policy.
```

### Training IF Implications

For training compatibility, this dataset should be treated as the target style
that `to_object_ann.py` writes (standard object_ann):

```text
annotation/object_ann.json
annotation/category.json
category name = traffic-light state
bbox = 2D signal box
attribute_tokens = truncation / light_status / occlusion metadata
optional traffic_light.json = instance_token -> traffic_light_linestring_id
listdata/{train,val,test,all}.txt = split files
```

For evaluator compatibility, do not infer RE timelines or reviewed A' from this
dataset alone unless `traffic_light.json` is present or an L3 matching step can
reconstruct map identity. With B' present, an adapter can produce A' sidecars,
RE timeseries, and DLR-oriented Traffic Light ID based rows from the
relationships in `traffic_light.json`. Without B', only map-less
detector/classification evaluation is well-defined.
