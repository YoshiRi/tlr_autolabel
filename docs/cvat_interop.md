# CVAT interop contract (L4, A' sidecar <-> CVAT)

信号アノテーションのCVAT往復(人手検証ラウンド)の契約。ツール:
`export_cvat_signal_task.py`(A' sidecar → CVATタスクzip)/
`import_cvat_signal_annotations.py`(CVAT XML → A' sidecar)。
状態語彙は README の正準stateトークン仕様(`{color}-{shape}[-{direction}]`、
ソート・カンマ結合)を全属性で共有する。

確定日: 2026-07-19(タスク1)。旧契約(dataset側
`docs/cvat_signal_interop_spec.md` の select型 `state`)はこの版で置き換え。

## A' sidecar: `traffic_signal_2d/v2`

`annotation/traffic_signal_2d_ann.json`。v1からの変更は `attributes` の契約のみ
(テーブル構造は不変)。v1読み込みフォールバック: `raw_state` が無ければ
`detector_signal` を読む。A' はCVATレビュー用の内部形式であり、t4dataset納品時の
map identity は t4devkit 定義の `annotation/traffic_light.json`
(`instance_token -> traffic_light_linestring_id`) に書く。RE / group relation は
mapから解決する。`traffic_signal_2d_ann.json` 自体は標準t4dataset B/B'ではない。

```jsonc
{
  "schema_version": "traffic_signal_2d/v2",
  "source": "map_projection_auto | cvat",
  "annotations": [
    {
      "token": "…",                    // stable id (CVAT annotation_uid と往復)
      "sample_token": "…", "sample_data_token": "…",
      "channel": "CAM_FRONT", "filename": "data/CAM_FRONT/00000.jpg",
      "timestamp": 1783325716091628,
      "label": "traffic_light",
      "box2d": [x0, y0, x1, y1],
      "occluded": false, "z_order": 0,
      "attributes": { /* 下表 */ }
    }
  ]
}
```

## 属性契約(sidecar `attributes` = CVATラベル属性)

| attribute | CVAT型 | 値 / default | 編集 | 意味 |
|---|---|---|---|---|
| `state` | text | 正準state文字列 / `unknown` | **する** | GTの信号状態。正準トークンのソート・カンマ結合。importで検証 |
| `signal_kind` | select | `vehicle` `pedestrian` `other` `unknown` / 導出値 | する | ped要素あり→`pedestrian`、要素あり→`vehicle`、なし→`unknown` で初期化 |
| `visibility` | select | `full` `partial` `occluded` `unknown` / source別に既定 | する | **判定軸=「stateが読めるか」**。full=全体見える / partial=一部隠れ・見切れ・灯火一部だが**読める** / occluded=大半隠れ・枠外で**読めない**(信号は在る) / unknown=未確認。見切れ(truncation)もこの軸に畳む。既定: 検出box=full、map-presence=unknown |
| `review_status` | select | `unchecked` `accepted` `rejected` `fixed` / `unchecked` | **する** | レビュー状態。評価のGTは `accepted`+`fixed` |
| `map_traffic_light_id` | text | lanelet2 way id / 自動対応付け結果 | する | CVAT上のレビュー用キャッシュ。誤対応の修正可。空=対応なし。B'では `traffic_light.json.traffic_light_linestring_id` に反映 |
| `regulatory_element_id` | text | relation idカンマ結合 / 導出値 | しない | 表示専用。B'では `traffic_light_linestring_id` からmapで解決 |
| `facing` | text | `front` `back` / 導出値(未マッチは空) | しない | 灯器面の向き(linestring方向-90°回転の法線、本地図で実証済み)。`back` は筐体裏面の検出 — colored stateなら誤対応疑い |
| `raw_state` | text | autolabelの元state | しない | 検出器の原文。人手修正後も不変(diff用) |
| `detector_score` | text | 検出スコア / 空(手動box) | しない | provenance |
| `source_type` | select | `manual` `projected_map` `auto` `interpolated` `map_presence` / CVAT新規box=`manual`(先頭)、自動生成分は `auto` 値で出力 | しない | 由来の区別 |
| `annotation_uid` | text | sidecar token | しない | 往復キー。空の新規boxはimportで採番 |
| `map_candidate_id` | text | 最近傍候補way id / 空 | しない | 未マッチ検出の**軟対応**。誤対応の手掛かり(authoritativeな `map_traffic_light_id` は空のまま) |
| `regulatory_element_id_candidate` | text | 候補wayのRE / 空 | しない | 同上。未マッチでもRE情報を残す |
| `unmatched_reason` | text | 未マッチ理由 / 空 | しない | `candidate_taken`/`beyond_gate`/`state_unknown_backside`/`geometry_mismatch`/`no_map_candidate_in_view` |

### 設計判断の記録

- **`state` はtext(正準文字列)を維持し、`circle_state`/`arrow_state` のselect分解は不採用。**
  理由: 実データに同時多方向矢印(`green-arrow-up,green-arrow-up_left` 等)や
  ped・amber複合があり、selectでは表現できないか属性爆発する。編集可能な
  分解フィールドを併設すると `state` と二重管理になり乖離する。タイポ対策は
  select ではなく **import時の厳格検証**(下記)で行う。
- **状態の大量修正はCVATではなくRE timeline reviewで行う。**
  CVAT上の `state` は局所修正・互換・diff表示のために残すが、複数フレーム/
  複数カメラ/複数REにまたがる信号状態は
  `traffic_signal_re_review/v1`(`docs/re_timeline_review.md`)で
  `signal_group_id` と時間区間に対して管理し、`apply_re_review.py` で A' sidecar へ
  伝播する。CVATは bbox / visibility / reject / map id 修正に集中させる。
- **`regulatory_element_id` はA'上では導出/表示値**: 1つのTL wayを複数レーンの
  regulatory elementが共有するため人手管理は誤りやすい。t4dataset B'では
  t4devkit定義の `traffic_light.json` が `instance_token` と
  `traffic_light_linestring_id` のRelationだけを持ち、RE / group relation はmapから解く。
- CVAT実機検証済みのzipレイアウトを維持: `annotations.xml` +
  `images/{CHANNEL}_{frame}.jpg`(`images/` 直下必須)。

## import時の検証(fail loud)

1. `state` の全トークンが正準文法(`state_tokens.CANON_RE`、旧形式は自動正規化)
   でパースできること。不明トークンは **エラーで列挙して失敗**(黙って落とさない)。
2. `state` はパース結果から再シリアライズして正規化(順序・重複ゆらぎ吸収)。
3. `map_traffic_light_id` が地図に存在するway idであること(空は可)。
4. `regulatory_element_id` はA'表示用に地図/traffic-light関係から整合化して上書き。
5. `signal_kind` が空なら `state` から導出。
6. box新規(annotation_uid空)は token 採番、`source_type=manual` を既定。

## クイックスタート(毎回これだけ)

```bash
# 全カメラのCVATレビューzipを生成(推論なし、既存ラベルから。数十秒/カメラ)
./make_cvat_review.sh <dataset_root>
# 1カメラだけ / 怪しいフレームだけ:
./make_cvat_review.sh <dataset_root> CAM_FRONT
MIN_PRIORITY=1.5 ./make_cvat_review.sh <dataset_root> CAM_FRONT
# A' sidecarを作り直す(検出は再利用、地図付与だけやり直す):
REFRESH=1 ./make_cvat_review.sh <dataset_root>
```
zipは `<dataset>/build/cvat_signal/` に出る。同フォルダに `cvat_labels.json` も出力される
(下記ラベル定義)。CVAT取り込み〜書き戻しは下記。

**ラベル定義は貼り付けるだけ**: 新規タスク作成時、ラベルの "Raw" エディタに
`cvat_labels.json` の中身を貼る。これで visibility / review_status / signal_kind /
source_type が全部**ドロップダウン**になる(手入力不要)。

## ワークフロー(レビューラウンド)

```bash
# 1. A' sidecar生成(match_traffic_lights.py が v2 属性で出力)
python3 match_traffic_lights.py --dataset-root <dataset>
#    (RE検証レポートも: aggregate_regulatory_signals.py --dataset-root <dataset>)
# 2. CVATタスクzip生成。--min-priority で「怪しいフレームだけ」の集中タスクに
#    できる(下記「高速レビュー」参照)
python3 export_cvat_signal_task.py --dataset-root <dataset> --camera CAM_FRONT \
    --count 0 --min-priority 1.0
# 3. CVATへzipをタスクとしてインポート → レビュー(state修正、review_status設定)
# 4. CVAT XMLエクスポート → 取り込み(検証+正規化)
python3 import_cvat_signal_annotations.py exported.xml --dataset-root <dataset> \
    --output annotation/traffic_signal_2d_ann.json
```

評価(タスク4)は `review_status in {accepted, fixed}` をGT、`raw_state` を
autolabel予測として突き合わせる。

状態レビューをRE timeline側で行う場合は、CVAT XMLをimportして bbox/visibility
修正をA' sidecarへ戻した後に:

```bash
python3 aggregate_regulatory_signals.py --dataset-root <dataset>
python3 render_re_review_timeline.py --dataset-root <dataset>
python3 apply_re_review.py --dataset-root <dataset> \
  --review annotation/traffic_signal_re_review.json \
  --output annotation/traffic_signal_2d_ann.reviewed.json
```

この流れでは、CVAT import済みの幾何と `visibility` を保持したまま、RE単位の
状態決定だけを `state` / `review_status` に反映する。

## 高速レビュー(bbox オートラベル検証)

エクスポータは各 box に triage 用属性を付与する(export 時に算出、import は無視):

- `review_priority`(number): 高いほど優先。box-local(未マッチ +1.0 / 低スコア
  `(0.7-score)*2` / state=unknown +0.5)+ RE検証レポートのフラグ由来
  (`review_priority = n_flags + (1-confidence)`)の合算。
- `flags`(text): 優先度の理由(`unmatched`, `low_score:0.64`,
  `cross_head_state_disagreement` 等)。

CVAT 側の速い回し方:
- **フィルタ**: `review_priority > 1.5` で怪しい box だけ表示 → 全数を見ない。
- **ショートカット**: `F`/`D` フレーム移動、`Tab` オブジェクト巡回、
  `review_status` を数字キーで `accepted`/`fixed`/`rejected` に即設定。
- **一括**: 大半が正しければ select-all → `review_status=accepted`、間違いだけ個別。

`--min-priority T` は「max box priority ≥ T」のフレームだけの集中タスクにする
(c1af6a38 CAM_FRONT: 578→120フレーム)。`--no-images` で XMLのみの軽量zip。

**フレーム順は撮影時刻順(=ファイル名順)に固定**。CVAT はアップロード画像を
名前順にフレーム 0..N と並べ、CVAT-for-images-1.1 の `<image id>` をフレーム
番号として使うため、id は必ず名前ソート順に一致させている。優先度で物理的に
並べ替えると id↔画像がずれて **box が別フレームに乗り**、ズレて見える(旧
`--worst-first` はこの理由で撤去)。worst-first レビューは CVAT 側で
`review_priority` フィルタ+ソートで行う。

## deepen(Tier C、対応表のみ)

変換ロジックは他リポジトリ管理。語彙対応は `configs/state_vocab/deepen.yaml`
(タスク5で作成)を参照。
