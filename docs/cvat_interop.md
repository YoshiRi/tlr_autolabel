# CVAT interop contract (L4, Tier B <-> Tier C)

信号アノテーションのCVAT往復(人手検証ラウンド)の契約。ツール:
`export_cvat_signal_task.py`(Tier B → CVATタスクzip)/
`import_cvat_signal_annotations.py`(CVAT XML → Tier B)。
状態語彙は README の正準stateトークン仕様(`{color}-{shape}[-{direction}]`、
ソート・カンマ結合)を全属性で共有する。

確定日: 2026-07-19(タスク1)。旧契約(dataset側
`docs/cvat_signal_interop_spec.md` の select型 `state`)はこの版で置き換え。

## Tier B sidecar: `traffic_signal_2d/v2`

`annotation/traffic_signal_2d_ann.json`。v1からの変更は `attributes` の契約のみ
(テーブル構造は不変)。v1読み込みフォールバック: `raw_state` が無ければ
`detector_signal` を読む。

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
| `visibility` | select | `clear` `partial_occluded` `heavy_occluded` `unknown` / `unknown` | する | 評価時のフィルタ用 |
| `review_status` | select | `unchecked` `accepted` `rejected` `fixed` / `unchecked` | **する** | レビュー状態。評価のGTは `accepted`+`fixed` |
| `map_traffic_light_id` | text | lanelet2 way id / 自動対応付け結果 | する | 誤対応の修正可。空=対応なし |
| `regulatory_element_id` | text | relation idカンマ結合 / 導出値 | しない | 表示専用。importで `map_traffic_light_id` から**常に再導出** |
| `facing` | text | `front` `back` / 導出値(未マッチは空) | しない | 灯器面の向き(linestring方向-90°回転の法線、本地図で実証済み)。`back` は筐体裏面の検出 — colored stateなら誤対応疑い |
| `raw_state` | text | autolabelの元state | しない | 検出器の原文。人手修正後も不変(diff用) |
| `detector_score` | text | 検出スコア / 空(手動box) | しない | provenance |
| `source_type` | select | `manual` `projected_map` `auto` / CVAT新規box=`manual`(先頭)、自動生成分は `auto` 値で出力 | しない | 由来の区別 |
| `annotation_uid` | text | sidecar token | しない | 往復キー。空の新規boxはimportで採番 |

### 設計判断の記録

- **`state` はtext(正準文字列)を維持し、`circle_state`/`arrow_state` のselect分解は不採用。**
  理由: 実データに同時多方向矢印(`green-arrow-up,green-arrow-up_left` 等)や
  ped・amber複合があり、selectでは表現できないか属性爆発する。編集可能な
  分解フィールドを併設すると `state` と二重管理になり乖離する。タイポ対策は
  select ではなく **import時の厳格検証**(下記)で行う。
- **`regulatory_element_id` は導出値**: 1つのTL wayを複数レーンのregulatory
  elementが共有するため人手管理は誤りやすい。importが地図から再計算する。
- CVAT実機検証済みのzipレイアウトを維持: `annotations.xml` +
  `images/{CHANNEL}_{frame}.jpg`(`images/` 直下必須)。

## import時の検証(fail loud)

1. `state` の全トークンが正準文法(`state_tokens.CANON_RE`、旧形式は自動正規化)
   でパースできること。不明トークンは **エラーで列挙して失敗**(黙って落とさない)。
2. `state` はパース結果から再シリアライズして正規化(順序・重複ゆらぎ吸収)。
3. `map_traffic_light_id` が地図に存在するway idであること(空は可)。
4. `regulatory_element_id` は地図から再導出して上書き。
5. `signal_kind` が空なら `state` から導出。
6. box新規(annotation_uid空)は token 採番、`source_type=manual` を既定。

## ワークフロー(レビューラウンド)

```bash
# 1. Tier B生成(match_traffic_lights.py が v2 属性で出力)
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
