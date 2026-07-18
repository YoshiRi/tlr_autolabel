# TLR autolabel — plan / remaining tasks

作業計画の単一管理ファイル。タスクの追加・完了・方針変更はここを更新する。
設計の契約は README.md 冒頭(5階層 / Tier A-C)。矛盾時は README を正とする。

最終更新: 2026-07-19

## ✅ 完了(日付順)

| 日付 | 内容 |
|---|---|
| 07-17 | CoMLOps-Large デコード仕様のリバースエンジニアリング(アンカー実測フィット → `comlops_large_detector_ml.param.yaml`) |
| 07-17 | fp32 ONNX のブラー誤検出問題を根本解明 → int8 `.engine` 対応(`trt_run.cpp`) |
| 07-18 | タイル推論 `--tiles`(578f 評価: 検出 +20%、デグレ 0) |
| 07-18 | 内部IF凍結: `tlr_autolabel/v1` スキーマ + 正準 state 語彙(README 参照) |
| 07-18 | L2 エクスポータ: `export_labels.py`(COCO / CVAT 1.1) |
| 07-18 | AWML 仕様調査(T4 object_ann 直読・state-as-category)→ アダプタ方式決定 → `export_awml.py` + `configs/state_vocab/db_tlr.yaml` |
| 07-18 | detector プリセット(`configs/detectors/`、`--preset` / `--no-tiles` / `meta.preset`) |
| 07-19 | スタンドアロンリポジトリ化(github.com/YoshiRi/tlr_autolabel, private)。launcher リポジトリは原状復帰済み(force-push 完了) |
| 07-19 | **c1af6a38 データセットの本番再ラベル**: 3カメラ×578f を int8+tiles で v1 再生成・設置(旧 fp32 ラベルは `build/tlr_autolabel_fp32_backup_20260719/`) |
| 07-19 | **Tier B 再生成**: 地図マッチ 12.9%→**77.5%**、unknown 81%→29%、正準語彙化(旧 sidecar は `build/traffic_signal_2d_ann.fp32_backup_20260719.json`) |
| 07-19 | **L3 3工程をリポジトリへ移植**: `match_traffic_lights.py` / `aggregate_regulatory_signals.py` / `render_re_timeline.py` + 共通パーサ `state_tokens.py`。IF修整(sample_data_token でフレーム解決、正準+旧形式両対応、amber↔yellow は地図比較境界のみ)。リグレッション一致(978/758)。dataset 側 `tools/` に残るは CVAT 往復ペア(L4)のみ |
| 07-19 | **8倍高速化**: 非連続配列 tofile(399ms/パス)修正 + yolox デコードのベクトル化。6.4s → 0.78s/frame(出力ビット一致) |
| 07-19 | L1 の新モデル試走対応: 多クラス yolox 一般化、動的shape/未知ファミリの明示エラー、プリセット env var 展開、README に flexibility contract(tiles はオプトイン) |
| 07-19 | **L4 CVAT往復を当リポジトリへ移植 + 契約v2**: 属性契約 `docs/cvat_interop.md`(`traffic_signal_2d/v2`: 正準state text + signal_kind/visibility/review_status + raw_state/detector_score。circle/arrow select分解は不採用=同時多方向矢印・二重管理回避)。importはfail-loud検証(stateトークン文法・way id実在・RE id再導出)。往復ロスレス(CAM_FRONT 299/299)、aggregator v2回帰OK。dataset側の旧ペア・旧specはMOVED注記済み(削除可) |

## 📋 残タスク(実行順)

### 1. 2セッション統合の残件(旧「統合整理」— 大半は自然消化済み)
`run_gpu.sh`+preset+engine の組合せは本番再ラベルで実証済み。残るのは掃除:
- [ ] README の重複整理(「GPU (recommended)」節と detector model matrix の統合)
- [ ] レガシー単体スクリプト(`tlr_detector_onnx.py` / `tlr_lamp_recognizer_onnx.py`)の扱い決定
      (案: `debug/` へ移動 or docstring に位置づけ明記のみ)
- [ ] ディレクトリ物理整理の要否判断(現状フラット。core/exporters/enrich 分割するか)
- 完了条件: README が一本化され、全スクリプトの位置づけが README から一意に分かる

### 2. ラベル成果物の恒久化(小、すぐ終わる)
- [ ] scratchpad(セッション終了で消える)にある評価成果物の退避:
      tiles/no-tiles 比較 JSON 群、COCO/CVAT エクスポート、AWML 派生データセット
      (案: dataset の `build/` 配下 or 専用成果物ディレクトリ)
- 完了条件: セッション非依存の場所に置かれ、場所がここに記録されている

### 2.5 CVAT レビューラウンド1 → autolabel 評価(進行中 — Claude 側トラックで追跡)
- [ ] レビュー用CVATタスクzip生成(3カメラ、v2属性の自動アノテーション同梱)→ CVATへインポート
- [ ] 人手レビュー(state修正 / map id正誤 / review_status付与。RE検証レポートのフラグを優先順位付けに使用)
- [ ] `import_cvat_signal_annotations.py` で取り込み(検証+正規化)
- [ ] 評価: `review_status∈{accepted,fixed}` をGT、`raw_state` を予測として検出/状態/地図IDの一致を集計
- 完了条件: レビュー済みTier Bと評価レポートが dataset `build/` に存在

### 3. [hold] AWML 結合テスト(ユーザー指示により保留中)
- [ ] AWML checkout 上で `create_data_t4dataset.py` を派生データセットに対して実行
- [ ] `mask: null` / `instance_token: null` の t4dev-kit 受容確認
- [ ] down 系矢印・単独矢印の `unknown` フォールバック妥当性の最終確認
- 完了条件: create_data が派生データセットからエラーなく info JSON を生成

### 4. [別枠] L5: ros2_pipeline 検証
- [ ] 動作検証(未実施)
- 完了条件(受け入れ基準): launch 済み int8 パイプラインと同一フレームで結果一致。
  合格まで L1-L4 の依存グラフに入れない

### 5. [外部依存] 小粒
- [x] deepen 形式対応表(2026-07-19: `configs/state_vocab/deepen.yaml` — db_tlr互換を仮定した契約表+v2属性パススルー。ラベルセットの実確認は変換リポジトリ側とのすり合わせ待ち)
- [ ] testM ディレクトリが空(モデル配置待ち → 配置されたらプリセット追加)

### 6. [backlog] 高速化の続き(現状 0.78s/frame で実用十分。必要になったら)
- [ ] フレーム間パイプライン化(JPEG デコード/前処理と GPU の重なり)→ ~0.3s/frame 見込み
- [ ] タイルのバッチ推論(エンジンを batch=5 でリビルド)
- [ ] 等倍タイルの no-op resize スキップ

## 運用メモ

- 本番ラベリングの標準手順:
  1. `./run_gpu.sh <dataset>/data/<CH> --preset yolox-1920-int8 --t4-dataset <dataset> --out-dir <out> --run-id <id>`(カメラ毎)
  2. `python3 match_traffic_lights.py --dataset-root <dataset>`
  3. `python3 aggregate_regulatory_signals.py --dataset-root <dataset>`
  4. `python3 render_re_timeline.py --dataset-root <dataset>` → `build/tl_match/re_timeline.html`
- このファイルと README、プロジェクトメモリ(`.claude-mine/.../memory/`)が3点セット
