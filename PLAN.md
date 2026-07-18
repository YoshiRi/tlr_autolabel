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

### 1. [done 2026-07-19] 2セッション統合の残件
- [x] README: GPU venv 節に model matrix への参照注記で整合(YOLOX=int8 engine / CoMLOps=onnxruntime-gpu)
- [x] レガシー単体スクリプト: docstring に位置づけ明記(`tlr_detector_onnx.py`=DEBUG standalone、
      `tlr_lamp_recognizer_onnx.py`=共有デコードモジュール+デバッグCLI)
- [x] ディレクトリ物理整理: 当面フラット維持と判断(10 py + configs/ + docs/。層の対応は README の
      Architecture 表が担う。分割はファイル数がさらに増えた時に再検討)

### 2. [done 2026-07-19] ラベル成果物の恒久化
- [x] dataset の `build/tlr_autolabel_eval_20260718/` へ退避(full_tiles / full_notiles /
      exports(COCO+CVAT) + 来歴README.txt)。AWML派生は `build/awml_derived/`

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
- [x] 等倍タイルの no-op resize スキップ(2026-07-19: 効果は誤差レベルだった — 前処理コストの主因は canvas 生成+float変換)

## 運用メモ

- 本番ラベリングの標準手順:
  1. `./run_gpu.sh <dataset>/data/<CH> --preset yolox-1920-int8 --t4-dataset <dataset> --out-dir <out> --run-id <id>`(カメラ毎)
  2. `python3 match_traffic_lights.py --dataset-root <dataset>`
  3. `python3 aggregate_regulatory_signals.py --dataset-root <dataset>`
  4. `python3 render_re_timeline.py --dataset-root <dataset>` → `build/tl_match/re_timeline.html`
- このファイルと README、プロジェクトメモリ(`.claude-mine/.../memory/`)が3点セット
