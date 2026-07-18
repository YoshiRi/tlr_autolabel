# TLR autolabel — plan / remaining tasks

作業計画の単一管理ファイル。タスクの追加・完了・方針変更はここを更新する。
設計の契約は README.md 冒頭(5階層 / Tier A-C)。ここは「残作業」のみを持つ。

最終更新: 2026-07-19(スタンドアロンリポジトリ化 + タスク2/3完了後)

## 完了済み(参照用サマリ)

- **リポジトリ独立(2026-07-19)**: launcher リポジトリから当リポジトリ
  (github.com/YoshiRi/tlr_autolabel, private)へ抽出。launcher 側の暫定コミットは
  ローカルでは除去済み(origin への force-push はユーザー実行待ち)
- L1: `tlr_autolabel.py` — int8 `.engine` 対応(`trt_run`)、`--tiles`(578f評価: +20%検出/デグレ0)、
  `tlr_autolabel/v1` スキーマ(相対パス主・T4リンク・provenance)、detector プリセット(`configs/detectors/`)、
  `--skip-existing`(バッチ再開)
- L2: `export_labels.py`(COCO / CVAT 1.1)、`export_awml.py`(db_tlr 派生データセット、整合性検証済み)
- 語彙: 正準 state トークン確定、`configs/state_vocab/db_tlr.yaml`(正準→db_tlr)
- 知見: fp32 ONNX のブラーFP問題(int8 engine で解消)、CoMLOps-Large デコード仕様
- **タスク2完了(2026-07-19)**: c1af6a38 データセット3カメラ×578f を
  `yolox-1920-int8`+tiles+GPU classifier で v1 再生成し設置(旧 fp32 ラベルは
  `build/tlr_autolabel_fp32_backup_20260719/`)。`match_traffic_lights.py` に v1
  対応パッチ(state キー優先、旧形式フォールバック)
- **L3取り込み完了(2026-07-19)**: dataset 側の3工程(`match_traffic_lights.py` 地図付与 /
  `aggregate_regulatory_signals.py` 複数カメラ・複数ヘッド融合 / `render_re_timeline.py` 可視化)を
  当リポジトリへ移植。IF修整: v1 は sample_data_token でフレーム解決(ファイル名規約から解放)、
  トークンパーサを `state_tokens.py` に共通化(正準+旧形式両対応、amber→yellow は地図比較境界のみ)。
  リグレッション一致(978/758)、RE融合 461 観測・7ヘッドグループ。
  dataset 側 `tools/` に残るのは CVAT 往復ペア(L4)のみ
- **タスク3完了(2026-07-19)**: Tier B(`traffic_signal_2d_ann.json`)を再生成。
  品質比較 — 旧 fp32: 2017 anns / 地図マッチ 12.9% / unknown 81%(粗い語彙)
  → 新 int8+tiles: **978 anns / 地図マッチ 77.5% / unknown 29%**、正準トークン
  (`green-arrow-up,red-circle` 等の複合 state 付き)。旧 sidecar は
  `build/traffic_signal_2d_ann.fp32_backup_20260719.json`

## 残タスク

優先度順。[hold] は着手保留の明示。

### 1. 2セッション統合整理(blocked: 他方セッションの完了待ち)
- [ ] 他方セッション成果(`run_gpu.sh` venv 路線, `ros2_pipeline/`)との突き合わせ
- [ ] バックエンド使い分けの最終化(YOLOX→int8 engine / CoMLOps→onnxruntime-gpu)— README には整合済み、コード側の導線を確認
- [ ] README の重複・矛盾の解消(両セッションが編集した状態の一本化)
- [ ] レガシー単体スクリプト(`tlr_detector_onnx.py` / `tlr_lamp_recognizer_onnx.py`)の扱い決定
- [ ] ディレクトリ物理整理(configs/ 済み。core/exporters 分割はこのタイミングで判断)

### 2. [done 2026-07-19] dataset 側ツールの旧IF依存解消 → 完了済みサマリ参照

### 3. [done 2026-07-19] Tier B 品質更新 → 完了済みサマリ参照

### 4. [hold] AWML 結合テスト(ユーザー指示によりスキップ中)
- [ ] AWML checkout 上で `create_data_t4dataset.py` を派生データセットに対して実行
- [ ] `mask: null` / `instance_token: null` の t4dev-kit 受容確認
- [ ] down 系矢印・単独矢印の `unknown` フォールバック妥当性の最終確認

### 5. L5: ros2_pipeline 検証(別枠トラック)
- [ ] 動作検証(未実施)。受け入れ基準: launch 済み int8 パイプラインと同一フレームで結果一致
- [ ] 合格まで L1-L4 の依存グラフに入れない

### 6. 小粒・外部依存
- [ ] deepen 形式対応表(Tier C、変換ロジックは他リポジトリ管理)
- [ ] testM ディレクトリが空(モデル配置待ち → プリセット追加)
- [ ] ラベル成果物の恒久化(現在 scratchpad: `full_tiles/` 等。必要なら退避)

## 進め方メモ

- タスク2→3 は連続作業(v1 再生成 → Tier B 再生成)で片付くので、統合整理(1)の際にまとめてやるのが効率的
- 4 は AWML 側の都合が出た時点で再開
- このファイルと README、プロジェクトメモリ(`.claude-mine/.../memory/`)が3点セット。
  文書間で矛盾が出たら README(設計契約)を正とする
