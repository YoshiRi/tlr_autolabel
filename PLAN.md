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

### 2.6 未マッチ検出のRE情報保持 + ギャップ補間 + 地図プレゼンス埋め(フィードバック#2/#3、2026-07-20)
- [x] 未マッチ検出に `unmatched_reason`+`map_candidate_id`+`regulatory_element_id_candidate` を保持(RE情報を捨てない)
- [x] `--fill-gaps` 地図投影ギャップ補間(既定on、境界付き)。CVAT export/import・aggregate に反映、往復対応
- [x] `--fill-mode` strict/bracketed(既定)/all — オセロ式(前後同一→その値)/異なる→unknown/片側→unknown
- [x] `--max-distance` 既定 150→**200m**、`--max-incidence-deg` 既定 75→**85°**(77°の横向き実信号が除外され「明らかにあるのに補完されない」原因だった。85°で復活: マッチ81.4→84.1%、評価側は facing_deg で別途フィルタ可)
- [x] `--map-fill`(既定on、@50m・front・時間裏付け必須): 検出が完全に漏れた近距離信号を地図投影でunknown埋め。
      裏付けなしのblind版は空/高架に偽箱を出すと判明→`--map-fill-window`で同一信号の近傍検出を必須化。画面外箱はクリップ/除外
- [x] 検証動画 `make_review_video.py`(緑=検出/シアン[I]=補間/マゼンタ[M]=地図埋め/橙=未マッチ)をリポジトリ追加

### 2.5 CVAT レビューラウンド1 → autolabel 評価(Claude 側トラックで追跡)
位置づけ確定(2026-07-19、ユーザー決定): GTは常に人力作成。評価層(L6)は
メトリクス算出基盤 — GT不要指標を常時出し、GT依存指標はレビュー済み
annotationの存在で自動有効化。
- [x] レビュー用CVATタスクzip生成(3カメラ×578f、v2属性同梱、dataset `build/cvat_signal/` 計2.3GB + `REVIEW_GUIDE.md`=フラグ付き143フレームの優先リスト)
- [x] **L6 評価層実装**(2026-07-19: `evaluate_signals.py` → `tlr_eval/v1`。距離ビン別の地図候補カバレッジ/IoU/unknown率、時系列安定性(flip率等)、`--baseline` でrun間比較、GTブロックは review_status 待ち。初回実行: fp32→int8 で unknown 81%→29%、single_frame_flip 2件、100-150mでカバレッジ7%に急落を定量化)
- [x] **後解析向けデータ設計 → `tlr_eval/v2`**(2026-07-19): 集計済みサマリでなく
      3つのtidyレジャー(`eval_detections`/`eval_candidates`/`eval_lamps`.jsonl、
      1行=1観測、全次元を列)を中間生成物として出力し、集計は汎用 `pivot()` の
      ビューに。歩行者/車両・灯火別(color×shape・矢印方向)・facing・channel・距離を
      任意group-byで切れる。スキーマ `docs/eval_records.md`。GT到着で正答率も同次元でスライス可。
      知見: 歩行者ヘッドはmatch 97%だがIoU中央値~0.12(`red_green` way形状と位置は合うが範囲不一致→ped専用投影チェック候補)
- [ ] [hold] 人手レビュー実施(CVAT環境が手元にない間は保留。zip・ガイドは生成済み)
- [ ] [hold] CVATでのGT作成を楽にする仕組み(プラグイン/自動アノテーションAPI等)の要否 — CVATプラグイン知識がなく判断保留。判断材料が必要になったら CVAT serverless(nuclio)/SDK の調査から
- [x] 地図候補の妥当性フィルタ(2026-07-19、カバレッジ過小疑いへの対応):
      ①エッジオン除外 — 入射角(符号なし法線vs視線)>75°を候補から除外(実測: 70-80°でmatched率12%、80-90°で6%に崩落)。`--max-incidence-deg`
      ②投影サイズ<8px(検出器min-box)は評価で「too_small」に分離しカバレッジ分母から除外
      ③**facing符号は解決済み(2026-07-19訂正)**: 当初「復元不能」と誤結論したが、
      colored state(灯火が読めた)マッチに限定した検証で **linestring方向を-90°回転した
      法線が正面**(coloredマッチの99%が同側)と実証。逆側のマッチは筐体裏面の
      `unknown` 検出だった(=「点順序不定」に見えた原因)。実装: 候補を front/back に
      分類(edge-onは除外)、Tier B に `facing` 属性、backでcolored stateなら
      `colored_state_on_back_face` フラグ+重み降格(誤対応検出器として機能、38件)。
      カバレッジは front 検出可能候補のみで算出 — 0-60m 58-61% / 60-100m 44% /
      100-150m 14%(残余因 = 遮蔽・画端・真の検出漏れ)
- [ ] レビュー取り込み後: `evaluate_signals.py` 再実行でGT指標(距離別正答率・要素P/R・FP/FN)を確認
- 完了条件: レビュー済みTier Bと評価レポート(GTブロック有効)が dataset `build/` に存在

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

### 7. L3議論の決定事項の実装(2026-07-19 実装完了、全数値は c1af6a38)
- [x] 単発フリップ自動修復(12件、`state_original`保持)
- [x] 矢印方向の地図スナップ(30件: up_right→up 18 + up_left→up 12。不一致 36→24 に減少)
- [x] 未マッチ分類(`unmatched_reason`: candidate_taken 142 / beyond_gate 53 / 背面23 / 幾何2)
- [x] ~~補間しないポリシー~~ → **境界付きギャップ補間に方針転換**(2026-07-20 ユーザー要望): 同一REの短い検出欠落(≤--max-gap-frames、前後同一state)を地図投影で埋める `--fill-gaps`(既定on)。c1af6a38で+191補間box、全てRE付き `source_type=interpolated`
- [x] `review_priority` によるレビュー優先順位(re_verification_report)
- [x] `run_dataset.py` オーケストレータ(複数データセット、L1は--skip-existing再開)
- [ ] 議論持ち越し: candidate_taken 142 の深掘り(1つの地図TLに複数検出が競合 — 地図側のヘッド数不足か、検出重複か)

### 8. [done 2026-07-19] L1 ポータビリティ(パス規約 + 依存 pin)
- [x] `${TLR_MODEL_ROOT}` 規約(env → ~/autoware_data → /opt/autoware/mlmodels)、
      プリセットの絶対パス撲滅、`--detector` も展開、`meta.model_root` 記録、欠落時に明示エラー
- [x] `requirements.txt`(CPU/GPU 2プロファイル、バージョン pin)、`setup_gpu_venv.sh`
- [x] `run_gpu.sh` の venv を `$TLR_GPU_VENV` 化、`trt_run` ビルドを `$CUDA_HOME` 化

### 9. [backlog] モデル管理(再現性の深掘り — 必要になるまで保留)
「.engine の移植不可」と「モデルの内容同一性」を根治する塊。1データセット・1マシン
運用では投資対効果が低いため後回し(2026-07-19 判断):
- [ ] モデルの sha256 をプリセットに記録・ロード時検証、`meta` に detector/classifier hash
- [ ] `.engine` のローカルキャッシュ化 + 自動ビルド(キー: model hash × GPU名 × TRT版)、
      fp32 フォールバック時は品質警告(yolox ブラーFP問題)
- [ ] プリセットに `source_url`(model zoo / webauto)を持たせ自動取得
- [ ] `meta` にランタイム来歴(onnxruntime/TRT版、GPU名、ツールの git commit)
- [ ] Dockerfile(CUDA/TRT pin)で最終保証

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
  5. `python3 evaluate_signals.py --dataset-root <dataset> [--baseline <old_sidecar>]` → `build/tl_match/eval_report.{json,md}`
- このファイルと README、プロジェクトメモリ(`.claude-mine/.../memory/`)が3点セット
