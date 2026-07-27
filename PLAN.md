# TLR autolabel — plan / remaining tasks

作業計画の単一管理ファイル。タスクの追加・完了・方針変更はここを更新する。
設計の契約は README.md 冒頭(処理層 / Tier A-C + B-review)。矛盾時は README を正とする。

最終更新: 2026-07-27

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
| 07-21 | **L4.5 RE timeline review基盤**: CVATのbbox/visibility修正と分離し、信号状態を物理信号グループ(`member_ways`)×時系列区間でレビューする `traffic_signal_re_review/v1` を追加。`render_re_review_timeline.py`(静的HTML編集UI)、`make_re_review_template.py`(ひな形生成)、`apply_re_review.py`(Tier Bへ伝播) + `docs/re_timeline_review.md`。c1af6a38既存autolabelで smoke: 24 RE→8 group、62 segment、2376/2793 annotationへaccepted伝播、再aggregate成功 |
| 07-27 | **B/B' IF再整理**: B' の差分を t4devkit 定義 `annotation/traffic_light.json` の有無に固定。schema は `{token, instance_token, traffic_light_linestring_id}`。RE/group は map から解決。`traffic_light_map_association.json` は temporary/deprecated と明記し、新規consumer禁止 |

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
- [x] `--max-distance` 既定 150→**200m**、`--max-incidence-deg` 既定 75→**85°**、`--map-fill-max-distance` 既定 50→70→**130m**(中央奥の正面信号way2180が59mで検出漏れ→ギャップ36フレーム>max_gap→50m超でmap-fill外、の三重で3.6秒未被覆だった。裏付け付きなら遠方でも投影は信号上に正確(123mの次交差点信号で実証)。既定130m=map-presence 722箱。評価は距離ビンで区切る)(77°の横向き実信号が除外され「明らかにあるのに補完されない」原因だった。85°で復活: マッチ81.4→84.1%、評価側は facing_deg で別途フィルタ可)
- [x] `--map-fill`(既定on、@50m・front・時間裏付け必須): 検出が完全に漏れた近距離信号を地図投影でunknown埋め。
      裏付けなしのblind版は空/高架に偽箱を出すと判明→`--map-fill-window`で同一信号の近傍検出を必須化。画面外箱はクリップ/除外
- [x] 検証動画 `make_review_video.py`(緑=検出/シアン[I]=補間/マゼンタ[M]=地図埋め/橙=未マッチ)をリポジトリ追加

### 2.7 RE timeline review layer(2026-07-21)
位置づけ: CVATはbbox/visibility/reject/map idの編集に集中させ、灯火状態は
物理信号グループ×時系列区間で一括レビューする。CVAT attribute `state` は
局所修正・互換・diff用に残すが、大量状態修正の主UIにはしない。
- [x] `traffic_signal_re_review/v1` 契約を `docs/re_timeline_review.md` に固定
- [x] `make_re_review_template.py`: `traffic_signal_re/v1` から review JSON ひな形生成
- [x] `render_re_review_timeline.py`: 静的HTMLでsegment選択→代表crop候補切替→accepted/fixed/rejected→JSON export
- [x] `apply_re_review.py`: review JSON をCVAT import済みTier Bに重ね、`state`/`signal_kind`/`review_status`だけ伝播
- [x] c1af6a38既存成果物でsmoke test(推論再走なし): accepted仮template 62 decisions → 2376 annotations更新、overlap 0、再aggregate成功
- [ ] 実レビュー運用: CVATでbbox/visibilityを直した後、RE timeline reviewでstateを確定し、L6 GT指標を初回算出

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
- [x] CVATでのGT作成を楽にする仕組みの初期解: CVATプラグインに寄せず、L4.5 RE timeline review sidecar/UIを追加。CVATはbbox/visibility、timelineはstate区間、`apply_re_review.py`で統合
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

### 2.7 暫定GT運用への移行(2026-07-21 ユーザー決定)
- [x] 現状ベスト出力を凍結: `build/gt_provisional_20260721/`(Tier B + RE + match_report + MANIFEST)
- 位置づけ: **固定ベースライン**(evaluate_signals --baseline でのrun/モデル比較用)。人手未レビュー
  (review_status=unchecked)なので「自己整合の基準」であって精度GTではない。
- 注意: 予測=autolabel自身を暫定GT=autolabel自身と比べると自明に一致 → 用途は「別設定/別モデルとの差分」。
- 時系列・共有ヘッドのstateラベリングはCVAT不向き → ユーザーがCodexで補助ツール制作中(L4.5相当)。
  完成したらそのツールとのIF(Tier B / RE timeseries)を整合させる。

### 2.8 モデル/設定比較(2026-07-22、暫定GTベースで初回)
- [x] `compare_runs.py`: 複数label dirを同一地図でマッチ→距離ビン別 candidate_coverage
      (GTフリー検出recall proxy)+未マッチ(FP的)+unknown率を並置
- [x] CAM_FRONT比較(int8, tiles): **驚きの結果 — S960 > L1920**。
      coverage overall S960 0.65 / L1920+tiles 0.48 / L1920 notiles 0.42。
      30-100mでS960が大差(60-100m: 0.95 vs 0.75)。未マッチ率はS960の方が低い(19% vs 22%)。
      目視検証: S960の追加検出は実在の遠方信号(00565で次交差点の信号2基、L1920は0検出)。
      ただしstate=unknown(遠方で色分類不可)なので検出recall向上であって状態精度ではない。
- [x] 2x2 ablation(S960_notiles追加): モデル効果(+0.137)>>タイル効果(+0.06〜0.09)。
      **S960_notiles 0.556 > L1920_tiles 0.476** → 勝因はモデル、タイルではない。
      100-150mはS960+tilesのみ到達(0.375)。結論: 検出器をS960に、tilesは遠方用の上乗せ。
      最終確定は人手GT後(遠方追加分はstate=unknown)
### 2.9 モデル比較の一般化検証(CAM_FRONT_FAR、2026-07-22)
- [x] 望遠カメラ CAM_FRONT_FAR(fx=5274、信号~4倍大)で同じ4configを比較
- [x] **S960>L1920の差は消失** — 4config横並び(coverage 0.525〜0.558)。
      30-100mは全config 1.0(信号が大きく全部検出)、差は≥150mのみで僅少。
      → S960優位は「広角・小信号」固有。カメラの信号サイズで最適構成が変わる。
- [x] 副次発見: 望遠では**タイルがFP増**(L1920_tiles未マッチ249=41%)。
      **S960_notilesが最クリーン**(マッチ率0.75/未マッチ101)。望遠はタイル不要。
- 結論: 検出器を1つに固定せず、カメラ特性で使い分け(望遠=S960 notiles、広角=S960 tiles)。
  ユーザーの「データ固有では?」という懐疑が正しかった。GTフリー・状態精度は別。
### 2.10 検出閾値の見直し(launchとの差の解明、2026-07-22)
- [x] 「同じengineなのにlaunchの方が検出する」の原因: オフラインが score 0.5(launch 0.35)+map ROI prior無し。
- [x] 閾値スイープ(L1920+tiles CAM_FRONT): 0.5→0.35 で実信号マッチ+30/FP+6のみ、coverage 0.476→0.540。
      0.2で0.600(S960に肉薄=S960優位の一部も閾値差だった)。中距離(30-60m)が最も回復。
- [x] **既定を --det-score-thr 0.5→0.35 に変更**。根拠だった「map prior無し」は誤り(L3で地図マッチ=FPフィルタ)。
      低閾値のFPは「未マッチ検出」として仕分けられGT汚染しない。
- [ ] 暫定GT(0.5生成)は据え置き。次回全カメラ再ラベル時に0.35で更新
### 2.11 実ノード評価の設計(2026-07-22)
- [x] 設計文書 `docs/eval_design.md`: 実ROSノードを3階層で評価
      (①ROI検出 ②分類 ③マッチ+Merge後のRE)、GTは本repoのTier B+RE timeseries。
- 原理: ノード出力を正準形式(tlr_autolabel/v1 + RE)に変換 → オフラインと同じ
  L3/L6エンジンでGTと突合(評価器はソース非依存)。ros2_pipeline collector が既に
  tlr_autolabel/v1 を吐くので流用可。
- 新規要素: 2ソース評価器 `eval_vs_gt.py`(pred run × GT run → 検出P/R・状態精度・RE精度)、
  GT-ROI feeder(分類器単体評価=検出recallと分離)、production graph harness(RE用、map+localization要)。
- フェーズ: ①検出器駆動harnessで検出+分類(暫定GTで機構検証、既存parity_check延長)→
  ②分類器単体(GT-ROI投入)→ ③full graphでRE時系列(実RE GT=時系列ツール完成後)。
- [x] `eval_vs_gt.py` 実装・検証(2ソース評価器: 検出P/R/IoU距離別 + 状態精度 + 混同行列。
      機構テスト S960 vs L1920 で動作確認。未レビューGTは非rejected扱い+警告)
- [x] 方針確定(2026-07-22): ノードを走らせた**結果入りrosbagを受領**して評価(ユーザーが別窓で生成)。
      ③RE評価はtier4 driving_log_replayer_v2が既存対応 → 我々は①②(検出+分類)を担当
- [x] `bag_to_labels.py`(rosbag→tlr_autolabel/v1、roi+分類をtl_idでjoin、要素enum→正準トークン検証済み、
      ±75msでGTキーframe整合)。要 source ROS2 humble
- [ ] ユーザーからbag受領後: bag_to_labels → match → eval_vs_gt をend-to-end実行
### 2.12 標準t4形式への集約(A→B/B' コア変換、2026-07-23開始 / 2026-07-27 IF更新)
- 合意: 標準 t4dataset annotation を固いIFとし、AWML/Deepen/CVAT/COCO/DLR変換は既存t4devkit/webautoに委譲。
  自作 export_awml/export_labels(COCO)/deepen表 は廃止方向。
- B契約: `object_ann.json` + `category.json` + `attribute.json` + `instance.json`。
  `instance.json` は2D画像上の信号Instance識別だけを持ち、TLR ID / lanelet2 ID を入れない。
- B'契約: B + `annotation/traffic_light.json`。BとB'の差分は**このファイルの有無だけ**で判定する。
  schemaは t4devkit 定義:
  `{"token": "...", "instance_token": "...", "traffic_light_linestring_id": "501"}`。
  必要なRE/group relationは全てmapにあり、`traffic_light_linestring_id` から解決する。
- temporary/deprecated: `traffic_light_map_association.json` は旧 `object_ann_token -> map_traffic_light_id`
  互換ファイル。現行IFに乗らないJSONなのでB/B'判定に使わず、新規consumerを追加しない。
- [x] `to_object_ann.py` 実装・検証: Tier A(autolabel-dir)/Tier A'(sidecar)両対応 → 標準object_ann+category+attribute+instance。
      現行実装は map relation がある場合だけ `traffic_light.json` を出力し、無ければ未生成/削除してTier Bにする。
      移行互換として deprecated `traffic_light_map_association.json` は `--write-deprecated-map-association` 明示時だけ一時出力。
- [ ] t4devkit側: `traffic_light.json` table/schema/validation を定義し、B/B'判定を `traffic_light.json` 有無に統一する。
- [x] RE-based annotation adapter: `export_re_to_t4dataset.py` を追加。`traffic_signal_re_review/v1`
      または `traffic_signal_re/v1` をA' geometry sidecarへ適用し、`to_object_ann.py` で標準t4dataset B/B'へ変換。
      unchecked templateをannotationとして昇格する場合は `--unchecked-as accepted` を明示。中間A'はtemporaryでdefault削除。
- [x] object_ann-based annotation adapter: `export_object_ann_to_t4dataset.py` を追加。既存
      `object_ann.json` を保持し、deprecated `traffic_light_map_association.json` を
      t4devkit定義 `traffic_light.json` へ正規化。複数linestringにまたがる2D instanceは
      deterministicに分割し、`instance_name` へmap/TLR IDを入れない。
- [ ] DLR adapter側: `traffic_light.json` + map から TLR ID ベースへ変換する。deprecated association は読まない。
- [x] `to_object_ann.py` 追加検証: 1つの `instance_token` が複数 `traffic_light_linestring_id` に張られないよう
      tracking時に異なるmap idの結合を禁止し、出力前にもfail-loud検証を追加(現IFでは通常1 instance = 1 map linestringを期待)。
- [x] L2 IF smoke(2026-07-27): `/tmp/tlr_l2_if_test.czPyGT` に cb7fd5c0 dataset の軽量コピー(annotation実体、
      data/input_bag/mapはsymlink)を作成して `to_object_ann.py --sidecar` 実行。
      B': 1766 object_ann / 228 instance / 110 traffic_light relation、schema key一致、instance/map参照欠落0、
      deprecated associationはdefault非出力。B: map id除去sidecarで `traffic_light.json` 未生成。
      同一out再利用でもB'→B再変換で stale `traffic_light.json` 削除確認。
- [x] RE→t4dataset smoke(2026-07-27): c1af6a38 dataset を入力に `/tmp/tlr_re_to_t4_c1af.C6xTDA` へ派生出力。
      `annotation/traffic_signal_re_review.template.json --unchecked-as accepted` と
      `annotation/traffic_signal_re_timeseries.json` の両経路が成功。出力は 2906 object_ann / 1346 instance
      (3D 284保持 + TLR 1062) / 839 traffic_light relation、schema key一致、instance/map参照欠落0、
      ambiguous instance 0、deprecated association非生成。data/input_bag/map/tlr_autolabel はsymlink参照。
- [x] 2パターン出力 smoke(2026-07-27): `/tmp/tlr_two_patterns_c1af.hX9aJb` に
      `01_re_object_ann_base`(既存object_ann正規化: 2906 object_ann / 1344 instance /
      837 traffic_light、旧ambiguous 6 instanceを101行だけsplit) と
      `02_re_plus_autolabel`(RE review templateをaccepted昇格してA'へ適用: 2906 object_ann /
      1346 instance / 839 traffic_light) を生成。両方ともschema key一致、instance/map参照欠落0、
      ambiguous instance 0、deprecated association非生成。
- [x] 実GT初測定(ad266d7c 人手object_ann 518箱 vs L1): 検出 P=0.58 R=0.85、状態精度0.75、最大誤り=分類器のX→unknown
- [x] export_awml/export_labels を `deprecated/` へ移動(実行時警告+deprecated/README、参照ゼロ確認)。
      deepen.yaml=reference-only注記、doc参照をto_object_annに修正。README/STATUSのレイヤ改訂も反映
- [x] unknown誤りの深掘り(実GT ad266d7c): 105件を分解 = 75件は極小信号(中央値10px、分類器が灯火を読めず=解像度限界)、
      29件は down系矢印(db_tlrに無く unknown化)。down_left→left/down_right→right をdb_tlrに追加 → 状態精度 0.751→0.780。
      残る誤りは①極小信号のX→unknown(green44/red20=解像度限界、系統的)②分類器の矢印左右取り違え(red→red_left 9)。
      示唆: 解像度限界はNEAR/FAR多カメラ融合の設計理由そのもの。矢印方向は分類器のモデル課題
- [x] object_ann変換の完全化(2026-07-23): instance(カメラ毎IoUトラッキング)+instance.json、
      mask(box矩形RLE、[W,H]自己整合・復号でbox一致確認)。標準キー厳密一致、参照整合dangling 0。
      ad266d7c: 787箱/703 instance(59が複数frame)。mask RLE variantのt4devkit一致のみ未検証(1関数で調整可)
### 2.13 c1af6a38 のobject_ann更新(FAR含む、3D保持、2026-07-23)
- [x] to_object_ann.py を**マージ方式**に修正: 既存の 3D テーブル(instance/attribute/category)を
      保持して TLR 2D を追記(以前は上書きで 3D を壊す危険があった)。`--in-place`(自動バックアップ付き)追加。
- [x] c1af6a38 in-place 更新: object_ann 空→**3323箱(FAR 530含む)**、instance 3D 284 + TLR 1256 = 1540、
      legacy map assoc 2854(当時IF)。sample_annotation(9348・3D箱)無傷・dangling参照0。backup: build/annotation_backup_20260723_163229
      現行IFで再生成する場合は `traffic_light.json` relation数を記録する。
- [x] t4devkit表示バグ修正(2026-07-23、実データで表示確認): 動作参照(JapanTaxi5 odaiba)と比較し2点差分特定 —
      ①mask: 参照は全レコード同一の空placeholder`UFhfUzU='`でt4devkitはbbox描画、我々の実RLEはデコード不能で非表示
      →placeholder mask既定化(`--real-masks`で実マスク選択可)。②属性名`truncation_state`→`Truncation_State`(参照に厳密一致)。
      再実行semantics修正: object_annは2D TLRとして毎回置換、instanceはsample_annotation参照の3Dのみ保持(dedup罠回避)
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
