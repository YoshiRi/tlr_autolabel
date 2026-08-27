# STATUS — capabilities, goals, vision

「今これは頼っていいか?」に答えるダッシュボード。マイルストーン時に見直す
(毎コミットではない)。役割分担: **能力・ゴール・成熟度はここだけ**、タスク
(動詞)は PLAN、契約・IF・層は README、横断事実はプロジェクトメモリ。

最終レビュー: 2026-08-27

## North Star

> **信号認識(TLR)モデルの評価と、その評価データの効率的な生成**。
> 走行データからレビュー可能・地図連携済みのアノテーションを自動生成し、
> 人手補正で評価用 GT に仕上げ、モデルを評価・比較する。
>
> 学習そのものは基本スコープ外。学習データ生成は「既存アノテーション
> (AWML / T4)との**互換性を保つための手段**」として持つのであって、
> 学習を回すことが目的ではない。

目的が広がったら: この段落を広げる(稀)か、下の Goals に行を1つ足す(通常)。

## 成熟度の定義

| level | 意味 |
|---|---|
| `experimental` | 動くが未検証 / 前提が揃っていない |
| `works-here` | このマシン・このデータセットで検証済み |
| `reproducible` | 他環境でも再現できる(パス/依存が抽象化済み) |
| `productized` | 無人で回せる |

## Goals(目的の分解 — ここが増える単位)

各ゴールは複数の層(README の L1-L5)を横断しうる。層=How、ゴール=Why。

主目的は **G1(評価)** と **G2(評価データ生成)**。G3-G5 はそれを支える手段。

| | ゴール | 位置づけ | 成熟度 | 前進させる作業 |
|---|---|---|---|---|
| **G1** | **モデル評価・比較**(指標・GT基準の精度/PR・run間比較) | 主目的 | experimental | PLAN 2.5(GT 供給) |
| **G2** | **評価データの効率生成**(自動ラベル→地図付与→人手レビューで GT 化) | 主目的 | works-here | PLAN 2.5 |
| **G3** | 新モデル立ち上げ(試走ベンチ)— 評価対象を素早く投入 | 手段 | reproducible | — |
| **G4** | 既存アノテーション互換(AWML/T4/CVAT/deepen 相互変換) | 手段(互換制約) | works-here | PLAN 3(実 AWML 検証・優先度低) |
| **G5** | 再現性・運用(パス/依存/来歴/一括実行/モデル管理) | 手段 | partial | PLAN 15(CI)→ 9 |
| **G6** | ライブ検証(ros2 パリティ) | 手段(別枠) | not-started | PLAN 4 |

## Capabilities(具体的に動くもの)

成熟度 × 所属ゴール。○=できる、△=限定/未検証、✗=未着手。

| 能力 | ゴール | 成熟度 | 状態・注意 |
|---|---|---|---|
| L1 検出+分類(YOLOX int8 engine / CoMLOps ONNX) | G2 | reproducible | ○ ~0.8s/frame(GPU)。engine は非移植(G5) |
| L1 low raw candidates | G2 | experimental | △ `--det-low-score-thr`で`raw_detections`を出力。lowは既定未分類、`--classify-low-detections`で明示分類 |
| L1 タイル推論(遠方小信号) | G2 | works-here | ○ +20%検出/デグレ0(c1af6a38) |
| L1 新モデル試走(素の --detector) | G3 | reproducible | ○ flexibility contract。多クラス/動的shape は明示エラー |
| L1 入力の抽象化(images/video/rosbag2/T4 dataset) | G3/G2 | works-here | ○ `frames/` frame source。video/bag は先に1回展開して全構成が同一ピクセルを見る。実bagでの試走は未(△) |
| L1 モデルIF plug-in(detector + classifier registry) | G3 | reproducible | ○ `detector_type`/`classifier_type`。新familyは1モジュール+registry登録。`--classifier none`で検出器のみ |
| 構成×N の一括推論(`scripts/run_compare.py`) | G1/G3 | works-here | ○ matrix YAML。出力は素のTier Aディレクトリなので下流はそのまま使える。実モデル(960 ONNX)+ 実 `.engine` で試走済(RTX 3060/TRT10.8、engine構成3本の逐次実行で6GB OOMなし・orphan `trt_run` 0)。実bagのみ未(△) |
| L6 比較(GTも地図も無し: 一致/不一致箇所/安定性/timing/並置グリッド) | G1 | works-here | ○ `scripts/compare_naive.py`。**差分の所在の可視化であって順位付けではない**(順位は地図参照 compare_runs.py かGT評価) |
| L3 地図マッチ(lanelet2 投影+Hungarian) | G2 | works-here | ○ 84.7%マッチ(6カメラ) |
| L3 時系列tracking(High/Low association + TTL) | G2 | experimental | △ 明示`--temporal-tracking`時のみON。low候補は既存track更新のみ、短欠落は`propagated`。Tier B converter defaultでは`tracked`/`propagated`を除外し、review aid扱い。現在投影bbox優先、無い場合は前回投影/前回bboxでfallback。同じ(channel,map way)はTTL後再観測でもtrack_id再利用。`TemporalAssociator.update -> TrackingResult`に集約。設定は`configs/tracking/bytetrack-lite.yaml`。合成T4 integration + cb7fd5c0 full smoke pass |
| L3 融合+自動補正(フリップ修復/方向スナップ/未マッチ分類) | G2 | works-here | ○ |
| L3 タイムライン可視化(レビュー優先度付き) | G2 | works-here | ○ build/tl_match/re_timeline.html |
| L4 CVAT 往復(人手レビュー→GT化) | G1/G2 | works-here | ○ ロスレス 299/299。bbox/visibility/reject/map id修正の主経路 |
| L4.5 RE timeline review(状態区間 + カメラ別visibility + 単一フレームROI→A' sidecar伝播) | G1/G2 | works-here | ○ 代表crop候補付きHTML + ROIキャンバス編集。draft/commit 分離(auto-saveはcommit済ファイルに触れない)+ commit前per-entry diff + スキーマ検証。3種の decision は独立適用。c1af6a38 smoke: 24 RE→8 group、62 segment、2376 annotation更新、再aggregate成功。**実GTレビューは未**(道具側の穴は埋まった) |
| L4.5 レビュー説明ビュー2面(per-frame / 俯瞰map) | G2 | works-here | ○ timeline は地図マッチ済グループ単位なので届かない箱がある(cb7fd5c0: 1642中742=**45%**、うち482が歩行者信号)。frame view=全体オーバーレイ+拡大crop+`unmatched_reason`、map view=ego経路+車線形状+Google Maps/Street Viewリンク(MGRS→WGS84 自作、pyproj と 0.088mm 一致・pyprojは非依存)。三面は現フレーム維持で相互リンク |
| L4.5 単一ランチャー(`re_review_all`) | G2/G5 | works-here | ○ 三面生成 + `:8765` serve + 整合チェック + commit毎にビュー再生成(draft auto-saveでは再生成しない=ビューは確定結果のみ)。個別エントリポイントも維持 |
| L4 地図/画像 整合チェック | G2 | works-here | ○ `review/re_map_consistency.py`。matcher判定と**独立**に再投影→`paired`/`map_only`/`image_only`→`unmapped_signal`/`signal_never_observed`/`low_observation_rate`。run全体で集約 + `unknown`除外 + 正対フレームのみ計上(誤警報対策)。`--fail-on-finding`。cb7fd5c0 で地図欠落281件と matcher 過剰却下(way 3595)を分離。**findings への対処は未着手** |
| L2 COCO / CVAT 出力 | G4 | works-here | ○ |
| L2 AWML 派生データセット | G4 | experimental | △ 互換手段。実 create_data で未検証(PLAN 3, 優先度低) |
| L4 deepen 変換 | G4 | experimental | △ 契約表のみ・変換は他リポジトリ・未検証 |
| L6 評価(GTフリー指標: 距離別プロファイル/時間安定性) | G1 | experimental | △ `scripts/evaluate_signals.py`。`eval_detections.jsonl`はtracking source列(`source_type`/`temporal_source`/`track_id`)も保持 |
| L6 評価(GT指標: 精度/PR、confusion) | G1 | **works-here** | ○ 実GT(ad266d7c 人手object_ann 518箱)でL1初測定: 検出P0.58/R0.85、状態精度0.78。`scripts/eval_vs_gt.py` |
| A→B 標準t4変換(object_ann) | G4→core | works-here | ○ `scripts/to_object_ann.py`。AWML/COCO/Deepen/CVATは既存ツール委譲。自作exporterはdeprecated |
| L5 ros2 パリティ検証 | G6 | not-started | ✗ 隔離中。受入=launch int8 と一致 |
| 一括実行(`scripts/run_dataset.py`) | G2/G5 | works-here | ○ 複数データセット・チャンネル自動発見 |
| パス/依存の抽象化(TLR_MODEL_ROOT, requirements) | G5 | reproducible | ○ |
| モデル管理(hash検証/engineキャッシュ/取得) | G5 | partial | △ **hash来歴は完了**(sha256をTier A `meta`へ記録 + `configs/models.yaml` 既知good照合 → vendored classifier と公開ストアのバイト乖離を検知可)。engineキャッシュ・自動取得・ランタイム来歴(TRT版/GPU名/git commit)は未(PLAN 9) |
| CI | G5 | not-started | ✗ `.github/workflows/` 無し。PRにチェックが付かない。テストは417件・約7秒・GPU不要なので費用対効果は高い(PLAN 15) |

## 今できないこと(能力の穴 — 主目的からの優先度順)

1. **レビューループを通した GT がまだ無い(最大の穴)** — 対象データセット
   (c1af6a38 / cb7fd5c0)で人が `accepted/fixed/rejected` を付けたラベルは0件。
   つまり「効率的な GT 生成 → 評価」の一周(G2→G1)が閉じていない(PLAN 2.5/2.7)。
   *区別すべき点*: L6 の GT ブロック自体は動く — 別データセットの**既存**人手
   object_ann(ad266d7c 518箱)を借りて L1 を初測定済み(検出 P0.58/R0.85、
   状態精度 0.78)。欠けているのは道具でも指標でもなく、**このループで作った GT**。
   道具側の穴は PLAN 13/14 で埋まった(三面ビュー + 整合チェックが1コマンド)ので、
   残っているのは実施そのもの。
2. **run間比較の「どちらが良いか」がまだGTに載っていない** — 構成間の差分の所在は
   `compare_naive.py`(GTフリー)/`compare_runs.py`(地図参照)で出せるようになったが、
   優劣を GT 上で示していない(G1)。1 が塞いでいる。
3. **整合チェックの findings が未対処** — 測れるようになった(PLAN 14)が動かしていない:
   ①cb7fd5c0 の地図欠落 281件(読める検出のそばに地図信号が投影されない、最近傍 way
   まで中央値 328px)②way 3595 の matcher 過剰却下(投影 94% vs matcher 65%)。
   ①は地図側、②は matcher 側の課題として既に分離できている(G2)。
4. **複数データセットのスケール未実証** — 1データセットで1回動いただけ(G2/G5)。
5. **CI が無い** — `.github/workflows/` が存在せず PR にチェックが付かない。
   417件・約7秒・GPU不要なので、空いている理由が無い穴(G5, PLAN 15)。
6. **`.engine` の移植性とランタイム来歴** — モデルの内容同一性は sha256 で管理下に
   入った(PLAN 9 前半)が、engine キャッシュ/自動ビルドと、評価結果に紐づく
   ランタイム来歴(TRT版・GPU名・git commit)は未(G5)。

互換系(AWML 実学習検証・deepen)は**手段**なので、上記が片付くまで優先度を下げる。

## 直近マイルストーン(主目的を前進させる的)

- **【最優先】G2→G1 の接続**: c1af6a38 のフラグ上位フレームだけでも人手レビューする。
  `re_review_all` で三面を立ち上げ、CVATでbbox/reject/map idを直し、timeline で
  state区間・カメラ別visibility・単一フレームROIを確定し、L6 の GT 指標を初回算出する。
  これで主目的「評価」が experimental→works-here に上がり、同時に「効率的な GT 生成」も
  実証される。**道具は揃ったので、次のマイルストーンは道具作りではなく運用**。
- **G1 run間比較**: 同じ GT 上で fp32 vs int8、tiles有無の精度差を数値化
  (構成×N の実行と GTフリー比較は PLAN 12 で通した)。
- **G2 地図品質**: 整合チェックの findings 上位を Street View で実在確認し、
  地図修正が必要な分と matcher 側で直す分に振り分ける。
- **G5**: CI(安い・すぐ効く)→ engine キャッシュとランタイム来歴。
