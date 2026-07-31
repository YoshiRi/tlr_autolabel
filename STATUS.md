# STATUS — capabilities, goals, vision

「今これは頼っていいか?」に答えるダッシュボード。マイルストーン時に見直す
(毎コミットではない)。役割分担: **能力・ゴール・成熟度はここだけ**、タスク
(動詞)は PLAN、契約・IF・層は README、横断事実はプロジェクトメモリ。

最終レビュー: 2026-07-23

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
| **G5** | 再現性・運用(パス/依存/来歴/一括実行/モデル管理) | 手段 | partial | PLAN 9 |
| **G6** | ライブ検証(ros2 パリティ) | 手段(別枠) | not-started | PLAN 4 |

## Capabilities(具体的に動くもの)

成熟度 × 所属ゴール。○=できる、△=限定/未検証、✗=未着手。

| 能力 | ゴール | 成熟度 | 状態・注意 |
|---|---|---|---|
| L1 検出+分類(YOLOX int8 engine / CoMLOps ONNX) | G2 | reproducible | ○ ~0.8s/frame(GPU)。engine は非移植(G5) |
| L1 low raw candidates | G2 | experimental | △ `--det-low-score-thr`で`raw_detections`を出力。lowは既定未分類、`--classify-low-detections`で明示分類 |
| L1 タイル推論(遠方小信号) | G2 | works-here | ○ +20%検出/デグレ0(c1af6a38) |
| L1 新モデル試走(素の --detector) | G3 | reproducible | ○ flexibility contract。多クラス/動的shape は明示エラー |
| L3 地図マッチ(lanelet2 投影+Hungarian) | G2 | works-here | ○ 84.7%マッチ(6カメラ) |
| L3 融合+自動補正(フリップ修復/方向スナップ/未マッチ分類) | G2 | works-here | ○ |
| L3 タイムライン可視化(レビュー優先度付き) | G2 | works-here | ○ build/tl_match/re_timeline.html |
| L4 CVAT 往復(人手レビュー→GT化) | G1/G2 | works-here | ○ ロスレス 299/299。bbox/visibility/reject/map id修正の主経路 |
| L4.5 RE timeline review(状態区間→A' sidecar伝播) | G1/G2 | works-here | ○ 代表crop候補付きHTML。c1af6a38 smoke: 24 RE→8 group、62 segment、2376 annotation更新、再aggregate成功。実GTレビューは未 |
| L2 COCO / CVAT 出力 | G4 | works-here | ○ |
| L2 AWML 派生データセット | G4 | experimental | △ 互換手段。実 create_data で未検証(PLAN 3, 優先度低) |
| L4 deepen 変換 | G4 | experimental | △ 契約表のみ・変換は他リポジトリ・未検証 |
| L6 評価(GTフリー指標: 距離別プロファイル/時間安定性) | G1 | experimental | △ evaluate_signals.py |
| L6 評価(GT指標: 精度/PR、confusion) | G1 | **works-here** | ○ 実GT(ad266d7c 人手object_ann 518箱)でL1初測定: 検出P0.58/R0.85、状態精度0.78。`eval_vs_gt.py` |
| A→B 標準t4変換(object_ann) | G4→core | works-here | ○ `to_object_ann.py`。AWML/COCO/Deepen/CVATは既存ツール委譲。自作exporterはdeprecated |
| L5 ros2 パリティ検証 | G6 | not-started | ✗ 隔離中。受入=launch int8 と一致 |
| 一括実行(run_dataset.py) | G2/G5 | works-here | ○ 複数データセット・チャンネル自動発見 |
| パス/依存の抽象化(TLR_MODEL_ROOT, requirements) | G5 | reproducible | ○ |
| モデル管理(hash検証/engineキャッシュ/取得) | G5 | not-started | ✗ backlog(PLAN 9) |

## 今できないこと(能力の穴 — 主目的からの優先度順)

1. **評価そのものがまだ回っていない(最大の穴)** — GT 指標が主目的の核なのに、
   GT(レビュー済みラベル)が1件も無く L6 の GT ブロックが動かせない(G1←G2, PLAN 2.5)。
2. **人間が確認したGTがまだ無い** — CVAT往復とRE timeline review基盤は通ったが、
   実際に人が `accepted/fixed/rejected` を付けたラベルはまだ無い(G2, PLAN 2.5/2.7)。
   これが 1 を塞いでいる。
3. **run間比較の実証がない** — fp32 vs int8、モデル間比較を GT 上で出していない(G1)。
4. **複数データセットのスケール未実証** — 1データセットで1回動いただけ(G2/G5)。
5. **モデルの内容同一性が未管理** — 評価結果とモデルの対応が名前依存(G5, PLAN 9)。

互換系(AWML 実学習検証・deepen)は**手段**なので、上記が片付くまで優先度を下げる。

## 直近マイルストーン(主目的を前進させる的)

- **【最優先】G2→G1 の接続**: c1af6a38 のフラグ上位フレームだけでも人手レビューする。
  CVATでbbox/visibility/reject/map idを直し、RE timeline reviewでstate区間を確定し、
  L6 の GT 指標を初回算出する。これで主目的「評価」が experimental→works-here に
  上がり、同時に「効率的な GT 生成」も実証される。
- **G1 run間比較**: 同じ GT 上で fp32 vs int8、tiles有無の精度差を数値化。
- **G5**: モデル hash 検証だけ先行(評価結果の証跡として安く効く)。
