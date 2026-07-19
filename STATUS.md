# STATUS — capabilities, goals, vision

「今これは頼っていいか?」に答えるダッシュボード。マイルストーン時に見直す
(毎コミットではない)。役割分担: **能力・ゴール・成熟度はここだけ**、タスク
(動詞)は PLAN、契約・IF・層は README、横断事実はプロジェクトメモリ。

最終レビュー: 2026-07-19

## North Star

> モデル非依存・環境非依存で、走行データを **レビュー可能・地図連携済み・
> 学習可能** な信号アノテーションへ変換し、人手補正と評価まで回せる、
> 信号認識(TLR)開発の道具箱。

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

| | ゴール | 成熟度 | 前進させる作業 |
|---|---|---|---|
| **G1** | 学習データ生成(走行→検出/分類→地図付与→レビュー→AWML) | works-here | PLAN 2.5 / 3 |
| **G2** | モデル評価・比較(指標・GT・run間比較) | experimental | PLAN 2.5(GT 供給) |
| **G3** | 新モデル立ち上げ(試走ベンチ) | reproducible | — |
| **G4** | ライブ検証(ros2 パリティ) | not-started | PLAN 4 |
| **G5** | 再現性・運用(パス/依存/来歴/一括実行/モデル管理) | partial | PLAN 9 |

## Capabilities(具体的に動くもの)

成熟度 × 所属ゴール。○=できる、△=限定/未検証、✗=未着手。

| 能力 | ゴール | 成熟度 | 状態・注意 |
|---|---|---|---|
| L1 検出+分類(YOLOX int8 engine / CoMLOps ONNX) | G1 | reproducible | ○ ~0.8s/frame(GPU)。engine は非移植(G5) |
| L1 タイル推論(遠方小信号) | G1 | works-here | ○ +20%検出/デグレ0(c1af6a38) |
| L1 新モデル試走(素の --detector) | G3 | reproducible | ○ flexibility contract。多クラス/動的shape は明示エラー |
| L2 COCO / CVAT 出力 | G1 | works-here | ○ |
| L2 AWML 派生データセット | G1 | experimental | △ 実 AWML の create_data で未検証(PLAN 3, hold) |
| L3 地図マッチ(lanelet2 投影+Hungarian) | G1 | works-here | ○ 84.7%マッチ(6カメラ) |
| L3 融合+自動補正(フリップ修復/方向スナップ/未マッチ分類) | G1 | works-here | ○ |
| L3 タイムライン可視化 | G1 | works-here | ○ build/tl_match/re_timeline.html |
| L4 CVAT 往復 | G1 | works-here | ○ ロスレス 299/299 |
| L4 deepen 変換 | G1 | experimental | △ 契約表のみ・変換は他リポジトリ・未検証 |
| L6 評価(GTフリー指標) | G2 | experimental | △ evaluate_signals.py。距離ビン別プロファイル |
| L6 評価(GT指標: 精度/PR) | G2 | experimental | ✗ レビュー済みラベル(GT)が未生成 |
| L5 ros2 パリティ検証 | G4 | not-started | ✗ 隔離中。受入=launch int8 と一致 |
| 一括実行(run_dataset.py) | G1/G5 | works-here | ○ 複数データセット・チャンネル自動発見 |
| パス/依存の抽象化(TLR_MODEL_ROOT, requirements) | G5 | reproducible | ○ |
| モデル管理(hash検証/engineキャッシュ/取得) | G5 | not-started | ✗ backlog(PLAN 9) |

## 今できないこと(能力の穴 — 優先度順)

1. **精度の実数値がない** — 人手レビューループが未実走で GT が無い(G2 の律速。PLAN 2.5)。
2. **AWML 学習が通るか未確認** — 派生データセットを実 create_data に通していない(PLAN 3)。
3. **複数データセットのスケール未実証** — 1データセットで1回動いただけ(G1/G5)。
4. **モデルの内容同一性が未管理** — 同名別モデルを検出できない(G5, PLAN 9)。
5. **ライブ整合が未検証** — オフライン結果が実ノードと一致する保証がない(G4)。

## 直近マイルストーン(次に成熟度を1段上げる的)

- **G2 を experimental→works-here**: レビュー済みラベルを少数でも作り、GT 指標を初回算出。
- **G1 を works-here→reproducible**: 2つ目のデータセットで run_dataset を通す。
- **G5**: モデル hash 検証だけ先行実装(engine キャッシュより安く効く)。
