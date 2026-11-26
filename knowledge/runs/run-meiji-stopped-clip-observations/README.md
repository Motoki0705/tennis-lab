# Meiji 4camの停止14 clipで2D観測を照合する

[#2363 の全69 clip評価](../run-i2363-trt-player-video/README.md)で停止した `meiji_4cam/video_000` の14件を調べた。人物対応で13件、コート向きで1件が停止した。検証設計と保存観測への介入条件は担当issueに記録した。 [#2636 の設計](https://github.com/Motoki0705/tennis-lab/issues/2636#issuecomment-7190567683) · [CPU再生条件](https://github.com/Motoki0705/tennis-lab/issues/2636#issuecomment-7191063580)

## 入力とレビュー手順

- #2363 はダブルス各側2人、4視点、59.94 fps、DINO TRT FP16、スコア閾値0.30。ボールを無効にし、選手を使ってclipごとにコート向きを判定する。進捗と終了状態のSHA-256を元runの記録に照合した。4視点の元動画、検出・追跡・選別の参照は [入力とフレームの索引](inputs.json) にある。
- 選手境界は次点groupの観測、人数不足は[#2440の不足側raw track](https://github.com/Motoki0705/tennis-lab/blob/63e9730fa23a57d46e7df4d9d4fed9e9204e9f94/knowledge/runs/run-i2440-player-selection/loss.json)、対応曖昧は最小marginペアの両方に観測があるフレーム、コート向きは約30 fps格子を使った。各集合の20・50・80%を基本時点とした。各±6元フレームと最長の内部欠測の前・中・後も抽出した。元フレーム番号は0始まり。同期154時点、計616カメラフレームを保存した。
- 1人のCodexレビュアーが、基本42時点の元画と4視点の重畳、焦点視点の欠測前後と近接した人物の拡大を確認した。衣服とプレイ中の動きを照合し、T（水色ショーツ）、W（白帽）、C（Wのペア）、D（Tのペア）の元DINO行を各基本時点で記録した。選手名は使っていない。 `outside` は画面外、`clipped` は画面端で判定に十分な可視部がない、`occluded` は他の選手で遮られて独立boxを評価できない。 [基本168カメラフレームの照合](anchor-labels.csv) · [追加確認箇所と元行](events.json)
- 非選手の検出数には、基本時点の審判台の人物と、焦点のgroupが採用した対象外のboxを記録した。後者はbox内の人物の役割を確定できない例を含む。DINOが人物を検出すること自体は人物検出タスクの仕様で、評価対象コートの選手候補に採用されたかも確認した。非選手の列はレビューで確認した下限。
- box統合は複数人を囲む元行、分裂・重複は同じ人の大小boxが同一時点にある箇所を、カメラ・フレームごとに数えた。途切れは、独立boxが前後にあり中間の確認時点で消えた区間数。ID切替は観測対象が変わったraw ID数、断片化は同一人物を短い確認区間で複数raw IDが担った件数。一つの事象が複数の種類に該当することがある。 [集計と確認フレーム](review-counts.json)

## 論文図

各clipのPDFは、同じ元フレームの4視点と、該当する場合に焦点視点3時点の拡大を収める。橙は判定対象または照合した行、緑は選別済み、青は追跡済みで選別外、灰は追跡IDのないDINO box。`t` はrawの安定追跡ID、`g` は選別group、`r` は元DINO行。人の可視性とbox配置が分かるよう、元のRGBのコート近傍をトリミングした。 [全14図の索引とSHA-256](figures/index.csv)

本文の図候補は、004の2人を囲むboxとIDの交代、020の検出・追跡済みだが選別外の選手、038の同一選手の重複boxと異なるgroup。 [004 f282](figures/clip_004-f0282.pdf) · [020 f131](figures/clip_020-f0131.pdf) · [038 f190](figures/clip_038-f0190.pdf)

## 再現

`I2363_RUN` はメインcheckoutの `.worktree-outputs/parent-2278-issue-2363-primary/i2363/i2363-all-v1` を指す。ローダーは #2440 の `63e9730fa23a57d46e7df4d9d4fed9e9204e9f94` と #2460 の `61fef16091952b5c0df06f89b32603656a29a745` のGitオブジェクトを使う。必要なら元PRのrefを取得してから実行する。 [#2440 の記録PR](https://github.com/Motoki0705/tennis-lab/pull/2450) · [#2460 の入力参照](https://github.com/Motoki0705/tennis-lab/blob/61fef16091952b5c0df06f89b32603656a29a745/knowledge/runs/run-meiji-doubles-player-reconstruction/cpu-upstream.csv)

checkoutのrootで、`OUT` に新規出力先を指定して実行する。レビュー画像の全解像度版と切り出し前の4視点は `extract` に保存される。記録済みのレビュー行を `audit.py` が照合する。CPU再生は、新規DINO boxに必要なpose・外観特徴を補わず、確認した行の除去、同一選手groupの統合、次点groupを外す感度比較を個別に行う。 [介入した元行とgroup](replay-edits.json)

```bash
bundle=knowledge/runs/run-meiji-stopped-clip-observations
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -B "$bundle/review.py" "$I2363_RUN" "$OUT/extract" --jobs 4
.venv/bin/python -B "$bundle/audit.py" "$OUT/extract" "$OUT/review-counts.json"
.venv/bin/python -B "$bundle/figures.py" "$OUT/extract" "$OUT/figures"
.venv/bin/python -B "$bundle/replay.py" "$I2363_RUN" "$OUT/replay" --jobs 4
```
