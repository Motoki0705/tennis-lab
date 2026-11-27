---
id: run-i2637-ball-temporal-context
type: run
task: ball_detection
sequence: 98
recorded_at: '2026-11-27'
title: 固定評価集合におけるボール検出の時系列入力と形状ベース基準
issue: 2637
provider: codex
session: 01a21796-0065-7040-91b3-20885883a394
date: '2026-11-27'
status: done
summary_impact: none
config:
  dataset: ball-mix1800-720p-all-frame-train-val-v1
  split: stratified_validation_subset
  seed: 42
  input_conditions: [ordered, repeat_anchor, shuffle_context]
  x3d_t32: {frames: 32, input_channels: 5, image_wh: [1280, 720], precision: bf16, presence_threshold: 0.5380124449729919}
  i935_t8: {frames: 8, input_channels: 2, image_wh: [512, 288], precision: fp32, presence_threshold: 0.5}
  shape_baseline: {image_wh: [1280, 720], calibration_groups: 6, presence_threshold: 0.9966928958892822}
metrics:
  evaluation_anchors: 144
  observed_anchors: 88
  reviewed_empty_anchors: 32
  out_of_frame_anchors: 24
  calibration_anchors: 120
  x3d_t32_mean_source_px: [24.2197, 253.8583, 88.4774]
  x3d_t32_p95_source_px: [66.3085, 699.9849, 510.1697]
  x3d_t32_within_20_px_count: [76, 2, 55]
  x3d_t32_false_positives: [12, 0, 10]
  x3d_t32_false_negatives: [13, 88, 30]
  i935_t8_mean_source_px: [34.9354, 233.1577, 58.0098]
  i935_t8_p95_source_px: [230.8529, 494.0689, 302.5817]
  i935_t8_within_20_px_count: [73, 0, 67]
  i935_t8_false_positives: [8, 0, 8]
  i935_t8_false_negatives: [15, 88, 32]
  shape_mean_source_px: 458.7350
  shape_p95_source_px: 987.1696
  shape_within_20_px_count: 7
  shape_false_positives: 35
  shape_false_negatives: 36
  gpu_job_capture_to_completion_seconds: 290.95
repro:
  commit: 0864dce25c4f3d2322994408ff5eacf1b1a80421
  branch: codex/parent-2627-issue-2637-primary
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: bash knowledge/runs/run-i2637-ball-temporal-context/reproduce.sh gpu '/mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/ball-query-desktop/ball_detection/train/query_controls/desktop-resume-20261112/dataset'
    '/mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/ball-query-desktop/ball_detection/train/query_controls/desktop-resume-20261112/prepared/evaluation_manifest.json'
    outputs/ball-paper/context
artifacts:
  run_dir: knowledge/runs/run-i2637-ball-temporal-context
  log: .training_queue/logs/0-8204305540952854139_2026-11-26T12-00-59-047145860Z_363728_ball-paper-temporal-x3d-i935.log
  output_dir: /mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2627-issue-2637-primary/ball-paper/context
parents: [run-ball-all-frame-protocol-20261022]
relations:
- {to: run-scene-x3d-dpt-20261116, rel: evaluates}
- {to: run-i935-evidence-mixed-e9-trainval-r17-20260930, rel: evaluates}
- {to: run-i2637-ball-image-difficulty, rel: related, note: 別収録meiji_4camにおける球像と見え方の測定}
- {to: run-ball-sequential-length-20261025, rel: related, note: 同じ位置・存在指標を使う全validationの学習履歴}
- {to: group-i1675-capacity-12k-s42, rel: related, note: 全validationの容量系列を同梱CSVに掲載}
- {to: group-i1899-controlled-query-ablation-s42, rel: related, note: 全validationの学習前半Tと構造変更を同梱CSVに掲載}
papers: []
tags: []
---

## 考察 / Findings

### 比較条件

#1369の長系列validationに含まれる3 sourceから、seed 42で144フレームを固定した。
Chatは12収録群から96件（observed 48、reviewed_empty 24、out_of_frame 24）、
Meiji 3camは3視点からobserved 24件、TrackNet game9はobserved 16、reviewed_empty 8件。
元の教師とclip・frame IDを保持し、全7条件で同じ144件を採点した。FPRの分母は確認済み不在56件、
FNRの分母はobserved 88件。位置は存在閾値で打ち切る前の座標を88件すべてで採点し、
元画像pxの距離が20以下の比率を求めた。存在の採点に位置条件は入れていない。
[アンカー、教師、入力sha](../../runs/run-i2637-ball-temporal-context/anchors.json) ·
[評価規則](000063-run-ball-all-frame-protocol-20261022.md)

各アンカーに連続32フレーム、i935では同じアンカーを含む中央8フレームを与えた。
OpenCVで同じ720p JPEGを復号し、i935の入力だけ512×288に縮小。時系列の通常入力、同一アンカーの
全時刻反復、固定seedでの周辺入力の並べ替えを比較した。すべてで出力を採るアンカーの位置は同じ
（0始まりでT32の15、T8の3）。既存の各モデルでMDDを入力RGBから作り、存在閾値も既存値
（X3D 0.53801244、i935 0.5）を条件間で固定した。推論はバッチ1、X3DはBF16 eager、i935はFP32。

形状基準は全画面に対して、局所コントラスト、Hessianの丸さ、白・黄の色prior、前後2・4・6 frameからの
背景差分を複合した。Chatの別6収録群から確保した120フレーム（observed 48、reviewed_empty 48、
out_of_frame 24）で `balanced (FPR + FNR) / 2` を最小にする閾値0.99669290を選んだ。
位置は形状基準でも閾値を適用する前の最大応答を使った。
[CPU基準の実装](../../runs/run-i2637-ball-temporal-context/shape_baseline.py)

### 結果

| モデル・入力 | mean / P95 元画像px | 20 px以内 | 存在FPR | 存在FNR |
| --- | ---: | ---: | ---: | ---: |
| X3D/CSN＋DPT T32・通常 | 24.22 / 66.31 | 76/88（86.36%） | 12/56（21.43%） | 13/88（14.77%） |
| X3D T32・同一フレーム反復 | 253.86 / 699.98 | 2/88（2.27%） | 0/56（0%） | 88/88（100%） |
| X3D T32・周辺の順序入替え | 88.48 / 510.17 | 55/88（62.50%） | 10/56（17.86%） | 30/88（34.09%） |
| i935 T8・通常 | 34.94 / 230.85 | 73/88（82.95%） | 8/56（14.29%） | 15/88（17.05%） |
| i935 T8・同一フレーム反復 | 233.16 / 494.07 | 0/88（0%） | 0/56（0%） | 88/88（100%） |
| i935 T8・周辺の順序入替え | 58.01 / 302.58 | 67/88（76.14%） | 8/56（14.29%） | 32/88（36.36%） |
| 色・丸さ・背景差分（CPU） | 458.73 / 987.17 | 7/88（7.95%） | 35/56（62.50%） | 36/88（40.91%） |

**時系列の寄与を支持。** 通常に比べ、反復で20 px以内率はX3Dで84.09ポイント、i935で82.95ポイント低下した。
両モデルが全144件を不在としたことが、反復時のFPR 0%とFNR 100%に対応する。i935はMDDの2chを受け取り、
同一RGBの反復では画像間差分も一定となる。実測スコアは全144件で約0.1665だった。
X3Dの反復はRGBを含む5chで、同じRGBを反復して残した状態でも位置誤差が増えた。

**順序の効果は条件によって異なる。** 順序入替えで20 px以内率がX3Dで23.86ポイント、i935で6.82ポイント低下。
通常だけが20 px以内だったのはX3Dで25件、i935で8件、入替えだけが20 px以内だったのは4件、2件だった。
i935のMeiji 3cam部分は18/24（75.00%）から20/24（83.33%）に上がり、平均誤差は75.71から35.80 pxに下がった。
順序を変えると隣接画素のMDDも変わる。結論は同じ学習済みチェックポイントへの入力介入で、固定validation
部分集合上の結果である。単フレーム専用に学習したモデルの性能と独立testへの一般化は未測定。

**形状ベース基準の限界を支持。** 720pのこの基準は、通常入力の両学習済みモデルより位置・存在の成績が低かった。
FPRは `reviewed_empty` で16/32、`out_of_frame` で19/24。色と局所形状への背景差分の併用でも、
この比較で球を背景や他の物体から分けきれなかった。形状手法一般への外挿は別の検証を要する。
[全条件・source別の表](../../runs/run-i2637-ball-temporal-context/comparison.csv) ·
[機械可読集計・paired hit数・スコア域](../../runs/run-i2637-ball-temporal-context/metrics.json)

### 小型モデルと長系列の既存結果

[過去14条件の表](../../runs/run-i2637-ball-temporal-context/historical-validation.csv)に
[容量7条件](000088-group-i1675-capacity-12k-s42.md)、[長系列4条件](000068-run-ball-sequential-length-20261025.md)、
[前半T・CNN・query構成の3追加条件](000094-group-i1899-controlled-query-ablation-s42.md)を同じ列で収録した。
元の85,473 observedフレームによる全validationの数値。容量系列の最小8.03Mはmean/P95 35.38/169.69 px、
20 px内73.62%。最大71.78Mは33.38/161.51 px、77.00%。mean最良は38.32Mの32.48 pxで、容量との単調関係は見られなかった。
長系列ではDPT T32→T64で20 px内84.03→86.58%、queryで30.64→76.29%。その比較では累積更新数も増えた。
最終T64とquery 12kをそろえた旧ConvNeXt＋1queryは、前半T32でmean 28.20 px、前半T64で28.53 pxだった。
座標queryに存在headがない行はFPR/FNRを空欄とした。全validationの履歴比較と今回の144件は評価の分母が異なる。
本比較はBall Detection summaryのモデル選定と評価手順の判断を維持する診断結果である。

### 実行と再集計

形状基準のsystemd job `20261126T113758Z_1100cbe90df9_ball-paper-shape` は終了値0、サービス経過396.96秒。
実行版は `365d15e92cc848479d3417a33c295a81fb31e781`。GPUは共有queueの `all` で2026-11-27 03:33:58 JSTに
実行版を保存し、03:38:49 JSTに `done`。保存から終了まで約291秒、モデル別ループはX3D 203.78秒、i935 28.02秒、
ピークCUDA割当はそれぞれ2.85/0.42 GiB（RTX 5060 Ti、torch 2.13.0+cu130）。
[GPU実行記録](../../runs/run-i2637-ball-temporal-context/run.json) ·
[終端記録](../../runs/run-i2637-ball-temporal-context/queue-completion.json)

実験のコマンドは [reproduce.sh](../../runs/run-i2637-ball-temporal-context/reproduce.sh) にある。
`cpu` は元storeとmanifestからアンカーを復元し、CPU基準を `outputs/ball-paper/context/shape.json` に保存する。
`gpu` はGitに固定したアンカーと同じstoreを読み、同出力先に `temporal.json` を保存する。
GPUコマンドはtraining queueに投入する。ローカル保管の2出力のSHA-256はmetrics.jsonに記録した。
集計では採点ID・教師座標・全分母の一致と位置誤差の再計算を確認した。

```bash
run=knowledge/runs/run-i2637-ball-temporal-context
out=outputs/ball-paper/context
PYTHONPATH=. .venv/bin/python -B "$run/record_comparison.py" --anchors "$run/anchors.json" \
  --shape "$out/shape.json" --temporal "$out/temporal.json" --output "$out/verification"
```
