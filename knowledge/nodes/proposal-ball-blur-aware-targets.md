---
id: proposal-ball-blur-aware-targets
type: proposal
title: ball detectionへblur-aware heatmap targetを導入する
curator: chatgpt-schedule
date: 2026-08-06
status: ready
task: ball_detection
repo_paths:
  - src/tasks/ball_detection/data/types.py
  - src/tasks/ball_detection/data/dataset.py
  - src/utils/data/heatmaps.py
  - src/tasks/ball_detection/training/lightning_module.py
evidence_runs: []
hypothesis:
  statement: modelとoptimizerを固定し、点Gaussian教師をstreak中心・方向・長さを表すblur-aware heatmapへ置換すると、高motion-blur frameの位置誤差とF1が改善する
  expected_effect: 高blur subsetでmean_distance_pxを10%以上低下させ、F1を1 percentage point以上改善しつつthroughput低下を10%以内に抑える
  failure_condition: mean_distance_px改善10%未満かつF1改善1 point未満、通常frameのF1が1 point超低下、またはFPSが10%超低下する
evaluation:
  metrics: [f1, mean_distance_px, high_blur_f1, high_blur_mean_distance_px, fps]
  baseline_nodes: [run-i551-dinov3-rope-tracknet-train-retry1]
  seeds: 3
  acceptance: 3 seeds平均で高blur subsetのmean_distance_pxを10%以上改善し、F1を1 percentage point以上改善し、通常frameのF1低下を1 point以内、FPS低下を10%以内にする
parents:
  - run-i551-dinov3-rope-tracknet-train-retry1
relations:
  - to: paper-arxiv-2509-18387
    rel: derived-from
tags: [literature, ball-detection, motion-blur, supervision]
---

## 背景

現行ball detectionはpoint Gaussian heatmapを教師とするため、露光中にstreakとなるballの中心と形状を表現しない。BlurBallはtable tennisでblur中心・方向・長さを明示するが、tennis-labで有効かは未検証である。

## 現行実装との差分

`FrameLabel`へ任意の`blur_angle`と`blur_half_length`を追加し、値が存在するsampleのみ線分または異方性Gaussian targetを生成する。baseline model、optimizer、split、input frames、augmentationは固定し、まずtarget生成だけを変更する。architectureへのSE blockや補助headは別実験とする。

## 最小検証

既存TrackNet splitへ固定の高motion-blur評価subsetを定義する。公開BlurBall datasetはdomain確認用に限定し、主要比較は同一tennis split・同一annotation policyで行う。blur属性が無い既存frameは点targetのまま保持する。

## 比較対象

baselineは`run-i551-dinov3-rope-tracknet-train-retry1`のDINOv3+RoPE detectorとpoint Gaussian targetである。treatmentは同一modelへblur-aware targetのみを適用する。

## 合格条件と停止条件

frontmatterのacceptanceを満たした場合のみ、SE blockまたはblur attribute headの追加へ進む。通常frameのF1が1 point超低下、label中心とstreak中心の不整合が5 px超で頻発、またはdataset licenseが確認できない場合は拡張を停止する。

## リスク

公開datasetはtable tennis中心でdomain gapがある。streakを直線とみなす仮定はbounceやracket contactで破綻し、追加annotation costも発生する。target変更とmodel変更を同時に行わず、因果帰属を保つ。
