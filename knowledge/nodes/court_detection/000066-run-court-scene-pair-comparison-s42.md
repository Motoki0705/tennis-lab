---
id: run-court-scene-pair-comparison-s42
type: run
task: court_detection
sequence: 66
recorded_at: '2026-11-10'
title: Court全6ペアの群contrastと9条件4シーン比較
issue: 1586
provider: codex
status: done
config:
  run.seed: 42
  primary_metric: kp_median_distance_px
  train_images: 5500
  updates: 33160
metrics:
  delta/kp_mean_distance_px: 49.41506481170654
  delta/kp_median_distance_px: 52.79571634531021
  delta/seg_miou: -0.2129666730761528
  delta/line_dice: -0.09248123317956924
  delta/semantic_line_miou: -0.04371355101466179
artifacts:
  run_dir: knowledge/runs/run-court-scene-pair-comparison-s42
  comparison: knowledge/runs/run-court-scene-pair-comparison-s42/comparison.json
  figure: knowledge/runs/run-court-scene-pair-comparison-s42/synthetic-kp-median.png
  sources: knowledge/runs/run-court-scene-pair-comparison-s42/sources.json
  crosscheck: knowledge/runs/run-court-scene-pair-comparison-s42/crosscheck.json
  drive_outputs:
  - outputs/court_detection/campaign/scene-pair-5500-s384-colab/documentation/final-comparison-20261110
parents:
- run-court-scene-pair-execution-s42
- run-court-scene-pair-manifests-v1
relations:
- to: group-court-scene-diversity-colab-s42
  rel: compares
  note: 同じ実画像valと旧checkpointを保ち、全6ペアと共通合成評価を追加する
papers: []
tags: []
summary_impact: none
date: '2026-11-10'
repro:
  commit: aac0f649a9ad8f26f28b6977799459084aca7e12
  command: bash knowledge/runs/run-court-scene-pair-comparison-s42/repro.sh
  note: 集計scriptはこのnodeのrun bundleに保存。baseのsrcを変更せずCPU集計した。
---

## CPU集計の結果と再現

9件のraw評価JSONを元done markerのSHAと照合し、checkpoint/33160更新、config/data digest、BS/seed、共通manifest、ID順序・一意性・件数、scene別macroを検証して集計した。P01は既存scene-2を一度だけ使う。主指標は実画像KP中央値。比較script、全5指標のCSV/JSONと図はこのrun bundleにあり、結論と制約の正本はcomparison groupを参照する。

独立したcrosscheck.pyはcompare.pyをimportせず、raw JSONからFractionで群平均と4scene平均を計算した。抽出/区分527項目、独立演算65項目、CSV/JSON780セルは全て完全一致した。5指標のraw δは符号を変換していない。KPでは正、IoU/Diceでは負が同じ群の悪化を意味する。加法性点検はvariance・標準誤差・検定・CIとして扱わない。

summary_impact: none。集計・再現性のrunとしてgroupの根拠を提供する。現在判断はgroupでsummaryへ反映し、pipelineのcheckpointは変更しない。
