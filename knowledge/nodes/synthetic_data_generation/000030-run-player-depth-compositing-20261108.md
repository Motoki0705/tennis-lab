---
id: run-player-depth-compositing-20261108
type: run
task: synthetic_data_generation
sequence: 30
recorded_at: '2026-11-08'
title: NHT背景と不透明前景の奥行き合成の情報不足とshader整合を検証する
provider: codex
issue: 1728
date: '2026-11-08'
status: running
summary_impact: none
config:
  model: existing_nht_shader_no_retraining
  background: B00
  court: court-000
  resolution: [256, 144]
  camera_count: 3
metrics:
  cpu_tests_passed: 35
artifacts:
  run_dir: knowledge/runs/run-player-depth-compositing-20261108
parents: [run-court-assets-colab-l4-20261022]
relations: []
papers: []
tags: []
---

## 現時点で確認したこと

RGB/alpha/EDが同一でも、不透明前景を挿入した結果が異なる既知層を構成した。
必要なのは前景までのRGB寄与と透過率であり、期待depthの単純比較では不足する。
現行NHTは調和特徴量の蓄積後に非線形MLPを適用するので、途中特徴量のdecodeを
premultiplied RGBとして扱うことも仕様からは導けない。

[研究記録と再現スクリプト](../../runs/run-player-depth-compositing-20261108/README.md)に
CPUの手計算・Colab probeの入口を置く。CPU小配列14件と既存NHT境界21件の35件が成功した。
GPU実測はまだ終わっていないため、合成方式を採用済みとはしない。

既存Court生成recipe・採用資産・checkpointの判断を変える結果ではないため、
現段階ではsummary_impactをnoneとする。背景再学習・本番統合・NHT本体変更は行わない。
