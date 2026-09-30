---
id: run-i936-condition-readout-r11-s936
type: run
task: ball_refiner_3d
sequence: 17
recorded_at: '2026-09-30'
title: 20k flowの凍結条件tokenから全混合平均を読むCPU診断
issue: 936
provider: codex
status: planned
config:
  encoder: fixed final flow at 20000 updates
  target: all-component mixture mean, no GT in fit
  fit_split: all 64 train rallies
  evaluation_split: all 16 val rallies
  solver: float64 gelsd, rcond=1e-12, intercept, no ridge
  diagnostic_tolerance_m: 0.1
  native_threads: 1
metrics: {}
artifacts: {}
parents:
- run-i936-h-dev-long-flow-regression-r10-s936-20260930
relations: []
papers: []
tags: []
date: '2026-09-30'
---

[事前固定した計画](https://github.com/Motoki0705/tennis-lab/issues/936#issuecomment-5908211256)のCPU read-outを実行予定。

条件encodingを先に確認するため、20k flowの重みを凍結し、pool直後tokenから
全混合平均への線形headだけを全trainでfitする。同じvalで固定評価し、
loss weight・生成入力・本体architecture・seedを変更しない。
成功は平均情報の保存を支持するが時間モデルが利用できる保証ではない。
失敗も非線形復元の不可能性を示さない。実験前なのでmetricsは空。
