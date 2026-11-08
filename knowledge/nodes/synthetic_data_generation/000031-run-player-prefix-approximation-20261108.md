---
id: run-player-prefix-approximation-20261108
type: run
task: synthetic_data_generation
sequence: 31
recorded_at: '2026-11-08'
title: 再学習しないNHT prefix近似の極限・遮蔽・輪郭を比較する
provider: codex
status: running
config:
  background: B00
  cameras: 3
  base_resolution:
  - 256
  - 144
  ssaa2_camera_count: 2
  candidates:
  - weighted
  - normalized
  - closed_normalized
  no_retraining: true
metrics:
  cpu_tests_passed: 55
artifacts:
  run_dir: knowledge/runs/run-player-prefix-approximation-20261108
parents:
- run-player-depth-compositing-20261108
relations: []
papers: []
tags: []
issue: 1728
date: '2026-11-08'
summary_impact: none
---

[親の人間判断](https://github.com/Motoki0705/tennis-lab/issues/1723#issuecomment-6722754922)により、
再学習せず近似で進める。alpha掛け直し・特徴正規化・全背景alphaでの正規化を比較する。
現在は研究用CPU小配列20件と既存35件の計55件が成功した（3.85秒、-n 2）。
GPUの比較はまだ終わっておらず、方式を採用したとはしない。

[再現bundle](../../runs/run-player-prefix-approximation-20261108/README.md)に候補・数値基準と
コードを置く。物理的な局所色の正解を得る試験ではない。背景・checkpoint・NHT本体の
変更や学習は行わない。方式選定前なので、既存の未対応という判断からはまだ変えず
summary_impactはnoneとする。完了後に結果と現行契約を照合する。
