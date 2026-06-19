---
id: run-i518-exp7
type: run
title: exp7 split, canon→rot
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_split
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 67.2
  angular_error_median_deg: 60.2
  angle_accuracy_30deg: 0.315
  position_error_m: 0.32
  position_error_median_m: 0.22
artifacts:
  log: experiments/logs/
  output_dir: ''
parents: [run-i518-exp5]
relations: []
tags: [plcs, rotation, split-trunk]
---

## 考察 / Findings

分離 trunk の回転側に **canonical ヘッド**を載せ、3D幾何の補助信号で回転を救えるか検証。

- 回転 `67.2°`・位置 `0.32m`。canonical を足しても**回転は依然崩壊**＝exp5 と大差なし。
- 決定的な失敗分析: 回転 trunk に**位置タスクの勾配が無い**こと自体が問題で、canonical（静的ポーズ）
  だけでは多視点三角測量/対応付けの特徴を学べない。

→ この失敗分析が exp10 着想の直接の引き金。「canonical では足りない、**位置**タスクの勾配を補助で
  回転 trunk に流せ」が [[run-i518-exp10]] の原理。
