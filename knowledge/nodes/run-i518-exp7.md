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
  curves: knowledge/runs/run-i518-exp7/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_11
parents:
- run-i518-exp5
relations: []
tags:
- plcs
- rotation
- split-trunk
---

## 考察 / Findings

### 要約
分離 trunk の回転側に canonical ヘッドを足しても回転は救えず `67.2°`。この失敗分析が exp10 着想の直接の引き金になった。

### アーキテクチャ詳細
完全分離 `multiview_axial_base_split` の回転側に canonical（静的ポーズ）ヘッドを追加。損失 `canonical_rot`。

### メトリクスの解釈
回転 `67.2°`・位置 `0.32m`。canonical を足しても回転は依然崩壊で exp5 と大差なし。

### アーキテクチャ⇄メトリクスの因果考察
回転 trunk に位置タスクの勾配が無いこと自体が問題で、canonical（静的ポーズ）だけでは多視点三角測量 / 対応付けの特徴を学べない。

### 既存実験との比較
親 [[run-i518-exp5]] とほぼ同等の崩壊。canonical 補助では不足という決定的な反証になった。

### 次に有効な実験
この失敗分析が直接の引き金。「canonical では足りない、位置タスクの勾配を補助で回転 trunk に流せ」が [[run-i518-exp10]] の原理。
