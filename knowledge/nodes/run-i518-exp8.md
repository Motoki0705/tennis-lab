---
id: run-i518-exp8
type: run
title: exp8 branched3 + pos6 (Pareto)
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_branched3
  loss: canonical_rot_v5
  data: multiview_sequence
metrics:
  ang_error_deg: 12.5
  angular_error_median_deg: 9.9
  angle_accuracy_30deg: 0.937
  position_error_m: 0.6
  position_error_median_m: 0.47
artifacts:
  log: experiments/logs/
  output_dir: ''
  curves: knowledge/runs/run-i518-exp8/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_12
parents:
- run-i518-exp3
relations: []
tags:
- plcs
- rotation
- branched
- pareto
---

## 考察 / Findings

### 要約
分岐 3 層 + 控えめな位置上げで、回転を崩さず位置を回収した共有 / 分岐路線の Pareto 最良（回転 12.5°・位置 0.60m）。

### アーキテクチャ詳細
`multiview_axial_base_branched3`（共有 8 + タスク別 3 + 3 層）。損失 `canonical_rot_v5`（強回転 angle1.0/rot0.5 + 控えめ位置 pos6）。

### メトリクスの解釈
回転 `12.5°`・位置 `0.60m`。回転を崩さず位置を回収したが、位置は `0.60m` 止まり。

### アーキテクチャ⇄メトリクスの因果考察
分岐に十分な深さを与えつつ位置重みを控えめにすることで競合点を回避。ただし共有 trunk の位置下限が ~0.5–0.6m に存在する。

### 既存実験との比較
親 [[run-i518-exp3]] 系の到達点。後続 [[run-i518-exp9]]（pos12）は回転崩壊し、共有 trunk のフロンティアを確定させる。

### 次に有効な実験
共有 / 分岐路線の到達点。これ以上は分離 trunk + 補助位置（exp10）でしか越えられない。
