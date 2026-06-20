---
id: run-i518-exp5
type: run
title: exp5 full split 0+6+6
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_split
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 71.0
  angular_error_median_deg: 56.1
  angle_accuracy_30deg: 0.332
  position_error_m: 0.32
  position_error_median_m: 0.27
artifacts:
  log: experiments/logs/
  output_dir: ''
  curves: knowledge/runs/run-i518-exp5/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_8
parents:
- run-i518-exp3
relations: []
tags:
- plcs
- rotation
- split-trunk
---

## 考察 / Findings

### 要約
trunk を完全分離すれば両立すると期待したが、位置は良いのに回転が `71.0°` と baseline 以下に崩壊。回転は位置タスクと co-train が必要という重要発見。

### アーキテクチャ詳細
完全分離 `multiview_axial_base_split`（共有 0 層 + 回転 6 層 + ポーズ 6 層）。損失は `canonical_rot`。

### メトリクスの解釈
位置 `0.32m` と良好だが、回転は `71.0°` と baseline (61.6°) より悪化。

### アーキテクチャ⇄メトリクスの因果考察
回転は HARD タスクで、位置タスク（多視点三角測量 / 対応付け）と同じ trunk で co-train される必要がある。位置勾配を切ると回転 trunk が三角測量特徴を学べず崩壊する。

### 既存実験との比較
親 [[run-i518-exp3]] の分岐路線から完全分離へ踏み込んだ結果、回転が悪化。後に [[run-i518-exp10]] がこの「分離は回転を殺す」を条件付きで覆す（`contradicts`）。

### 次に有効な実験
回転 trunk に位置の信号を補助的に戻す道（exp7→exp10）が見えてきた。
