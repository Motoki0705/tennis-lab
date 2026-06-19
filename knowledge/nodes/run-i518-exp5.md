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
parents: [run-i518-exp3]
relations: []
tags: [plcs, rotation, split-trunk]
---

## 考察 / Findings

trunk を完全分離（共有0層 + 回転6層 + ポーズ6層、loss は `canonical_rot`）。容量競合を断てば
両立すると期待した。

- ところが**位置は `0.32m` と良いが回転が `71.0°` に崩壊**＝baseline より悪い。
- 重要な発見: 回転は HARD タスクで、**位置タスク（多視点三角測量/対応付け）と同じ trunk で
  co-train される必要**がある。位置勾配を切ると回転を学習できない。

→ 「完全分離は回転を殺す」。回転 trunk に位置の信号を**補助的に**戻す道（exp7→exp10）が見えてきた。
