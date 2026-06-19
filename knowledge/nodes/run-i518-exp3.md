---
id: run-i518-exp3
type: run
title: exp3 branched 8+2+2
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_branched
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 13.6
  angular_error_median_deg: 9.7
  angle_accuracy_30deg: 0.886
  position_error_m: 0.82
  position_error_median_m: 0.68
artifacts:
  log: experiments/logs/
  output_dir: ''
parents: [run-i518-exp1]
relations: []
tags: [plcs, rotation, branched]
---

## 考察 / Findings

回転に**専用の分岐 readout**を与えた分岐モデル（共有8層 + タスク別2層、loss は exp1 と同じ
`canonical_rot`）。

- 回転 `13.6°` と exp1(20.4°) からさらに改善し、位置も `1.10 → 0.82m` と緩和。**両方向に前進**。
- ただし共有 trunk が依然回転支配で、位置は `0.82m` と高止まり。

→ 「分岐で回転に容量を与えると効く」が分かった有望ノード。ここから幅/深さ/分離/重みを枝分かれ探索
（exp4=重み, exp5=完全分離, exp8=深さ3）。
