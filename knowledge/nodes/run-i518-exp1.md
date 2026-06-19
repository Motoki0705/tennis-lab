---
id: run-i518-exp1
type: run
title: exp1 angle loss (shared)
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 20.4
  angular_error_median_deg: 16.4
  angle_accuracy_30deg: 0.789
  position_error_m: 1.10
  position_error_median_m: 0.99
artifacts:
  log: experiments/logs/
  output_dir: ''
parents: [run-i518-baseline]
relations: []
tags: [plcs, rotation, angle-loss]
---

## 考察 / Findings

反転サドル対策として **wrapped-angle smooth-L1 の `angle` 損失**を追加（`canonical_rot`:
`angle_weight=1.0`, `rotation_weight=0.5`）。180° でも勾配が消えない（grad@180 = [0,1.0]）。

- 回転は `61.6° → 20.4°` と大幅改善。反転問題に angle 損失が効くことを実証。
- 一方、共有 trunk のため**位置が `0.26 → 1.10m` に崩壊**。回転と位置が trunk 容量を奪い合う構図が露呈。

→ 「angle 損失は必須だが、共有 trunk では位置と競合する」。この競合の解き方を exp2 以降で探索。
