---
id: run-i518-exp2
type: run
title: exp2 reweight pos30 (shared)
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base
  loss: canonical_rot_v2
  data: multiview_sequence
metrics:
  ang_error_deg: 52.0
  angular_error_median_deg: 35.6
  angle_accuracy_30deg: 0.437
  position_error_m: 0.30
  position_error_median_m: 0.23
artifacts:
  log: experiments/logs/
  output_dir: ''
parents: [run-i518-exp1]
relations:
  - {to: run-i518-exp1, rel: compares}
tags: [plcs, rotation, reweight]
---

## 考察 / Findings

exp1 の位置崩壊を**損失リバランスだけ**で戻せるか検証（`canonical_rot_v2`: position 2→30,
angle 1.0→0.3, rotation 0.5→0.1）。

- 位置は `1.10 → 0.30m` に回復するが、**回転が `20.4 → 52.0°` に逆戻り**。
- exp1↔exp2 で「共有 trunk はスカラー重み調整では片方しか立たない」**ハードなトレードオフ**が確定。

→ **重み調整では win-win 不可能**。アーキテクチャ（trunk 分離 / 分岐）で容量競合を解く方針へ。
