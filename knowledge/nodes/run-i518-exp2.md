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
  position_error_m: 0.3
  position_error_median_m: 0.23
artifacts:
  log: experiments/logs/
  output_dir: ''
  curves: knowledge/runs/run-i518-exp2/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_5
parents:
- run-i518-exp1
relations:
- to: run-i518-exp1
  rel: compares
tags:
- plcs
- rotation
- reweight
---

## 考察 / Findings

### 要約
位置崩壊を損失リバランスだけで戻せるか検証。位置は回復するが回転が逆戻りし、共有 trunk のハードなトレードオフが確定。

### アーキテクチャ詳細
exp1 と同じ共有 trunk。損失を `canonical_rot_v2` にリバランス（position 2→30, angle 1.0→0.3, rotation 0.5→0.1）。

### メトリクスの解釈
位置は `1.10→0.30m` に回復するが、回転が `20.4→52.0°` に逆戻り。

### アーキテクチャ⇄メトリクスの因果考察
スカラー重みを動かしても共有 trunk の固定容量を奪い合うだけで、片方を立てれば片方が倒れる。重み調整は競合の本質を解かない。

### 既存実験との比較
親 [[run-i518-exp1]] と対（`compares`）。exp1↔exp2 で「共有 trunk はスカラー重み調整では片方しか立たない」ハードなトレードオフが確定。

### 次に有効な実験
重み調整では win-win 不可能。アーキテクチャ（trunk 分離 / 分岐）で容量競合を解く方針へ（exp3 以降）。
