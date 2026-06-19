---
id: run-i518-exp4
type: run
title: exp4 branched + pos30
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_branched
  loss: canonical_rot_v3
  data: multiview_sequence
metrics:
  ang_error_deg: 54.1
  angular_error_median_deg: 26.9
  angle_accuracy_30deg: 0.529
  position_error_m: 0.40
  position_error_median_m: 0.29
artifacts:
  log: experiments/logs/
  output_dir: ''
parents: [run-i518-exp3]
relations:
  - {to: run-i518-exp3, rel: compares}
tags: [plcs, rotation, branched, reweight]
---

## 考察 / Findings

分岐モデル(exp3)で位置を強く上げたら回転が保てるか（`canonical_rot_v3`: pos30）を検証。

- 位置は `0.82 → 0.40m` に下がるが、**回転は `13.6 → 54.1°` に崩壊**。
- 分岐 readout があっても**共有 trunk の勾配は依然 position 上げで回転を潰す**。exp2 と同じ轍。

→ 分岐だけでは不十分。trunk そのものの勾配共有を断つ（分離 / detach）方向を exp5/exp6 で試す。
