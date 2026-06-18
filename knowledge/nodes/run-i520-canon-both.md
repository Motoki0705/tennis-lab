---
id: run-i520-canon-both
type: run
title: canon_both
issue: 520
provider: claude
status: done
config:
  model: multiview_axial_canon_split_both
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 15.908191
  angle_accuracy: 0.590794
  angle_accuracy_10deg: 0.433285
  angle_accuracy_15deg: 0.590794
  angle_accuracy_30deg: 0.877958
  angular_error_deg: 15.516711
  angular_error_median_deg: 11.934991
  angular_error_std_deg: 14.058346
  loss: 0.171861
  loss_canonical_pose: 0.005282
  pos_error_m: 0.363859
  position_accuracy: 0.810592
  position_accuracy_0.5m: 0.810592
  position_accuracy_1m: 0.96275
  position_accuracy_2m: 0.988063
  position_error_m: 0.352659
  position_error_median_m: 0.251662
  position_error_std_m: 0.355166
  x_error_m: 0.126115
  y_error_m: 0.294282
  z_error_m: 0.049609
artifacts:
  log: .training_queue/logs/1781739570290942725_6440_canon_both.log
  job: .training_queue/done/1781739570290942725_6440_canon_both.job
  output_dir: ''
parents: [run-i520-canon-none]
relations: []
tags: [plcs, canonical, split-trunk]
---

## 考察 / Findings

canonical の**位置・回転の両パスを分離**した構成。#520 内で `ang_error 15.91°` と角度は最良だが、
`position_error 0.353m` と baseline (0.273m) より**位置は悪化**。

→ **canonical のパス分離は角度に効くが位置を犠牲にするトレードオフ。** 単純な全分離より、角度に
効く分離と位置補助を両立させる設計（cf. #521 `run-i521-ex10-vel`）が有望。

