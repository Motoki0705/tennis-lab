---
id: run-i530-bone-direction
type: run
title: 骨方向 + sequence共有scale canonical head
issue: 530
provider: codex
date: '2026-06-19'
status: done
config:
  model: multiview_axial_issue530_bone_direction
  loss: canonical_rot
  data: multiview_sequence
  seed: 42
  epochs: 100
  batch_size: 6
  canonical_pose_head: bone_direction
metrics:
  ang_error_deg: 10.906995
  angle_accuracy: 0.775594
  angle_accuracy_10deg: 0.591056
  angle_accuracy_15deg: 0.775594
  angle_accuracy_30deg: 0.950802
  angular_error_deg: 11.012454
  angular_error_median_deg: 8.008836
  angular_error_std_deg: 13.110751
  canonical_bone_length_relative_error: 0.041216
  canonical_joint_angle_error_deg: 13.408106
  canonical_joint_angle_velocity_error_deg: 0.545578
  canonical_mpjpe_m: 0.134221
  canonical_torsion_error_deg: 17.908552
  canonical_torsion_velocity_error_deg: 0.873739
  canonical_torso_twist_error_deg: 10.690053
  canonical_torso_twist_velocity_error_deg: 0.616215
  loss: 0.086254
  loss_canonical_pose: 0.00516
  pos_error_m: 0.25115
  position_accuracy: 0.916408
  position_accuracy_0.5m: 0.916408
  position_accuracy_1m: 0.986255
  position_accuracy_2m: 1.0
  position_error_m: 0.251931
  position_error_median_m: 0.196969
  position_error_std_m: 0.204265
  x_error_m: 0.095119
  y_error_m: 0.205568
  z_error_m: 0.041543
artifacts:
  log: .training_queue/logs/1781796309429899347_295640_i530_bone_direction.log
  job: .training_queue/done/1781796309429899347_295640_i530_bone_direction.job
  output_dir: outputs/plcs/issue_530/bone_direction
parents:
- run-i530-direct-baseline
relations: []
tags:
- plcs
- canonical
- structured-head
- bone-direction
- split-trunk
---

## 考察 / Findings

単位骨方向と sequence 共有 scale から姿勢を復号する構成。構造化3案では最良で、
baseline 比で canonical MPJPE を約 `3.0%`、骨長相対誤差を約 `3.9%` 改善した。
一方、yaw 誤差は `+1.38°`、位置誤差は約 `+5 mm` で baseline に届かなかった。

学習された head 側の骨長には最大 `-19.4%` の縮みがあり、単一 scale と固定骨長比が
強すぎる可能性がある。direct head の置換ではなく、補助 head または bounded residual
付き復号として使う価値がある。
