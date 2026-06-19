---
id: run-i530-mean-residual
type: run
title: 平均ポーズ残差 canonical head
issue: 530
provider: codex
date: '2026-06-19'
status: done
config:
  model: multiview_axial_issue530_mean_residual
  loss: canonical_rot
  data: multiview_sequence
  seed: 42
  epochs: 100
  batch_size: 6
  canonical_pose_head: mean_residual
metrics:
  ang_error_deg: 13.556933
  angle_accuracy: 0.695121
  angle_accuracy_10deg: 0.510882
  angle_accuracy_15deg: 0.695121
  angle_accuracy_30deg: 0.902878
  angular_error_deg: 13.433054
  angular_error_median_deg: 9.719762
  angular_error_std_deg: 12.751104
  canonical_bone_length_relative_error: 0.04189
  canonical_joint_angle_error_deg: 13.415669
  canonical_joint_angle_velocity_error_deg: 0.560293
  canonical_mpjpe_m: 0.136166
  canonical_torsion_error_deg: 17.933571
  canonical_torsion_velocity_error_deg: 0.881391
  canonical_torso_twist_error_deg: 10.742508
  canonical_torso_twist_velocity_error_deg: 0.598423
  loss: 0.10977
  loss_canonical_pose: 0.005408
  pos_error_m: 0.296788
  position_accuracy: 0.855503
  position_accuracy_0.5m: 0.855503
  position_accuracy_1m: 0.968681
  position_accuracy_2m: 1.0
  position_error_m: 0.298285
  position_error_median_m: 0.23672
  position_error_std_m: 0.237558
  x_error_m: 0.110421
  y_error_m: 0.255003
  z_error_m: 0.042162
artifacts:
  log: .training_queue/logs/1781796308855797848_295569_i530_mean_residual.log
  job: .training_queue/done/1781796308855797848_295569_i530_mean_residual.job
  output_dir: outputs/plcs/issue_530/mean_residual
parents:
- run-i530-direct-baseline
relations: []
tags:
- plcs
- canonical
- structured-head
- mean-residual
- split-trunk
---

## 考察 / Findings

平均ポーズに残差を加える再パラメータ化。direct baseline に対して canonical
MPJPE は `0.1384 m` から `0.1362 m` へ約 `1.6%` 改善したが、yaw 誤差は
`9.64°` から `13.43°`、位置誤差は `0.247 m` から `0.298 m` へ悪化した。

平均姿勢による初期バイアスだけでは有効な構造制約にならず、下流表現を弱めた。
単独 head として採用する根拠はない。
