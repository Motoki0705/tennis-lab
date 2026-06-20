---
id: run-i530-parent-delta
type: run
title: 親相対デルタ canonical head
issue: 530
provider: codex
date: '2026-06-19'
status: done
config:
  model: multiview_axial_issue530_parent_delta
  loss: canonical_rot
  data: multiview_sequence
  seed: 42
  epochs: 100
  batch_size: 6
  canonical_pose_head: parent_delta
metrics:
  ang_error_deg: 13.392245
  angle_accuracy: 0.681283
  angle_accuracy_10deg: 0.466909
  angle_accuracy_15deg: 0.681283
  angle_accuracy_30deg: 0.910194
  angular_error_deg: 13.317651
  angular_error_median_deg: 10.694895
  angular_error_std_deg: 12.493397
  canonical_bone_length_relative_error: 0.046092
  canonical_joint_angle_error_deg: 13.373359
  canonical_joint_angle_velocity_error_deg: 0.564414
  canonical_mpjpe_m: 0.136663
  canonical_torsion_error_deg: 17.870905
  canonical_torsion_velocity_error_deg: 0.894244
  canonical_torso_twist_error_deg: 10.350979
  canonical_torso_twist_velocity_error_deg: 0.588317
  loss: 0.104966
  loss_canonical_pose: 0.00527
  pos_error_m: 0.287231
  position_accuracy: 0.888387
  position_accuracy_0.5m: 0.888387
  position_accuracy_1m: 0.973035
  position_accuracy_2m: 0.99959
  position_error_m: 0.285171
  position_error_median_m: 0.207584
  position_error_std_m: 0.25007
  x_error_m: 0.104231
  y_error_m: 0.236752
  z_error_m: 0.040583
artifacts:
  log: .training_queue/logs/1781796308914256071_295586_i530_parent_delta.log
  job: .training_queue/done/1781796308914256071_295586_i530_parent_delta.job
  output_dir: outputs/plcs/issue_530/parent_delta
  curves: knowledge/runs/run-i530-parent-delta/curves.png
  tb_logdir: .claude/worktrees/issue-530-canonical-pose-head/outputs/plcs/issue_530/parent_delta/logs/version_0
parents:
- run-i530-direct-baseline
relations: []
tags:
- plcs
- canonical
- structured-head
- parent-delta
- split-trunk
---

## 考察 / Findings

### 要約
各関節を親関節からの相対デルタとして復号。canonical MPJPE は微改善だが骨長誤差・下流とも悪化。単独 head 化の優先度は低い。

### アーキテクチャ詳細
`multiview_axial_issue530_parent_delta` + `canonical_rot`、`canonical_pose_head=parent_delta`。`seed=42`, `epochs=100`, `batch_size=6`。

### メトリクスの解釈
canonical MPJPE は baseline より約 `1.2%` 改善するが、骨長相対誤差 `4.29%→4.61%` と悪化、yaw 誤差 `13.32°`、位置誤差 `0.285m` と下流性能も低下。

### アーキテクチャ⇄メトリクスの因果考察
局所オフセット回帰だけでは骨格制約にならず、親からの累積復号も利点を示さなかった（累積誤差が骨長に乗る）。

### 既存実験との比較
親 [[run-i530-direct-baseline]] に対し下流悪化。#530 構造化 3 案では [[run-i530-bone-direction]] が最良で、本案は最下位。

### 次に有効な実験
この形式を単独 head として進める優先度は低い。
