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
  curves: knowledge/runs/run-i530-bone-direction/curves.png
  tb_logdir: .claude/worktrees/issue-530-canonical-pose-head/outputs/plcs/issue_530/bone_direction/logs/version_0
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

### 要約
単位骨方向 + sequence 共有 scale から姿勢を復号。構造化 3 案では最良だが、下流の yaw・位置は direct baseline に届かず。

### アーキテクチャ詳細
`multiview_axial_issue530_bone_direction` + `canonical_rot`、`canonical_pose_head=bone_direction`。`seed=42`, `epochs=100`, `batch_size=6`。

### メトリクスの解釈
baseline 比で canonical MPJPE 約 `-3.0%`、骨長相対誤差 約 `-3.9%` 改善。一方 yaw 誤差 `+1.38°`、位置誤差 約 `+5mm` で baseline に届かず。head 側の骨長に最大 `-19.4%` の縮みがある。

### アーキテクチャ⇄メトリクスの因果考察
単位方向 + 共有 scale は骨格の内部整合（MPJPE / 骨長）を高めるが、単一 scale と固定骨長比の制約が強すぎて骨長が縮み、下流の yaw / 位置をわずかに毀損した可能性。

### 既存実験との比較
親 [[run-i530-direct-baseline]] に対し canonical 指標で改善・下流で微減。同 #530 の [[run-i530-mean-residual]] / [[run-i530-parent-delta]] より良好。

### 次に有効な実験
direct head の置換ではなく、補助 head または bounded residual 付き復号として使う価値がある。
