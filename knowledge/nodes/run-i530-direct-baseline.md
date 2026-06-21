---
id: run-i530-direct-baseline
type: run
title: direct canonical head baseline（EX10再評価）
issue: 530
provider: codex
date: '2026-06-19'
status: done
config:
  model: multiview_axial_base_split_auxpos
  loss: canonical_rot
  data: multiview_sequence
  seed: 42
  epochs: 100
  batch_size: 6
  canonical_pose_head: direct
metrics:
  ang_error_deg: 9.644101
  angle_accuracy: 0.813211
  angle_accuracy_10deg: 0.653333
  angle_accuracy_15deg: 0.813211
  angle_accuracy_30deg: 0.954862
  angular_error_deg: 9.63505
  angular_error_median_deg: 6.929528
  angular_error_std_deg: 9.793705
  canonical_bone_length_relative_error: 0.04287
  canonical_joint_angle_error_deg: 13.650642
  canonical_joint_angle_velocity_error_deg: 0.557987
  canonical_mpjpe_m: 0.138357
  canonical_torsion_error_deg: 17.872805
  canonical_torsion_velocity_error_deg: 0.857327
  canonical_torso_twist_error_deg: 10.434834
  canonical_torso_twist_velocity_error_deg: 0.60572
  loss: 0.073955
  loss_canonical_pose: 0.005669
  pos_error_m: 0.247902
  position_accuracy: 0.923976
  position_accuracy_0.5m: 0.923976
  position_accuracy_1m: 0.991437
  position_accuracy_2m: 1.0
  position_error_m: 0.246616
  position_error_median_m: 0.206561
  position_error_std_m: 0.175092
  x_error_m: 0.106773
  y_error_m: 0.19657
  z_error_m: 0.044684
artifacts:
  log: .claude/worktrees/issue-530-canonical-pose-head/outputs/plcs/issue_530/direct_baseline_eval.log
  checkpoint: outputs/plcs/plcs_multiview_axial/logs/version_15/checkpoints/last.ckpt
  output_dir: outputs/plcs/plcs_multiview_axial/logs/version_15
  curves: knowledge/runs/run-i530-direct-baseline/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_15
parents:
- run-i518-exp10
relations: []
tags:
- plcs
- canonical
- direct-head
- baseline
- split-trunk
---

## 考察 / Findings

### 要約
EX10 の direct canonical pose head を追加指標で再評価した #530 の比較基準。下流の yaw・位置は構造化 3 案より最良。

### アーキテクチャ詳細
`multiview_axial_base_split_auxpos` + `canonical_rot`、`canonical_pose_head=direct`（canonical pose を直接回帰）。`seed=42`, `epochs=100`, `batch_size=6`。

### メトリクスの解釈
yaw 誤差 `9.64°`、位置誤差 `0.247m` と最良。canonical MPJPE と骨長誤差は構造化 head 3 案よりわずかに劣る。

### アーキテクチャ⇄メトリクスの因果考察
canonical pose を直接回帰する自由度が、最終的な向き・位置推定には依然有効。構造制約を課す 3 案より下流表現が強い。

### 既存実験との比較
親 [[run-i518-exp10]] の head を評価軸を変えて再測定。#530 の 3 実行（[[run-i530-bone-direction]] / [[run-i530-mean-residual]] / [[run-i530-parent-delta]]）はこのノードを親として比較する。

### 次に有効な実験
構造化 head は direct の置換ではなく、補助 head や bounded residual として使う方向を検討（[[run-i530-bone-direction]] が候補）。
