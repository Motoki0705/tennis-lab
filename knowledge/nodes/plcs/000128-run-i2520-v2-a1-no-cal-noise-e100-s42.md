---
id: run-i2520-v2-a1-no-cal-noise-e100-s42
type: run
task: plcs
sequence: 128
recorded_at: '2026-11-25'
title: i2520_v2_a1_no_cal_noise_e100_s42
issue: 2520
provider: gemini
session: ec99afa3-0ddf-4af9-b92f-9e3912cd1273
date: '2026-11-25'
status: failed
summary_impact: none
config:
  model: monocular_gvhmr_htd
  loss: monocular_htd
  data: monocular_sequence
metrics: {}
repro:
  commit: 59faa32a28d2f54e6492abdba2429846fa0397e3
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=monocular_gvhmr_htd loss=monocular_htd data=monocular_sequence data.scene_dir=plcs/single_object_mixed
    loss.world_kp_3d_weight=2.0 loss.position_weight=2.0 loss.rotation_weight=0.02
    loss.angle_weight=0.0 loss.reprojection_weight=0.5 loss.joint_angle_weight=0.1
    loss.torsion_angle_weight=0.1 loss.torso_twist_weight=0.1 loss.bone_length_weight=0.5
    loss.kinematic_velocity_consistency_weight=0.0 loss.kinematic_acceleration_consistency_weight=0.0
    training.trainer.max_epochs=100 training.warmup_steps=500 training.trainer.check_val_every_n_epoch=5
    training.compile.enabled=false training.checkpoint.save_top_k=1 training.checkpoint.save_last=true
    run.seed=42 data.augmentation.camera_calibration_noise.enabled=false run.output_dir=plcs/i2520_v2_a1_no_cal_noise_e100_s42
artifacts:
  run_dir: knowledge/runs/run-i2520-v2-a1-no-cal-noise-e100-s42
  log: .training_queue/logs/0-8204408697809014220_2026-11-25T07-21-42-190985779Z_1486667_i2520_v2_a1_no_cal_noise_e100_s42.log
  output_dir: outputs/plcs/i2520_v2_a1_no_cal_noise_e100_s42/logs/version_0
  curves: knowledge/runs/run-i2520-v2-a1-no-cal-noise-e100-s42/curves.png
  tb_logdir: outputs/plcs/i2520_v2_a1_no_cal_noise_e100_s42/logs/version_0
parents:
- run-i2520-v2-m1-mono-gvhmr-htd-e100-s42
relations:
- to: run-i2520-v2-m1-mono-gvhmr-htd-e100-s42
  rel: compares
  note: 学習時のカメラ校正摂動（camera_calibration_noise）をOFFにしたアブレーション（A1 vs M1）
papers:
- paper-2024-gvhmr
- paper-2026-htd-refine
tags:
- plcs
- monocular-gvhmr
- htd-refine
- ablation
- calibration-noise
- torsion-collinearity
- i2475
- i2520
---

## 考察 / Findings

### 実験目的と設定
- M1（`run-i2520-v2-m1-mono-gvhmr-htd-e100-s42`）から学習時の **カメラ校正摂動増強のみを無効化**（`data.augmentation.camera_calibration_noise.enabled=false`）したアブレーション（**A1**）。他のモデル・損失・データ・seed条件は M1 と完全同一。

### 観測（Epoch 55 途中停止と横断評価）
- **Epoch 55 step 49/50** で M1 と同一箇所の 1 フレーム共線例外（`losses.py:271` `pred torsion_angle` インデックス `[[295, 1]]`）により停止したため、`status: failed` / `metrics: {}` とする。
  - 停止直前ログ: `train/loss=1.750, train/position_error_m=1.280, val/loss=2.110, val/position_error_m=1.440, val/angular_error_deg=18.00, val/canonical_mpjpe_m=0.244`
- 保存済み `last.ckpt`（Epoch 54）を用いた共通オフライン評価（`run-i2520-cross-eval-and-htd-stage-b`）では、**合成3 split・実動画すべてで M1（Epoch 59）より悪化**した:
  - 合成 `fixed`: 位置誤差 `1.894 m`（M1 `1.591 m` 比 **+19.0%悪化**）
  - 合成 `broadcast`: 位置誤差 `2.617 m`（M1 `2.035 m` 比 **+28.6%悪化**）
  - 合成 `mixed`: 位置誤差 `1.028 m`（M1 `0.940 m` 比 **+9.4%悪化**）
  - 実動画 `meiji_4cam`: `vs DLT MPJPE = 3.958 m`（M1 `3.718 m` 比 **+6.5%悪化**）、`LOVO = 646.6 px`（M1 `612.6 px` 比 +5.6%）、`jerk = 188,540 m/s³`（M1 `163,336 m/s³` 比 +15.4%）。

### 考察と限界
- 5エポックの学習差（Epoch 54 vs 59）がある点を差し引いても、特に `broadcast`（+28.6%）と `fixed`（+19.0%）での劣化幅が大きく、学習時のカメラ内外パラメータ摂動が **未知の焦点距離・カメラ配置・実動画校正誤差に対する汎化性能に強く寄与している** ことを支持する結果である。
- `summary_impact: none` の理由: アブレーション単体記録であり、全体判断は `group-i2520-monocular-gvhmr-htd` で統合する。
