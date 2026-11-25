---
id: run-i2520-v2-m0-mono-gvhmr-base-e100-s42
type: run
task: plcs
sequence: 126
recorded_at: '2026-11-25'
title: i2520_v2_m0_mono_gvhmr_base_e100_s42
issue: 2520
provider: gemini
session: ec99afa3-0ddf-4af9-b92f-9e3912cd1273
date: '2026-11-25'
status: done
summary_impact: none
config:
  model: monocular_gvhmr_base
  loss: monocular_base
  data: monocular_sequence
metrics:
  position_error_m: 1.165295
  angular_error_deg: 17.625
  position_accuracy_0.5m: 0.368529
  angle_accuracy_15deg: 0.510695
  canonical_mpjpe_m: 0.251696
  canonical_pck_0.1m: 0.107388
repro:
  commit: d08f191cc66ac3a50980bcd2a19de4f7c40d2551
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=monocular_gvhmr_base loss=monocular_base data=monocular_sequence data.scene_dir=plcs/single_object_mixed
    loss.world_kp_3d_weight=2.0 loss.position_weight=2.0 loss.rotation_weight=0.02
    loss.angle_weight=0.0 loss.reprojection_weight=0.5 loss.joint_angle_weight=0.1
    loss.torsion_angle_weight=0.1 loss.torso_twist_weight=0.1 loss.bone_length_weight=0.5
    loss.kinematic_velocity_consistency_weight=0.0 loss.kinematic_acceleration_consistency_weight=0.0
    training.trainer.max_epochs=100 training.warmup_steps=500 training.trainer.check_val_every_n_epoch=5
    training.compile.enabled=false training.checkpoint.save_top_k=1 training.checkpoint.save_last=true
    run.seed=42 run.output_dir=plcs/i2520_v2_m0_mono_gvhmr_base_e100_s42
artifacts:
  run_dir: knowledge/runs/run-i2520-v2-m0-mono-gvhmr-base-e100-s42
  log: .training_queue/logs/0-8204408698370037614_2026-11-25T07-21-41-629962385Z_1486490_i2520_v2_m0_mono_gvhmr_base_e100_s42.log
  output_dir: outputs/plcs/i2520_v2_m0_mono_gvhmr_base_e100_s42/logs/version_0
  curves: knowledge/runs/run-i2520-v2-m0-mono-gvhmr-base-e100-s42/curves.png
  tb_logdir: outputs/plcs/i2520_v2_m0_mono_gvhmr_base_e100_s42/logs/version_0
parents:
- run-i2520-b0-multiview-axial-e100-s42
- run-i2520-v1-m1-mono-htd-collapse-e100-s42
relations:
- to: run-i2520-b0-multiview-axial-e100-s42
  rel: compares
  note: 同一データ・100エポック・seed 42での多視点参照基準（B0）と単眼GVHMR基本形（M0）の比較
papers:
- paper-2024-gvhmr
tags:
- plcs
- monocular-gvhmr
- baseline
- i2475
- i2520
---

## 考察 / Findings

### 実験目的と設定
- 単眼GVHMR型PLCSの基本構成（**M0**: `monocular_gvhmr_base`, 40.0Mパラメータ, PVA-Headsなし）を、v1診断で見つかった損失バランス問題（`run-i2520-v1-m1-mono-htd-collapse-e100-s42`）を修正した **v2共通損失設定** で学習した。
- v2損失重み: `world_kp_3d=2.0`, `position=2.0`, `rotation=0.02`, `angle=0.0`, `reprojection=0.5`, `joint_angle=0.1`, `torsion_angle=0.1`, `torso_twist=0.1`, `bone_length=0.5`, `kinematic_velocity_consistency=0.0`, `kinematic_acceleration_consistency=0.0`。データ・予算は `single_object_mixed`, `max_epochs=100`, `warmup_steps=500`, `batch=16`, `seed=42`。
- 論文参照: `paper-2024-gvhmr` の Early Fusion（COCO17 + CLIFF bbox + カメラ内外パラメータ）＋12層RoPE Transformer構成をHMR2特徴なしで実装・検証した。

### 観測（test split / 学習完了後評価）
- `single_object_mixed` test split（`trainer.test`, best `plcs-epoch=64.ckpt`）:
  - `position_error_m = 1.1653 m`（`position_accuracy_0.5m = 36.85%`）
  - `angular_error_deg = 17.63°`（`angle_accuracy_15deg = 51.07%`）
  - `canonical_mpjpe_m = 0.2517 m`（`canonical_pck_0.1m = 10.74%`）
- 共通オフライン評価（`run-i2520-cross-eval-and-htd-stage-b`）での横断結果:
  - 合成3 split: `fixed` `pos=1.454 m / PA-MPJPE=0.123 m / jerk=14,398 m/s³`、`broadcast` `pos=1.910 m / PA-MPJPE=0.100 m / jerk=27,428 m/s³`、`mixed` `pos=0.840 m / PA-MPJPE=0.095 m / jerk=21,094 m/s³`。
  - 実動画 `meiji_4cam`（6クリップ）: `LOVO再投影=885.8 px`（中央値 `115.6 px`）、`vs DLT MPJPE=4.131 m`（中央値 `3.456 m`）、`vs DLT PA-MPJPE=0.123 m`、`jerk=198,069 m/s³`、DLT欠損392フレームの復元率 `100%`、CPU遅延 `1.05 ms/frame`。

### 考察と限界
- v2損失修正により深度崩壊と初期エポック退化が完全に解消し、単眼のみで `mixed` `0.84〜1.17 m` のコート定位、および多視点 `B0`（`41.8〜57.4°`）を大きく凌ぐ方位角精度（`15.5〜34.1°`）を達成した。
- 一方で、単眼予測のみ（M0）では高周波時間ジッタ（実動画 `jerk = 198,069 m/s³`）および未見実カメラでの光線深度誤差（`4.13 m`）が大きい。次段階として Stage A PVA-Heads（M1）および Stage B 軌道最適化（M2）の効果を検証する。
- `summary_impact: none` の理由: 単体runの記録であり、PLCS全体の判断更新は `run-i2520-cross-eval-and-htd-stage-b` および `group-i2520-monocular-gvhmr-htd` でまとめて行う。
