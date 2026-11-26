---
id: run-i2520-v2-m1-mono-gvhmr-htd-e100-s42
type: run
task: plcs
sequence: 127
recorded_at: '2026-11-25'
title: i2520_v2_m1_mono_gvhmr_htd_e100_s42
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
    run.seed=42 run.output_dir=plcs/i2520_v2_m1_mono_gvhmr_htd_e100_s42
artifacts:
  run_dir: knowledge/runs/run-i2520-v2-m1-mono-gvhmr-htd-e100-s42
  log: .training_queue/logs/0-8204408698086470602_2026-11-25T07-21-41-913529397Z_1486577_i2520_v2_m1_mono_gvhmr_htd_e100_s42.log
  output_dir: outputs/plcs/i2520_v2_m1_mono_gvhmr_htd_e100_s42/logs/version_0
  curves: knowledge/runs/run-i2520-v2-m1-mono-gvhmr-htd-e100-s42/curves.png
  tb_logdir: outputs/plcs/i2520_v2_m1_mono_gvhmr_htd_e100_s42/logs/version_0
parents:
- run-i2520-v2-m0-mono-gvhmr-base-e100-s42
- run-i2520-v1-m1-mono-htd-collapse-e100-s42
relations:
- to: run-i2520-v1-m1-mono-htd-collapse-e100-s42
  rel: retries
  note: v1の深度崩壊を引き起こした加速度整合損失重みを0.0へ修正したv2バランス設定での再実行
- to: run-i2520-v2-m0-mono-gvhmr-base-e100-s42
  rel: compares
  note: Stage A PVA-Heads（速度・加速度・接地確率予測）追加の有無を比較（M1 vs M0）
papers:
- paper-2024-gvhmr
- paper-2026-htd-refine
tags:
- plcs
- monocular-gvhmr
- htd-refine
- stage-a-pva
- torsion-collinearity
- i2475
- i2520
---

## 考察 / Findings

### 実験目的と設定
- 単眼GVHMR基本形（M0）に HTD-Refine Stage A の **`PVA-Heads`**（3D速度・3D加速度・足接地確率の同時予測、`pva_velocity=0.5, pva_acceleration=0.1, static_contact=0.25`）を追加した **M1**（`monocular_gvhmr_htd`, 43.6Mパラメータ）を、v2共通損失設定で学習した（`single_object_mixed`, `max_epochs=100`, `seed=42`）。
- 論文参照: `paper-2024-gvhmr`（バックボーン・CLIFF表現）および `paper-2026-htd-refine`（Stage A PVA予測ヘッド）。

### 観測（Epoch 68 途中停止と収束チェックポイント）
- 学習は収束プラトー到達後（best checkpoint: **`plcs-epoch=59.ckpt`**、M0のbest Epoch 64・A2のbest Epoch 59と一致する窓）の **Epoch 68 step 30/50** で以下により異常終了したため、`status: failed` および `metrics: {}`（`trainer.test`未達）として正直に記録する:
  - 終了直前ログ: `train/loss=1.630, train/position_error_m=1.180, val/loss=2.000, val/position_error_m=1.310, val/angular_error_deg=21.90, val/canonical_mpjpe_m=0.261`
  - 例外: `src/tasks/plcs/training/losses.py:271` にて `ValueError: pred torsion_angle: degenerate/invalid skeleton features at indices [[904, 1]]`（ミニバッチ中 1 フレームのみ、上肢の完全伸展に伴う3点共線により捻れ角の法線ベクトルが退化した）。
- 保存済み best checkpoint（`plcs-epoch=59.ckpt`）を用いた共通オフライン横断評価（`run-i2520-cross-eval-and-htd-stage-b`）では全フレームで `skeleton_valid_rate = 100.0%` であり、以下の結果を得た:
  - 合成 `mixed`: `pos=0.940 m`, `ang=17.82°`, `PA-MPJPE=0.0915 m`, `jerk=20,472 m/s³`（M0比で実測位置誤差はやや増 `+0.10 m`、PA-MPJPEとjerkは小幅改善）。
  - 実動画 `meiji_4cam`（6クリップ）: M0比で **`LOVO再投影 885.8 → 612.6 px`（-30.8%）**、**`cross-camera 3D 6.44 → 5.83 m`（-9.5%）**、**`vs DLT MPJPE 4.131 → 3.718 m`（-10.0%）**、**`jerk 198,069 → 163,336 m/s³`（-17.5%）** と全主要指標で一貫して改善した。

### 考察と課題
- **仮説**: Stage A の速度・加速度・接地確率マルチタスク教師学習は、合成データ単体の位置回帰より **未見の実カメラドメインにおける時間的一貫性と深度安定化（正則化効果）** に強く寄与している。加えて Stage B（`PLCSHTDRefiner`）への高品質なPVA事前分布を供給する基盤となる（`run-i2520-cross-eval-and-htd-stage-b` で Stage B 併用の大幅改善を確認）。
- **実装上の課題**: `losses.py:271` の `pred.require_valid` は、身体的に妥当な完全伸展姿勢（180°直線の腕）での `torsion_angle` 共線をハードクラッシュ扱いにしてしまう（`A1` でも Epoch 55 に同一インデックス1で再発、一方 `torsion_angle_weight=0.0` の `A3` では100エポック無停止）。共線フレームをGT側と同じ有効マスクで安全に除外するか、微小正則化で学習継続できるよう後続PRで修正が必要である。
- `summary_impact: none` の理由: 単体runの記録であり、評価結果に基づく判断更新は `run-i2520-cross-eval-and-htd-stage-b` と `group-i2520-monocular-gvhmr-htd` で実施する。
