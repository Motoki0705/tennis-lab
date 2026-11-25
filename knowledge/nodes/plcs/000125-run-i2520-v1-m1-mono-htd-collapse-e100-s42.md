---
id: run-i2520-v1-m1-mono-htd-collapse-e100-s42
type: run
task: plcs
sequence: 125
recorded_at: '2026-11-25'
title: i2520_m1_mono_gvhmr_htd_e100_s42
issue: 2520
provider: gemini
session: ec99afa3-0ddf-4af9-b92f-9e3912cd1273
date: '2026-11-25'
status: done
summary_impact: none
config:
  model: monocular_gvhmr_htd
  loss: monocular_htd
  data: monocular_sequence
metrics:
  position_error_m: 25.316055
  angular_error_deg: 9.25
  position_accuracy_0.5m: 0.0
  angle_accuracy_15deg: 0.804514
  canonical_mpjpe_m: 0.360643
  canonical_pck_0.1m: 0.018837
repro:
  commit: d08f191cc66ac3a50980bcd2a19de4f7c40d2551
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=monocular_gvhmr_htd data=monocular_sequence loss=monocular_htd data.scene_dir=plcs/single_object_mixed
    training.trainer.max_epochs=100 training.warmup_steps=500 training.trainer.check_val_every_n_epoch=5
    training.compile.enabled=false training.checkpoint.save_top_k=1 training.checkpoint.save_last=true
    run.output_dir=plcs/i2520_m1_mono_gvhmr_htd_e100_s42 run.seed=42
artifacts:
  run_dir: knowledge/runs/run-i2520-v1-m1-mono-htd-collapse-e100-s42
  log: .training_queue/logs/0-8204421196708458945_2026-11-25T03-53-23-291541054Z_24550_i2520_m1_mono_gvhmr_htd_e100_s42.log
  output_dir: outputs/plcs/i2520_m1_mono_gvhmr_htd_e100_s42/logs/version_0
  curves: knowledge/runs/run-i2520-v1-m1-mono-htd-collapse-e100-s42/curves.png
  tb_logdir: outputs/plcs/i2520_m1_mono_gvhmr_htd_e100_s42/logs/version_0
parents:
- run-i2520-b0-multiview-axial-e100-s42
relations: []
papers:
- paper-2024-gvhmr
- paper-2026-htd-refine
tags:
- plcs
- monocular-gvhmr
- htd-refine
- negative-result
- depth-collapse
- i2475
- i2520
---

## 考察 / Findings

### 実験目的と設定
- PR #2535 実装直後の既定設定 `model=monocular_gvhmr_htd` + `loss=monocular_htd` + `data=monocular_sequence`（`single_object_mixed`, `max_epochs=100`, `seed=42`, 約23分）をそのまま学習した初回実行（v1）。
- 論文参照: `paper-2024-gvhmr`（Early Fusion + RoPE + CLIFFカメラパラメータ化）および `paper-2026-htd-refine`（Stage A PVA-Heads と高次時間動態損失）の実装採用版。

### 観測（負の結果：CLIFF深度崩壊）
- 100エポック完走したが、test split（`single_object_mixed`）で **深度崩壊** を起こした:
  - `position_error_m = 25.3161 m`（`position_accuracy_0.5m = 0.0%`）
  - `angular_error_deg = 9.25°`（`angle_accuracy_15deg = 80.45%`）
  - `canonical_mpjpe_m = 0.3606 m`（`canonical_pck_0.1m = 1.88%`）
- 同じv1損失重みで実行した診断ランでも同一の深度崩壊が再現した（`A1` `25.318 m`、`A2` `25.497 m`）。一方、v1 `M0` は既定 `rotation_weight=1.0 + angle_weight=0.5` の過剰重みにより Epoch 1 step 23 で予測骨格の股関節近接による `pred joint_angle` 退化例外で停止した。

### 原因診断（実測に基づく特定）
- 学習バッチ上の損失項別内訳を実測したところ、原因は **`kinematic_acceleration_consistency_weight = 0.1` のスケール支配** であることを確認した:
  - 学習時の `monocular_bbox` 増強により、`bbx_s` がフレーム間で平均 `13.95%/frame` 変動する（val時は `0.25%/frame`）。
  - これにより 2階差分 `Δ²x · fps²`（`fps² = 900〜3600`）の無次元化されない加速度整合損失が **`8,915`（重み掛け後 `891.5` = 総損失 `952.6` の `93.6%`）** を占め、`world_kp_3d` 損失（`3.6`）を250倍上回った。
  - 最適化器は2階差分を0に近づける自明解として、CLIFF予測スケール `s` を `1.06 → 7〜10` へ押し上げ、カメラ深度 `tz = 2f / (s · bbx_s) → 0`（全選手をカメラレンズ直前へ潰す解）へ収束した。

### 次のアクションと教訓
- Stage A学習では PVA-Heads 自体をGT速度・加速度・接地確率で直接教師学習し、予測座標に対する明示的な2階差分ペナルティ（`kinematic_velocity_consistency_weight`, `kinematic_acceleration_consistency_weight`）は `0.0` とし、運動学整合は Stage B（`PLCSHTDRefiner`）で課す v2 損失バランスへ修正した（`run-i2520-v2-m1-mono-gvhmr-htd-e100-s42` へ再実行）。
- `src/tasks/plcs/configs/loss/monocular_htd.yaml` および `monocular_base.yaml` の既定重み（`kinematic_acceleration_consistency_weight=0.1`, `rotation_weight=1.0`, `angle_weight=0.5`）は後続コードPRで修正が必要である。
- `summary_impact: none` の理由: 既定損失重みの設計欠陥による負の診断結果であり、最終判断は修正後v2群を束ねた `group-i2520-monocular-gvhmr-htd` で反映する。
