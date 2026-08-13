---
id: run-i719-plcs-kp7-no-reference
type: run
title: i719-plcs-kp7-no-reference
issue: 719
provider: codex
session: 019ff6e5-b975-7671-bdd4-f03843734cf5
date: '2026-08-13'
status: done
config:
  model: track_query_kp7_no_reference
metrics:
  loss: 0.932461
  loss_position: 0.165851
  loss_rotation: 0.16712
  loss_presence: 0.68305
  loss_track_smoothness: 0.0
  position_error: 0.944179
  presence_precision: 0.238095
  presence_recall: 0.625
  presence_f1: 0.344828
  lifecycle_presence_f1: 0.344828
  birth_frame_error: 24.0
  death_frame_error: 24.0
  query_reuse_count: 0.0
  illegal_overlap_count: 0.0
  segment_id_switches: 0.0
  id_switches: 0.0
  duplicate_active_tracks: 0.0
  missed_gt_frames: 192.0
  inactive_query_false_positives: 1024.0
  position_mae_x_m: 3.555042
  position_mae_y_m: 3.868129
  position_mae_z_m: 0.289151
  y_sign_accuracy: 0.75
  source_frame_position_mae_y_m: 3.868129
  source_frame_heading_error_deg: 29.407087
  reference_positive_position_mae_y_m: 4.729281
  reference_negative_position_mae_y_m: 3.351437
  angular_error_deg: 29.407085
  reference_consistency_y_m: 3.796875
  reference_consistency_heading_deg: 45.651726
repro:
  commit: 64c2da16e029acd2202c3b592e31eb9952c916d7
  branch: feat/issue-719-courtkp7-reference
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_tracking model=track_query_kp7_no_reference
    data.scene_dir=issue719/plcs data.seq_len_range='[64,64]' data.num_views_range='[3,5]'
    data.batch_size=8 data.num_workers=4 training.trainer.max_epochs=3 training.trainer.check_val_every_n_epoch=1
    training.warmup_steps=0 training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.seed=719 run.output_dir=issue719/i719-plcs-kp7-no-reference run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i719-plcs-kp7-no-reference
  predictions: knowledge/runs/run-i719-plcs-kp7-no-reference/pred_test.npz
  log: .training_queue/logs/1786553727847511793_2144429_i719-plcs-kp7-no-reference.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue719/i719-plcs-kp7-no-reference/logs/version_2
  curves: knowledge/runs/run-i719-plcs-kp7-no-reference/curves.png
  tb_logdir: outputs/issue719/i719-plcs-kp7-no-reference/logs/version_2
parents:
- run-i719-plcs-kp14
relations:
- to: run-i719-plcs-kp14
  rel: compares
tags:
- plcs
- tracking
- court-kp7
- no-reference
- issue-719
---

## 考察 / Findings

### 要約

unordered CourtKP7を使いreference value deltaを無効化したPLCS対照条件。3 epochで `position_mae_y_m=3.868129`、`y_sign_accuracy=0.750000`、`angular_error_deg=29.407085`、`position_error=0.944179` を得た。

### アーキテクチャ詳細

visibility-aware player anchorとfull 2D pose featureをobject queryとし、CourtKP7 peak setへcross-attentionする。target position/headingはreference camera基準だが、どのviewがreferenceかを示すlearned deltaは入力しない。

### メトリクスの解釈

Y MAE、Y sign、heading errorはPLCS 3条件中で最良だった。paired `reference_consistency_y_m=3.796875` と `reference_consistency_heading_deg=45.651726` も他2条件より小さい。

### アーキテクチャ⇄メトリクスの因果考察

reference signalなしでY signが高いため、固定splitのcourt/object geometryやtarget-side分布のshortcut、または短期学習の分散が混在した可能性がある。

### 既存実験との比較

KP7 referenceよりY MAEが2.396617 m小さく、Y sign accuracyが0.375000高く、heading errorが69.439167°小さい。

### 次に有効な実験

reference sideとview orderを均衡化した複数seed counterfactual testで、no-referenceの優位がshortcutか再現可能な最適化差かを判定する。
