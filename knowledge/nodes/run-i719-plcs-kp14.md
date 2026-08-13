---
id: run-i719-plcs-kp14
type: run
title: i719-plcs-kp14
issue: 719
provider: codex
session: 019ff6e5-b975-7671-bdd4-f03843734cf5
date: '2026-08-13'
status: done
config:
  model: track_query_kp14
metrics:
  loss: 1.120139
  loss_position: 0.243619
  loss_rotation: 0.397805
  loss_presence: 0.677618
  loss_track_smoothness: 0.0
  position_error: 1.134778
  presence_precision: 0.307692
  presence_recall: 0.5
  presence_f1: 0.380952
  lifecycle_presence_f1: 0.380952
  birth_frame_error: 32.0
  death_frame_error: 32.0
  query_reuse_count: 0.0
  illegal_overlap_count: 0.0
  segment_id_switches: 0.0
  id_switches: 0.0
  duplicate_active_tracks: 0.0
  missed_gt_frames: 256.0
  inactive_query_false_positives: 576.0
  position_mae_x_m: 3.388205
  position_mae_y_m: 7.104686
  position_mae_z_m: 0.610223
  y_sign_accuracy: 0.375
  source_frame_position_mae_y_m: 7.104686
  source_frame_heading_error_deg: 45.087933
  reference_positive_position_mae_y_m: 11.006618
  reference_negative_position_mae_y_m: 4.763526
  angular_error_deg: 45.087929
  reference_consistency_y_m: 8.8125
  reference_consistency_heading_deg: 99.1884
repro:
  commit: 64c2da16e029acd2202c3b592e31eb9952c916d7
  branch: feat/issue-719-courtkp7-reference
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_tracking model=track_query_kp14
    data.scene_dir=issue719/plcs data.seq_len_range='[64,64]' data.num_views_range='[3,5]'
    data.batch_size=8 data.num_workers=4 training.trainer.max_epochs=3 training.trainer.check_val_every_n_epoch=1
    training.warmup_steps=0 training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.seed=719 run.output_dir=issue719/i719-plcs-kp14 run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i719-plcs-kp14
  predictions: knowledge/runs/run-i719-plcs-kp14/pred_test.npz
  log: .training_queue/logs/1786553727736504404_2144393_i719-plcs-kp14.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue719/i719-plcs-kp14/logs/version_2
  curves: knowledge/runs/run-i719-plcs-kp14/curves.png
  tb_logdir: outputs/issue719/i719-plcs-kp14/logs/version_2
parents: []
relations: []
tags:
- plcs
- tracking
- court-kp14
- reference-orientation
- issue-719
---

## 考察 / Findings

### 要約

PLCS ordered-information baseline。3 epochで `position_mae_y_m=7.104686`、`y_sign_accuracy=0.375000`、`angular_error_deg=45.087929`、`position_error=1.134778` を得た。

### アーキテクチャ詳細

ordered CourtKP14を用い、KP7系と同じreference-oriented position/heading targetとreference roleを受ける。player-local canonical poseは反転せず、court-space positionとheadingだけを同一signで変換する。

### メトリクスの解釈

paired counterfactualは `reference_consistency_y_m=8.812500`、`reference_consistency_heading_deg=99.188400`。通常のtarget errorは `source_frame_position_mae_y_m=7.104686` と `source_frame_heading_error_deg=45.087933` に分離されている。

### アーキテクチャ⇄メトリクスの因果考察

ordered geometryが存在しても3 epochのposition/heading同時学習ではorientationが十分に収束せず、reference変更に対する出力整合も低かった。

### 既存実験との比較

KP7 no-referenceはY MAE、Y sign、heading error、paired consistencyの全てで本baselineを上回る。KP7 referenceはtotal position errorのみ小さいが、Y・heading指標は悪い。

### 次に有効な実験

長期学習とpresence matchingの安定化後に、ordered情報上限としてのreference counterfactual整合が改善するか再評価する。
