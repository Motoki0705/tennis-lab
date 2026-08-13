---
id: run-i719-blcs-kp14
type: run
title: i719-blcs-kp14
issue: 719
provider: codex
session: 019ff6e5-b975-7671-bdd4-f03843734cf5
date: '2026-08-13'
status: done
config:
  model: track_query_kp14
metrics:
  loss: 1.026375
  loss_position: 0.46969
  loss_position_x: 0.210457
  loss_position_y: 0.195029
  loss_position_z: 1.537476
  loss_presence: 0.556686
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 2.295817
  presence_precision: 0.373811
  presence_recall: 0.998047
  presence_f1: 0.543906
  lifecycle_presence_f1: 0.543906
  birth_frame_error: 0.0
  death_frame_error: 1.5
  query_reuse_count: 0.0
  illegal_overlap_count: 0.0
  segment_id_switches: 24.0
  id_switches: 24.0
  duplicate_active_tracks: 0.0
  missed_gt_frames: 1.0
  inactive_query_false_positives: 856.0
  position_mae_x_m: 2.938131
  position_mae_y_m: 5.89394
  position_mae_z_m: 2.093181
  y_sign_accuracy: 0.669922
  source_frame_position_mae_y_m: 5.89394
  reference_positive_position_mae_y_m: 7.160789
  reference_negative_position_mae_y_m: 5.471657
  reference_oriented_position_mae_y_m: 5.89394
  reference_consistency_y_m: 3.40625
repro:
  commit: 64c2da16e029acd2202c3b592e31eb9952c916d7
  branch: feat/issue-719-courtkp7-reference
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_tracking model=track_query_kp14
    data.scene_dir=issue719/blcs data.seq_len_range='[64,64]' data.num_views_range='[3,5]'
    data.batch_size=8 data.num_workers=4 training.trainer.max_epochs=3 training.trainer.check_val_every_n_epoch=1
    training.warmup_steps=0 training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.seed=719 run.output_dir=issue719/i719-blcs-kp14 run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i719-blcs-kp14
  predictions: knowledge/runs/run-i719-blcs-kp14/pred_test.npz
  log: .training_queue/logs/1786553727507583924_2144317_i719-blcs-kp14.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue719/i719-blcs-kp14/logs/version_3
  curves: knowledge/runs/run-i719-blcs-kp14/curves.png
  tb_logdir: outputs/issue719/i719-blcs-kp14/logs/version_3
parents: []
relations: []
tags:
- blcs
- tracking
- court-kp14
- reference-orientation
- issue-719
---

## 考察 / Findings

### 要約

Issue #719 の ordered-information baseline。seed 719、固定 32/8/8 scene split、T=64、batch 8、3 epoch で完走し、test の `position_mae_y_m=5.893940`、`y_sign_accuracy=0.669922`、`position_error=2.295817`、`lifecycle_presence_f1=0.543906` を得た。

### アーキテクチャ詳細

`track_query_kp14` は ordered CourtKP14 の point-fusion を使う。target と reference role は KP7 条件と同じ reference-camera 座標契約に従うため、physical point order を既知とした上限比較である。

### メトリクスの解釈

Y MAE と Y sign accuracy は refreshed BLCS 3条件中で最良だった。paired reference counterfactual の `reference_consistency_y_m=3.406250` は、通常の target error である `source_frame_position_mae_y_m=5.893940` と分離して測定されている。

### アーキテクチャ⇄メトリクスの因果考察

ordered physical point情報がnear/far ambiguityを直接除く設計と、Y指標の優位は整合する。ただし単一seed・3 epochの観測であり、因果を確定するものではない。

### 既存実験との比較

KP7 no-referenceよりY MAEが0.296168 m小さく、Y sign accuracyが0.031250高い。KP7 referenceに対してはY MAEが1.454248 m小さく、Y sign accuracyが0.236328高い。

### 次に有効な実験

複数seed・長期学習でもordered baselineの優位が残るかを確認し、KP7 reference側のfailure modeをreference-side strataごとに分解する。
