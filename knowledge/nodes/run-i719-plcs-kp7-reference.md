---
id: run-i719-plcs-kp7-reference
type: run
title: i719-plcs-kp7-reference-benchmark
issue: 719
provider: codex
session: 019ff682-9749-74d2-ad8c-7a39e815dcd5
date: '2026-08-13'
status: done
config:
  model: track_query_kp7_reference
metrics:
  loss: 1.313485
  loss_position: 0.12523
  loss_rotation: 1.1385
  loss_presence: 0.619005
  loss_track_smoothness: 0.0
  position_error: 0.81152
  presence_precision: 0.25
  presence_recall: 0.875
  presence_f1: 0.388889
  lifecycle_presence_f1: 0.388889
  birth_frame_error: 8.0
  death_frame_error: 8.0
  query_reuse_count: 0.0
  illegal_overlap_count: 0.0
  segment_id_switches: 0.0
  id_switches: 0.0
  duplicate_active_tracks: 0.0
  missed_gt_frames: 64.0
  inactive_query_false_positives: 1344.0
  position_mae_x_m: 2.283655
  position_mae_y_m: 6.264746
  position_mae_z_m: 0.201551
  y_sign_accuracy: 0.375
  source_frame_position_mae_y_m: 6.264745
  source_frame_heading_error_deg: 98.846252
  reference_positive_position_mae_y_m: 8.313441
  reference_negative_position_mae_y_m: 5.035528
  angular_error_deg: 98.846252
  reference_consistency_y_m: 6.3125
  reference_consistency_heading_deg: 52.187069
  court_peak_fusion_latency_ms: 3.21601
  court_peak_fusion_peak_memory_mb: 43.975586
repro:
  commit: 6daaebce341c02acbc1fd3948093afad9508bcb7
  branch: feat/issue-719-courtkp7-reference
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_tracking model=track_query_kp7_reference
    data.scene_dir=issue719/plcs data.seq_len_range='[64,64]' data.num_views_range='[3,5]'
    data.batch_size=8 data.num_workers=4 training.trainer.max_epochs=3 training.trainer.check_val_every_n_epoch=1
    training.warmup_steps=0 training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.seed=719 run.output_dir=issue719/i719-plcs-kp7-reference-benchmark run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i719-plcs-kp7-reference
  predictions: knowledge/runs/run-i719-plcs-kp7-reference/pred_test.npz
  log: .training_queue/logs/1786605179178483245_418273_i719-plcs-kp7-reference-benchmark.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue719/i719-plcs-kp7-reference-benchmark/logs/version_0
  curves: knowledge/runs/run-i719-plcs-kp7-reference/curves.png
  tb_logdir: outputs/issue719/i719-plcs-kp7-reference-benchmark/logs/version_0
parents:
- run-i719-plcs-kp7-no-reference
relations:
- to: run-i719-plcs-kp7-no-reference
  rel: compares
- to: run-i719-plcs-kp14
  rel: compares
tags:
- plcs
- tracking
- court-kp7
- reference-conditioned
- issue-719
---

## 考察 / Findings

### 要約

unordered CourtKP7 set tokenへ1 viewだけのreference deltaを加えたPLCS主条件。3 epochで `position_mae_y_m=6.264746`、`y_sign_accuracy=0.375000`、`angular_error_deg=98.846252`、`position_error=0.811520` を得た。

### アーキテクチャ詳細

visibility-aware player anchorとfull-pose featureをshared unordered CourtKP7 encoderへ入力し、set aggregation後だけreference roleを加える。player-local canonical poseは反転せず、court-space position/headingだけをreference signへ変換する。

### メトリクスの解釈

no-referenceよりY MAEが2.396617 m大きく、Y sign accuracyが0.375000低く、heading errorが69.439167°大きい。paired Y consistencyも2.515625 m悪化した。paired heading consistencyは52.187069°だった。

### アーキテクチャ⇄メトリクスの因果考察

total position errorとX/Z MAEは改善したが、reference roleはlimited runでY orientationとheadingの学習を安定化できなかった。Issueのproduction採用条件を明確に満たさない。

### 既存実験との比較

KP14に対してtotal position errorは0.323258小さいが、Y MAE・Y sign・heading error・paired Y consistencyはいずれも劣る。no-referenceが主要orientation指標で最良だった。

### 次に有効な実験

heading warmup、reference-side balanced split、複数seed・長期学習を組み合わせ、Y MAE・Y sign・heading consistencyの全てでno-referenceを上回ることを再評価条件とする。

C=7/N=4/D=4、B=1/V=3/T=64、10 warmups/50 repeatsでactual PLCS fusionを計測し、`court_peak_fusion_latency_ms=3.216010`、`court_peak_fusion_peak_memory_mb=43.975586` を得た。
