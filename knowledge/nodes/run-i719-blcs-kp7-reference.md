---
id: run-i719-blcs-kp7-reference
type: run
title: i719-blcs-kp7-reference-benchmark
issue: 719
provider: codex
session: 019ff682-9749-74d2-ad8c-7a39e815dcd5
date: '2026-08-13'
status: done
config:
  model: track_query_kp7_reference
metrics:
  loss: 1.024383
  loss_position: 0.50035
  loss_position_x: 0.29285
  loss_position_y: 0.268764
  loss_position_z: 1.378522
  loss_presence: 0.524032
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 2.290416
  presence_precision: 0.801431
  presence_recall: 0.875
  presence_f1: 0.836601
  lifecycle_presence_f1: 0.836601
  birth_frame_error: 8.0
  death_frame_error: 8.0
  query_reuse_count: 0.0
  illegal_overlap_count: 0.0
  segment_id_switches: 1.0
  id_switches: 1.0
  duplicate_active_tracks: 0.0
  missed_gt_frames: 64.0
  inactive_query_false_positives: 111.0
  position_mae_x_m: 3.642925
  position_mae_y_m: 7.348188
  position_mae_z_m: 1.902952
  y_sign_accuracy: 0.433594
  source_frame_position_mae_y_m: 7.348188
  reference_positive_position_mae_y_m: 10.615898
  reference_negative_position_mae_y_m: 6.258952
  reference_oriented_position_mae_y_m: 7.348188
  reference_consistency_y_m: 4.09375
  court_peak_fusion_latency_ms: 4.065558
  court_peak_fusion_peak_memory_mb: 43.828125
repro:
  commit: 6daaebce341c02acbc1fd3948093afad9508bcb7
  branch: feat/issue-719-courtkp7-reference
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_tracking model=track_query_kp7_reference
    data.scene_dir=issue719/blcs data.seq_len_range='[64,64]' data.num_views_range='[3,5]'
    data.batch_size=8 data.num_workers=4 training.trainer.max_epochs=3 training.trainer.check_val_every_n_epoch=1
    training.warmup_steps=0 training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.seed=719 run.output_dir=issue719/i719-blcs-kp7-reference-benchmark run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i719-blcs-kp7-reference
  predictions: knowledge/runs/run-i719-blcs-kp7-reference/pred_test.npz
  log: .training_queue/logs/1786605179063381861_418238_i719-blcs-kp7-reference-benchmark.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue719/i719-blcs-kp7-reference-benchmark/logs/version_0
  curves: knowledge/runs/run-i719-blcs-kp7-reference/curves.png
  tb_logdir: outputs/issue719/i719-blcs-kp7-reference-benchmark/logs/version_0
parents:
- run-i719-blcs-kp7-no-reference
relations:
- to: run-i719-blcs-kp7-no-reference
  rel: compares
- to: run-i719-blcs-kp14
  rel: compares
tags:
- blcs
- tracking
- court-kp7
- reference-conditioned
- issue-719
---

## 考察 / Findings

### 要約

unordered CourtKP7 set aggregation後に選択された1 viewだけへreference deltaを加えるBLCS主条件。3 epochで `position_mae_y_m=7.348188`、`y_sign_accuracy=0.433594`、`position_error=2.290416` を得た。

### アーキテクチャ詳細

peak featureはnormalized UV、score、explicit covariance、relative UV、`E_class`だけを持つ。near/far、peak-index、final type embeddingを持たず、`[B,V,T,D,H]`を既存spatial/temporal track-query backboneへ渡す。

### メトリクスの解釈

no-referenceよりY MAEが1.158080 m大きく、Y sign accuracyが0.205078低い。paired `reference_consistency_y_m=4.093750` もno-referenceより2.828125 m大きい。一方、lifecycle F1は0.836601で3条件中最高だった。

### アーキテクチャ⇄メトリクスの因果考察

reference signalはpresence/lifecycle最適化には寄与した可能性があるが、limited budgetでは狙ったY orientation reasoningを改善しなかった。通常Y errorとpaired consistencyが両方悪化しており、production採用条件を満たさない。

### 既存実験との比較

KP14よりY MAEが1.454248 m大きく、Y sign accuracyが0.236328低い。total position errorはKP14より0.005401小さいが、Issue #719の主要Y指標では劣る。

### 次に有効な実験

reference-side balanced複数seed・長期学習を行い、Y MAEとY signの双方がno-referenceを上回らない限りproduction defaultへ昇格しない。

C=7/N=4/D=4、B=1/V=3/T=64、10 warmups/50 repeatsでactual BLCS fusionを計測し、`court_peak_fusion_latency_ms=4.065558`、`court_peak_fusion_peak_memory_mb=43.828125` を得た。
