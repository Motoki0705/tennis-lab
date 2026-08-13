---
id: run-i719-blcs-kp7-no-reference
type: run
title: i719-blcs-kp7-no-reference
issue: 719
provider: codex
session: 019ff6e5-b975-7671-bdd4-f03843734cf5
date: '2026-08-13'
status: done
config:
  model: track_query_kp7_no_reference
metrics:
  loss: 1.109934
  loss_position: 0.492487
  loss_position_x: 0.231721
  loss_position_y: 0.221648
  loss_position_z: 1.555695
  loss_presence: 0.617447
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 2.361911
  presence_precision: 0.284507
  presence_recall: 0.986328
  presence_f1: 0.441627
  lifecycle_presence_f1: 0.441627
  birth_frame_error: 0.875
  death_frame_error: 0.0
  query_reuse_count: 0.0
  illegal_overlap_count: 0.0
  segment_id_switches: 26.0
  id_switches: 26.0
  duplicate_active_tracks: 0.0
  missed_gt_frames: 7.0
  inactive_query_false_positives: 1272.0
  position_mae_x_m: 2.863918
  position_mae_y_m: 6.190108
  position_mae_z_m: 2.102844
  y_sign_accuracy: 0.638672
  source_frame_position_mae_y_m: 6.190107
  reference_positive_position_mae_y_m: 9.145902
  reference_negative_position_mae_y_m: 5.204843
  reference_oriented_position_mae_y_m: 6.190107
  reference_consistency_y_m: 1.265625
repro:
  commit: 64c2da16e029acd2202c3b592e31eb9952c916d7
  branch: feat/issue-719-courtkp7-reference
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_tracking model=track_query_kp7_no_reference
    data.scene_dir=issue719/blcs data.seq_len_range='[64,64]' data.num_views_range='[3,5]'
    data.batch_size=8 data.num_workers=4 training.trainer.max_epochs=3 training.trainer.check_val_every_n_epoch=1
    training.warmup_steps=0 training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.seed=719 run.output_dir=issue719/i719-blcs-kp7-no-reference run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i719-blcs-kp7-no-reference
  predictions: knowledge/runs/run-i719-blcs-kp7-no-reference/pred_test.npz
  log: .training_queue/logs/1786553727587662402_2144353_i719-blcs-kp7-no-reference.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue719/i719-blcs-kp7-no-reference/logs/version_3
  curves: knowledge/runs/run-i719-blcs-kp7-no-reference/curves.png
  tb_logdir: outputs/issue719/i719-blcs-kp7-no-reference/logs/version_3
parents:
- run-i719-blcs-kp14
relations:
- to: run-i719-blcs-kp14
  rel: compares
tags:
- blcs
- tracking
- court-kp7
- no-reference
- issue-719
---

## 考察 / Findings

### 要約

unordered CourtKP7を使い、reference value deltaだけを無効化した対照条件。3 epochで `position_mae_y_m=6.190108`、`y_sign_accuracy=0.638672`、`position_error=2.361911` を得た。

### アーキテクチャ詳細

semantic class内のpeakを順序なし集合としてobject-query cross-attentionで集約する。target orientationは他条件と同じだが、選択reference viewを示すvalue-stream signalは入力しない。

### メトリクスの解釈

paired `reference_consistency_y_m=1.265625` はBLCS 3条件で最小だが、Y target errorはKP14より0.296168 m大きい。counterfactual reference indexは全sampleで元indexと異なり、通常誤差とは独立に再計算できる。

### アーキテクチャ⇄メトリクスの因果考察

reference signalなしでもY signが0.638672であるため、固定splitのgeometryやside分布からorientation shortcutを得た可能性がある。paired consistencyの小ささも、reference roleに応答しない対照設計と整合する。

### 既存実験との比較

KP7 referenceよりY MAEが1.158080 m小さく、Y sign accuracyが0.205078高く、paired Y consistencyも2.828125 m小さい。

### 次に有効な実験

reference sideとview orderを均衡化した複数seedでshortcutを検査し、no-referenceの高い符号正解率が再現するか確認する。
