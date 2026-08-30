---
id: run-i801-dref-pose-beta005-inact05-s42
type: run
title: PLCS tracking + pose inactive weight 0.5（seed 42）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-29'
status: done
config:
  model: track_query_ablation_d_v2_selector
  architecture: track_query_ablation_d
  loss: tracking_all_outputs_beta01_reprojection
  data: plcs/multi_object_camera_view_v2
  rotation_weight: 0.05
  angle_weight: 0.05
  canonical_pose_weight: 1.0
  reprojection_weight: 1.0
  presence_inactive_weight: 0.5
  match_presence_weight: 0.5
  seed: 42
  seq_len: 128
  num_views: 6
  batch_size: 8
  accumulate_grad_batches: 4
  epochs: 75
  precision: bf16-mixed
  cswa_backend: cuda
metrics:
  loss: 1.212074
  loss_position: 0.160322
  loss_rotation: 0.281908
  loss_presence: 0.92196
  loss_track_smoothness: 0.0
  loss_angle: 0.30477
  loss_canonical_pose: 0.010332
  loss_reprojection: 0.090127
  position_error: 0.46381
  presence_precision: 0.500696
  presence_recall: 0.997971
  presence_f1: 0.662426
  lifecycle_presence_f1: 0.662426
  birth_frame_error: 12.843446
  death_frame_error: 15.593406
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 44.400002
  id_switches: 44.400002
  duplicate_active_tracks: 307.200012
  missed_gt_frames: 2.96
  inactive_query_false_positives: 1394.880005
  angular_error_deg: 36.555664
  heading_error_deg: 36.220001
  position_error_m: 5.2616
  x_error_m: 1.909063
  y_error_m: 4.51625
  z_error_m: 0.268516
  y_sign_accuracy: 0.657656
  reference_index_0_position_error_m: 6.457237
  reference_index_1_position_error_m: 5.664583
  reference_index_2_position_error_m: 5.269531
  reference_index_3_position_error_m: 5.368304
  reference_index_4_position_error_m: 5.327446
  canonical_mpjpe_m: 0.183633
  world_mpjpe_m: 5.273126
  reprojection_error_px: 159.306885
  behind_camera_fraction: 0.0
  reference_index_5_position_error_m: 5.11632
repro:
  commit: 9ffae3d34d34fe93b41a4f7e64e60238b3254cd4
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_tracking_pose model=track_query_ablation_d_v2_selector
    court_keypoints=camera_view_v2 model.cswa.backend=cuda paths.data_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=plcs/multi_object_camera_view_v2 'data.seq_len_range=[128,128]'
    'data.num_views_range=[6,6]' data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=camera_2
    loss.presence_inactive_weight=0.5 training.compile.enabled=false training.trainer.precision=bf16-mixed
    training.trainer.accumulate_grad_batches=4 training.trainer.max_epochs=75 training.trainer.check_val_every_n_epoch=5
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.seed=42 run.fast_dev_run=false run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_inact05_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-inact05-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-inact05-s42/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_inact05_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-a2-plcs-d-reference
  rel: compares
tags:
- plcs
- tracking
- canonical-pose
- reprojection
- camera-view-v2
- presence
- inactive-weight
- beta005
- seed-42
---

## 考察 / Findings

### 要約

親 run の `presence_inactive_weight` だけを `0.25→0.5` に上げたが、precision / F1 と
pose・位置が悪化し、inactive false positive も減らなかった。duplicate と ID switch は僅かに
改善したものの既存 tracking baseline には届かず、この重みをさらに上げる根拠は得られなかった。

### アーキテクチャ詳細

モデル、camera-view-v2 data、T=128 / V=6、effective batch=32、seed 42、
`rotation_weight=angle_weight=0.05`、canonical / reprojection weight `1.0` は親 run と同一。
変更点は `presence_inactive_weight=0.5` と最大 epoch `75` のみである。presence weight は
最終 BCE だけでなく Hungarian の presence cost にも使われるため、重み変更は assignment にも影響する。

### メトリクスの解釈

test precision / recall / F1 は `0.5007 / 0.9980 / 0.6624`、inactive-query false positive
`1394.88`、duplicate `307.20`、ID switch `44.40` だった。position `5.2616m`、heading
`36.2200deg`、canonical MPJPE `0.1836m`、reprojection `159.31px` である。

test total `1.21207` の約 76% を presence が占め、angle の重み込み寄与は約 1.26% に留まる。
best val loss は epoch 34 の `1.05775` で、最終 `1.18751` は 12.3% 悪化した。train presence
が低下する一方で val presence は上昇し、親 run より早い過学習が観測された。

### アーキテクチャ⇄メトリクスの因果考察

inactive weight を上げれば false positive が単調に減るという仮説は棄却された。実装上、同じ
inactive weight が Hungarian presence cost にも渡されるため、discrete な query-target assignment
が変化し、BCE の calibration 効果と identity 対応の効果が混ざる。これは仮説だが、正例・transition の
相対寄与低下と assignment の揺れが、recall をほぼ 1.0 に保ったまま汎化を悪化させた可能性がある。

### 既存実験との比較

親 run に対し duplicate は `357.60→307.20`、ID switch は `46.24→44.40` と僅かに改善した。
一方、F1 は `0.6639→0.6624`、inactive false positive は `1361.92→1394.88`、position は
`5.0050→5.2616m`、heading は `32.98→36.22deg` と悪化した。tracking-only の
`run-i801-a2-plcs-d-reference` に対しても F1、ID switch、duplicate、position、heading の総合で劣る。

### 次に有効な実験

inactive weight は `0.25` に戻す。presence を assignment から切り離すため
`match_presence_weight=0` を一変数として検証し、position / rotation matching だけで identity を
固定寄りにする。保存 logits の threshold calibration も別途評価し、学習 loss と inference threshold の
効果を混同しない。
