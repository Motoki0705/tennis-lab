---
id: run-i801-dref-pose-beta005-s42-r1
type: run
title: PLCS D tracking + pose beta005（seed 42）
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
  presence_inactive_weight: 0.25
  seed: 42
  seq_len: 128
  num_views: 6
  batch_size: 8
  accumulate_grad_batches: 4
  epochs: 100
  precision: bf16-mixed
  cswa_backend: cuda
metrics:
  loss: 0.909356
  loss_position: 0.150452
  loss_rotation: 0.241518
  loss_presence: 0.627003
  loss_track_smoothness: 0.0
  loss_angle: 0.258177
  loss_canonical_pose: 0.010398
  loss_reprojection: 0.096519
  position_error: 0.436962
  presence_precision: 0.504449
  presence_recall: 0.989816
  presence_f1: 0.66394
  lifecycle_presence_f1: 0.66394
  birth_frame_error: 13.48305
  death_frame_error: 15.425745
  query_reuse_count: 0.24
  illegal_overlap_count: 0.0
  segment_id_switches: 46.240002
  id_switches: 46.240002
  duplicate_active_tracks: 357.600006
  missed_gt_frames: 15.36
  inactive_query_false_positives: 1361.920044
  angular_error_deg: 33.487823
  heading_error_deg: 32.98
  position_error_m: 5.005014
  x_error_m: 1.824062
  y_error_m: 4.293125
  z_error_m: 0.248086
  y_sign_accuracy: 0.703125
  reference_index_0_position_error_m: 5.442434
  reference_index_1_position_error_m: 5.452083
  reference_index_2_position_error_m: 4.976562
  reference_index_3_position_error_m: 4.890625
  reference_index_4_position_error_m: 5.006793
  canonical_mpjpe_m: 0.179558
  world_mpjpe_m: 5.017619
  reprojection_error_px: 160.501266
  behind_camera_fraction: 9.6e-05
  reference_index_5_position_error_m: 4.663195
repro:
  commit: 9ffae3d34d34fe93b41a4f7e64e60238b3254cd4
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_tracking_pose model=track_query_ablation_d_v2_selector
    court_keypoints=camera_view_v2 model.cswa.backend=cuda paths.data_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=plcs/multi_object_camera_view_v2 'data.seq_len_range=[128,128]'
    'data.num_views_range=[6,6]' data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=camera_2
    training.compile.enabled=false training.trainer.precision=bf16-mixed training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    training.qualitative_logging.enabled=false run.seed=42 run.fast_dev_run=false
    run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_s42_r1
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-s42-r1
  predictions: knowledge/runs/run-i801-dref-pose-beta005-s42-r1/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_s42_r1/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-s42-r1/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_s42_r1/logs/version_0
parents:
- run-i801-a2-plcs-d-reference
relations:
- to: run-plcs-canonical-temporal-decomp-beta01-noaug
  rel: compares
- to: run-plcs-multiview-axial-all-outputs-beta01-reprojection-w1-v4-t128
  rel: compares
tags:
- plcs
- tracking
- canonical-pose
- reprojection
- camera-view-v2
- ablation-d
- beta005
- seed-42
---

## 考察 / Findings

### 要約

tracking query ごとの canonical pose と再投影を追加し、`rotation_weight=0.05`、
`angle_weight=0.05` で 100 epoch 学習した。angle は test total loss の約 1.4% に留まり
支配的ではなく、既存 D-reference に対して position / angular / heading は改善した。一方で
ID switch と duplicate track が大幅に増え、tracking + pose の両立には至らなかった。

### アーキテクチャ詳細

`track_query_ablation_d_v2_selector` の各 query に 17-joint canonical pose head を追加した。
Hungarian assignment 後の active track だけに canonical Smooth L1 を適用し、予測 position / rotation
で world pose に戻して clean 2D keypoints への multi-view reprojection を計算する。
tracking の position / presence / matching 設定は既存値を維持し、rotation と wrapped angle のみ
各 `0.05`、canonical と reprojection は各 `1.0` とした。T=128、V=6、effective batch=32、
camera-view-v2、seed 42、CUDA CSWA で学習した。

### メトリクスの解釈

test は position `5.0050m`、angular `33.4878deg`、heading `32.9800deg`、canonical MPJPE
`0.1796m`、reprojection `160.50px` だった。`behind_camera_fraction=0.000096` のため、
再投影誤差の主因はカメラ背後への崩壊ではない。test total `0.90936` に対し、重み込み寄与は
presence 約 69.0%、position 約 16.5%、reprojection 約 10.6%、angle 約 1.4%、rotation
約 1.3%、canonical 約 1.1% であり、angle 支配は解消されている。

tracking は precision / recall / F1 が `0.5044 / 0.9898 / 0.6639`、ID switch
`46.24`、duplicate active track `357.60` だった。最良 val loss は epoch 69 の `0.86680` で、
最終 val loss は `0.88749`。train loss が低下し続ける一方で epoch 70 前後から val が悪化し、
軽い過学習が見られた。

### アーキテクチャ⇄メトリクスの因果考察

pose / reprojection supervision により position と heading の表現は改善した可能性がある。一方、
これは仮説だが、共有 query trunk へ追加された pose 勾配が presence / identity の分離を助けず、
高 recall のまま inactive query を抑えられなかったため duplicate と ID switch が増えたと考える。
`presence_inactive_weight=0.25` に対し precision が約 0.50、recall が約 0.99 であることは、
次の律速が angle ではなく inactive-query false positive であるという観測と整合する。

### 既存実験との比較

親の `run-i801-a2-plcs-d-reference` に対し、position は `5.1151→5.0050m`（2.15%改善）、
angular は `35.4895→33.4878deg`（5.64%改善）、heading は
`35.7100→32.9800deg`（7.65%改善）した。一方、F1 は `0.6654→0.6639`、ID switch は
`24.76→46.24`、duplicate は `152.16→357.60` と悪化した。

canonical-only の `run-plcs-canonical-temporal-decomp-beta01-noaug` は MPJPE `0.091m` だが、
single-object / canonical-only 条件なので直接比較できない。標準 PLCS の reprojection run と同様、
再投影は角度・位置を改善し得る一方、tracking tail との trade-off が残った。

### 次に有効な実験

rotation / angle は各 `0.05` のまま、`presence_inactive_weight` を `0.25→0.5` に上げ、
precision、inactive-query false positive、duplicate、ID switch と recall の変化を測る。過学習を避けるため
最大 epoch は 75 とし、改善が不足し recall が維持される場合に `1.0` を続けて比較する。
