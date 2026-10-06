---
id: run-i1032-c-recon-consistency-s42-v1
type: run
task: ball_refiner_3d
sequence: 14
recorded_at: '2026-10-07'
title: 物理head＋パラメータ教師＋積分再構成・整合性（勾配は直接出力側）
issue: 1032
provider: claude
session: 72300f25-ca01-4859-9c8e-1ce0c340f9c3
date: '2026-10-07'
status: done
config:
  model:
    dimensions: 3
    architecture: regression
    width: 256
    layers: 8
    heads: 4
    dropout: 0.05
    flow_steps: 16
    ffn_dim: 704
    ffn_type: swiglu
    rope_dim: 64
    rope_theta: 10000.0
    physics_heads: true
  loss:
    position: normalized L1
    event: soft cross-entropy
    position_weight: 1.0
    event_weight: 1.0
    physics:
      field_weight: 1.0
      segment_weight: 1.0
      reconstruction_weight: 1.0
      consistency_weight: 1.0
      consistency_gradient: direct
  data:
    dataset: ball_refiner/single_object (ball_physics.v1, 1,280 rallies, 60fps)
    window:
      length: 128
      long_probability: 0.25
      max_length: 1024
    augmentation: event_only (event gaps 50%, no noise)
    inference: whole clip, one forward
metrics:
  best_step: 4000.0
  checkpoint_step: 4000.0
  inference_ms_per_frame: 0.289617
  test_acceleration_rmse_mps2: 353.218984
  test_event_brier: 0.007533
  test_event_f1: 0.926339
  test_event_recall: 0.888651
  test_event_rmse_m: 0.391389
  test_event_soft_ce: 0.100602
  test_fit_residual_missing_rmse_m: 0.249048
  test_fit_residual_rmse_m: 0.133652
  test_frame_missing_rate: 0.111831
  test_implausible_acceleration_rate: 0.631256
  test_integrated_fit_residual_rmse_m: 0.155404
  test_integrated_missing_rmse_m: 1.374298
  test_integrated_rmse_m: 1.222837
  test_integrated_truth_segments_rmse_m: 0.581091
  test_jerk_ratio: 4844.201194
  test_k_drag_relative_error_median: 0.277779
  test_missing_rmse_m: 0.539214
  test_rmse_m: 0.257223
  test_segment_velocity_error_median_mps: 1.008811
  test_segmentation_failure_rate: 0.695312
  test_surface_accuracy: 0.960938
  test_velocity_rmse_mps: 5.125955
  test_wind_error_median_mps: 1.629562
repro:
  commit: 5251d2f693dc7cabe5b766a5073b8585f6b98b2b
  branch: feat/issue-1032-p5-physics-heads
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.tasks.ball_refiner_3d.scripts.train
    paths.data_root=/home/kamimura/projects/tennis-lab/data paths.output_root=/home/kamimura/projects/tennis-lab/outputs
    run.output_dir=ball_refiner_3d/train/i1032-physics/c-recon-consistency-s42-v1
    model.physics_heads=true loss.physics.field_weight=1.0 loss.physics.segment_weight=1.0
    loss.physics.reconstruction_weight=1.0 loss.physics.consistency_weight=1.0 loss.physics.consistency_gradient=direct
artifacts:
  run_dir: knowledge/runs/run-i1032-c-recon-consistency-s42-v1
  predictions: knowledge/runs/run-i1032-c-recon-consistency-s42-v1/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791305838832827675_219894_i1032-c-recon-consistency-s42-v1.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/ball_refiner_3d/train/i1032-physics/c-recon-consistency-s42-v1/logs/version_0
parents:
- group-i1032-physics-heads-s42
relations:
- to: run-i1032-b-params-s42-v1
  rel: compares
papers: []
tags:
- coordinate-refiner
- 3d
- physics
---

## 要約

Bに、積分軌道とGTの再構成L1、積分軌道（detach）へ直接出力を寄せる整合性L1を加えた。GT区間分割での積分軌道RMSEは1.107m→0.581m（欠損0.709m）と半減し、予測区間分割では1.223m。直接座標は全体0.257m・欠損0.539mで、対照Aより悪い。

## メトリクスの解釈

整合性項は直接出力を滑らかにしなかった：直接出力の加速度RMSE 353m/s²・非物理率0.631（Bは447・0.655、Aは184・0.406）。積分軌道は区間内では物理的（GT区間で加速度RMSE 2.35m/s²）だが、予測区間分割では区間境界のずれが段差となり、GT区間で測る加速度RMSEは631m/s²。surface正解率0.961、場の誤差はBと同程度（風1.63m/s、k_drag 0.28）。学習時間はA/Bの約2.3倍（2,543秒）、最大割当6.5GB。

## 比較の限界

3条件は同じcommit・データ・評価劣化（イベント欠損50%、ノイズなし）・seed42・4,000更新・batch32で、物理指標は `physics_eval.v1`（GT残差0を確認済みのLM当てはめ）で測った。単一seed・合成データのみで、実動画・追加seedでの再現性は未確認。
validation位置RMSEのbestはいずれも最終step4,000で、収束は断定しない。lossの係数は実測値の同等化ではなく、field lossは最後まで約1.0（surface CEを含む）と他項より大きい。
