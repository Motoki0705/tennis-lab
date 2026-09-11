---
id: run-court-pose-b01-group-02524-epoch05
type: run
title: Court pose epoch 5 の B01 軌道 CPU 推論
provider: codex
date: '2026-09-11'
status: done
config:
  model: court_hierarchical + DINOv3 ViT-B/16 frozen backbone + LoRA rank 8 + 8-layer transformer + DPT large + pose10D
  loss: dense kp/seg/line + translation/rotation/focal pose loss; consistency disabled
  data: mixed B00 Synthetic Court V3 (4/batch) + TennisCourtDetector (4/batch); evaluated on B01 test group-02524
  checkpoint: court-detection-epoch=05.ckpt
  device: cpu
metrics:
  sample_count: 80
  translation_error_m_mean: 6.7785996198654175
  translation_error_m_median: 6.034774541854858
  rotation_error_deg_mean: 12.124517463892698
  rotation_error_deg_median: 8.856838703155518
  focal_relative_error_mean: 0.5042255602777004
  focal_relative_error_median: 0.5090126395225525
  pose_reprojection_mean_px_mean: 415.01745586395265
  pose_reprojection_mean_px_median: 146.54092407226562
  dense_kp_mean_px_mean: 73.09424302577972
  dense_kp_mean_px_median: 51.13874626159668
  inference_seconds_mean: 0.4416888580253726
  peak_rss_bytes: 3009228800
repro:
  commit: 7f2d909c
  branch: codex/court-pose-predictor
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: bash knowledge/runs/run-court-pose-b01-group-02524-epoch05/repro.sh
artifacts:
  run_dir: knowledge/runs/run-court-pose-b01-group-02524-epoch05
  measurements: knowledge/runs/run-court-pose-b01-group-02524-epoch05/measurements.json
  metrics_plot: knowledge/runs/run-court-pose-b01-group-02524-epoch05/trajectory_metrics.png
  overlays: knowledge/runs/run-court-pose-b01-group-02524-epoch05/trajectory_overlays.png
  runtime_config: knowledge/runs/run-court-pose-b01-group-02524-epoch05/config.yaml
  source_checkpoint: outputs/court_detection/mixed-source/dense-pose-lora-b8-e20-s42/logs/version_0/checkpoints/court-detection-epoch=05.ckpt
  log: .training_queue/logs/1789052279895515108_1993066_court_mixed_pose_lora_b8_e20_s42.log
parents:
- run-court-mixed-pose-lora-b8-e20-s42
relations: []
tags:
- court-detection
- camera-pose
- b01
- cpu-inference
---

## 考察 / Findings

### 要約

学習中 run の epoch 5 中間 checkpoint は CPU で正常に復元でき、B01 の test 軌道 80 frame を完走した。ただし pose は平均 6.78 m / 12.12 deg、焦点距離相対誤差 50.4% で、実用精度には未到達である。

### アーキテクチャ詳細

DINOv3 ViT-B/16 の backbone 本体を frozen とし、rank 8 の LoRA、8-layer transformer、DPT large dense decoder、pose10D head を学習する構成。batch は B00 Synthetic Court V3 と TennisCourtDetector を 4:4 で混合し、pose 教師は synthetic sample のみが持つ。評価対象 B01 は学習 scene_ids に含まれず、scene 外汎化の実測である。

### メトリクスの解釈

80 frame の median は translation 6.03 m、rotation 8.86 deg、focal relative error 50.9%、pose reprojection 146.54 px だった。dense KP の median 51.14 px より pose reprojection が大きく、dense head が捉えた画像上のコート構造を pose head が十分な射影幾何へ変換できていない。CPU 実行は checkpoint load 3.42 s、推論平均 0.442 s/frame、peak RSS 3,009,228,800 bytes で完走した。

### アーキテクチャ⇄メトリクスの因果考察

観測として camera-center 軌跡は大まかな運動を再現せず、特に frame 69 付近で rotation error が約 140 deg に跳ねる。仮説として、epoch 5 では pose head の距離・焦点スケール同定が未収束であり、consistency loss が無効なため dense KP の比較的良い画像空間信号を pose head の幾何へ直接拘束できていないことが寄与している。

### 既存実験との比較

同一条件の既存 knowledge run は未登録のため、数値比較は行っていない。このノードを当該学習 run の epoch 5 baseline とする。

### 次に有効な実験

学習完了後の best / last checkpoint を同じ B01 group-02524 へ適用し、epoch 5 からの改善を同一指標で比較する。その後も dense KP と pose reprojection の乖離が残る場合は、pose consistency loss を有効化した run を比較対象にする。
