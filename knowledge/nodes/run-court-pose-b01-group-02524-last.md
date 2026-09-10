---
id: run-court-pose-b01-group-02524-last
type: run
title: Court pose last.ckpt の B01 軌道 CPU 推論
provider: codex
date: '2026-09-11'
status: done
config:
  model: court_hierarchical + DINOv3 ViT-B/16 frozen backbone + LoRA rank 8 + 8-layer transformer + DPT large + pose10D
  loss: dense kp/seg/line + translation/rotation/focal pose loss; consistency disabled
  data: mixed B00 Synthetic Court V3 (4/batch) + TennisCourtDetector (4/batch); evaluated on B01 test group-02524
  checkpoint: last.ckpt
  checkpoint_sha256: 1eb9b4fe02336d472824fe68189f0cb406016b43b81102e1b22d43505f9200a8
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
  inference_seconds_mean: 0.4436689023877989
  peak_rss_bytes: 3009736704
repro:
  commit: dde8036f
  branch: codex/court-pose-predictor
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: bash knowledge/runs/run-court-pose-b01-group-02524-last/repro.sh
artifacts:
  run_dir: knowledge/runs/run-court-pose-b01-group-02524-last
  measurements: knowledge/runs/run-court-pose-b01-group-02524-last/measurements.json
  metrics_plot: knowledge/runs/run-court-pose-b01-group-02524-last/trajectory_metrics.png
  overlays: knowledge/runs/run-court-pose-b01-group-02524-last/trajectory_overlays.png
  runtime_config: knowledge/runs/run-court-pose-b01-group-02524-last/config.yaml
  source_checkpoint: outputs/court_detection/mixed-source/dense-pose-lora-b8-e20-s42/logs/version_0/checkpoints/last.ckpt
  log: .training_queue/logs/1789052279895515108_1993066_court_mixed_pose_lora_b8_e20_s42.log
parents:
- run-court-pose-b01-group-02524-epoch05
relations:
- to: run-court-pose-b01-group-02524-epoch05
  rel: confirms
tags:
- court-detection
- camera-pose
- b01
- cpu-inference
- checkpoint-equivalence
---

## 考察 / Findings

### 要約

指定された `last.ckpt` で B01 test 軌道 80 frame の CPU 推論を完走した。全精度指標が epoch 5 run と完全一致し、両 checkpoint の SHA-256 も一致したため、現時点の `last.ckpt` は epoch 5 と同一内容である。

### アーキテクチャ詳細

親 run と同じ DINOv3 ViT-B/16 frozen backbone、rank 8 LoRA、8-layer transformer、DPT large、pose10D head を使用した。入力軌道、前処理、CPU thread 数も親 run と同一で、checkpoint path だけを `court-detection-epoch=05.ckpt` から `last.ckpt` に変更した。

### メトリクスの解釈

translation mean 6.78 m、rotation mean 12.12 deg、focal relative error mean 50.4%、pose reprojection median 146.54 px であり、親 run と全て一致する。checkpoint load は 7.02 s、推論平均は 0.444 s/frame、peak RSS は 3,009,736,704 bytes だった。

### アーキテクチャ⇄メトリクスの因果考察

モデル出力と評価値の完全一致は推論の決定性だけでなく、両 checkpoint が同じ SHA-256 (`1eb9b4fe...f9200a8`) を持つことから説明できる。したがって `last.ckpt` という名前から学習後半の重みであると解釈してはならない。

### 既存実験との比較

親 `run-court-pose-b01-group-02524-epoch05` と精度は同一で、実質的なモデル比較にはなっていない。相違は checkpoint load 時間の実行時ばらつきだけである。

### 次に有効な実験

学習ジョブが新しい `last.ckpt` を保存した後に SHA-256 と更新日時を再確認し、親 run と異なることを確認してから同一軌道を再評価する。
