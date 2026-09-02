---
id: run-court-align-b00-s200-max-t025-v2
type: run
title: court-align-b00-s200-max-t025-v2
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN（4-down U-Net、KP14 heatmap + 2ch center vote、sigma=2.0 checkpoint）
  loss: inference-only（decoder threshold 0.25）
  data: B00 accepted ground-UV mean-probability heatmap（max resize、content fraction
    1.0）
  input_shape: 999x908 -> 256x256
  aggregate_views: 32
  training_scale_range_px_per_m:
  - 3.0
  - 6.0
metrics:
  predicted_instance_count: 2
  reference_instance_count: 2
  instance_tp: 0
  instance_fp: 2
  instance_fn: 2
  instance_f1: 0.0
  matched_instance_count: 0
  coverage_gate_pass_rate: 0.5
  sim2_pair_count: 0
  sim2_translation_error_m: 50.34199788790197
  sim2_rotation_error_deg: 180.0
  sim2_scale_relative_error: 1.0
repro:
  commit: a3861691b1954779fd3dc2ab754cc99313018994
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.evaluate_real_heatmap paths.data_root=/home/kamimura/projects/tennis-lab/.tmp/b00-alignment-eval
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    paths.output_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    real_evaluation.archive_path=synthetic_data_generation/scenes/B00/alignment/line-heatmaps/heatmaps.npz
    real_evaluation.manifest_path=synthetic_data_generation/scenes/B00/alignment/line-heatmaps/manifest.json
    real_evaluation.alignment_path=synthetic_data_generation/scenes/B00/alignment/alignment.json
    real_evaluation.device=cuda real_evaluation.checkpoint_path=court_alignment/ablation_sigma_200/logs/version_0/checkpoints/court-alignment-epoch\\=047.ckpt
    real_evaluation.preprocess.method=max decoder.threshold=0.25 real_evaluation.preprocess.content_fraction=1.0
    real_evaluation.output_dir=court_alignment/real_heatmap/b00_s200_max_t025_v2
artifacts:
  run_dir: knowledge/runs/run-court-align-b00-s200-max-t025-v2
  predictions: knowledge/runs/run-court-align-b00-s200-max-t025-v2/pred_test.npz
  log: .training_queue/logs/1788305344107401016_177370_court-align-b00-s200-max-t025-v2.log
parents:
- run-court-align-kp14-ablation-sigma-200
relations:
- to: run-i618-b00-ground-line-court-fit-v1
  rel: compares
- to: run-i618-b00-alignment-calibration-local-v2
  rel: compares
- to: run-i618-b00-alignment-holdout-v1
  rel: compares
tags:
- court-alignment
- kp14
- multi-court
- ground-uv
- real-heatmap
- b00
- sigma-200
- inference
---

## 考察 / Findings

### 要約

synthetic test で最良だった sigma=2.0 checkpoint を B00 の実 ground-UV heatmap に適用したが、2 面を出力しても accepted alignment の2面とは1件も対応せず、TP=0、FP=2、FN=2、F1=0 だった。accepted alignment は比較用 reference であり、独立 ground truth ではない。

### アーキテクチャ詳細

4-down U-Net の KP14 heatmap + center-vote 出力を用いる inference-only run である。B00 の 48 views 中32 viewsを集約した `mean_probability` raster（999x908）を max resize と letterbox で256x256へ変換し、threshold 0.25、最大2 instanceで decode した。学習時の court scale は3--6 px/mだが、本入力の reference scale は7.1668 px/mで範囲外だった。

### メトリクスの解釈

raw prediction は2面だが、semantic keypoint 数は9/6、Sim(2) scale は0.02462/0.03170 px/mで、reference 7.1668 px/mから大きく外れた。対応 pair が0のため、50.342 m / 180 deg / relative scale 1.0 は未match時の penalty であり、連続 pose 誤差として解釈してはならない。coverage gate は4候補中2候補だけが通った。

### アーキテクチャ⇄メトリクスの因果考察

観測として、instance count だけは2で一致した一方、14 semantic peaksとcourt scaleを復元できなかった。仮説として、reference scaleの学習範囲外に加え、実heatmapの線幅・blur、欠落線、平行/ghost line、確率振幅の分布がbinary procedural maskと異なり、局所peakをcourt全体へ誤結合した可能性がある。

### 既存実験との比較

親の synthetic sigma=2.0 run は F1 0.99737、KP 0.82098 pxだったが、本runはF1 0でありsynthetic-to-real domain gapが支配的である。既存B00 system baselineはline-template fit/calibrationの比較対象で、holdoutは棄却済みである。本reference自体がそのaccepted alignmentに依存するため、本runだけで絶対精度やsystem超越は主張できない。

### 次に有効な実験

scale範囲を7.2 px/m超まで拡張し、morphologicalな線幅変動/blur、line dropout、spurious・parallel・ghost lines、probability amplitude/noise、view dropoutを独立に追加する。content fractionだけを変えるscale normalizationは別runでF1を改善しなかったため、単独解として扱わない。
