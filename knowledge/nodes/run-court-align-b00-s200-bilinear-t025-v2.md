---
id: run-court-align-b00-s200-bilinear-t025-v2
type: run
title: court-align-b00-s200-bilinear-t025-v2
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN（4-down U-Net、KP14 heatmap + 2ch center vote、sigma=2.0 checkpoint）
  loss: inference-only（decoder threshold 0.25）
  data: B00 accepted ground-UV mean-probability heatmap（bilinear resize、content fraction
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
  coverage_gate_pass_rate: 0.0
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
    real_evaluation.preprocess.method=bilinear decoder.threshold=0.25 real_evaluation.preprocess.content_fraction=1.0
    real_evaluation.output_dir=court_alignment/real_heatmap/b00_s200_bilinear_t025_v2
artifacts:
  run_dir: knowledge/runs/run-court-align-b00-s200-bilinear-t025-v2
  predictions: knowledge/runs/run-court-align-b00-s200-bilinear-t025-v2/pred_test.npz
  log: .training_queue/logs/1788305344184390971_177421_court-align-b00-s200-bilinear-t025-v2.log
parents:
- run-court-align-b00-s200-max-t025-v2
relations:
- to: run-court-align-kp14-ablation-sigma-200
  rel: compares
tags:
- court-alignment
- kp14
- multi-court
- ground-uv
- real-heatmap
- b00
- sigma-200
- bilinear
- inference
---

## 考察 / Findings

### 要約

bilinear resizeへ変更しても、2 predictionsに対しTP=0、FP=2、FN=2、F1=0だった。accepted alignmentは独立 ground truth ではなく、system-relative referenceである。

### アーキテクチャ詳細

sigma=2.0 checkpoint、threshold 0.25、content fraction 1.0を固定し、32/48 views集約の999x908 `mean_probability` rasterを256x256へ縮小する補間だけをmaxからbilinearへ変えた。学習scale範囲は3--6 px/m、referenceは7.1668 px/mである。

### メトリクスの解釈

raw semantic countは5/5、predicted scaleは0.004802/0.009532 px/mで、reference 7.1668 px/mから大きく外れた。coverage pass rateは0、match pairも0だった。50.342 m / 180 deg / relative scale 1.0は未match penaltyであり、連続誤差を表さない。

### アーキテクチャ⇄メトリクスの因果考察

観測として、bilinear平滑化はmax resizeよりsemantic peakを9/6から5/5へ減らし、raw scaleも小さくした。仮説として、細いcourt線の局所maximumを平均化して弱め、すでにあるblur/確率振幅のdomain gapを増幅した可能性がある。

### 既存実験との比較

親のmax runもF1 0だがcoverage 0.5、semantic 9/6だったため、bilinearは改善を示さない。synthetic sigma=2.0はF1 0.99737であり、resize方式だけではreal gapを埋められない。

### 次に有効な実験

補間方式の追加sweepより、scale range拡張、morphological線幅/blur、line dropout、spurious・parallel・ghost lines、probability amplitude/noise、view dropoutを学習augmentationとして導入する。別runでscale normalization単独も無効だった。
