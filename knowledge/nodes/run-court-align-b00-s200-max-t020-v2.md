---
id: run-court-align-b00-s200-max-t020-v2
type: run
title: court-align-b00-s200-max-t020-v2
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN（4-down U-Net、KP14 heatmap + 2ch center vote、sigma=2.0 checkpoint）
  loss: inference-only（decoder threshold 0.20）
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
    real_evaluation.preprocess.method=max decoder.threshold=0.20 real_evaluation.preprocess.content_fraction=1.0
    real_evaluation.output_dir=court_alignment/real_heatmap/b00_s200_max_t020_v2
artifacts:
  run_dir: knowledge/runs/run-court-align-b00-s200-max-t020-v2
  predictions: knowledge/runs/run-court-align-b00-s200-max-t020-v2/pred_test.npz
  log: .training_queue/logs/1788305344336393362_177528_court-align-b00-s200-max-t020-v2.log
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
- threshold-020
- inference
---

## 考察 / Findings

### 要約

threshold 0.20でもTP=0、FP=2、FN=2、F1=0で、accepted alignmentの2面にはmatchしなかった。referenceは独立 ground truth ではない。

### アーキテクチャ詳細

sigma=2.0 checkpoint、max resize、content fraction 1.0を固定し、32/48 views集約の999x908 heatmapを256x256へ入力した。primaryとの差はthreshold 0.20だけで、学習scale範囲は3--6 px/m、referenceは7.1668 px/mである。

### メトリクスの解釈

raw semantic countは10/6、scaleは0.02471/0.03170 px/m、coverage pass rate 0.5、match pair 0だった。50.342 m / 180 deg / relative scale 1.0は未match penaltyであり連続誤差ではない。

### アーキテクチャ⇄メトリクスの因果考察

観測としてthreshold 0.05--0.20の結果は完全に同じで、0.25でも1 semantic peakしか変わらない。仮説として、正解peakのscore cutoffではなくsynthetic-to-realのline appearanceとtopology差が支配している。

### 既存実験との比較

0.05/0.10/0.15と同値で、親0.25もF1 0である。synthetic sigma=2.0のF1 0.99737をthreshold調整だけでは移送できなかった。

### 次に有効な実験

scale range拡張、morphological線幅/blur、line dropout、spurious・parallel・ghost lines、probability amplitude/noise、view dropoutを優先する。scale normalization単独のnegative resultも再現したため、decoder tuningよりaugmentationが必要である。
