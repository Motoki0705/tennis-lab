---
id: run-court-align-b00-eval-aug-pilot-structure-v1
type: run
title: court-align-b00-eval-aug-pilot-structure-v1
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN KP14 + center vote
  loss: focal heatmap + masked Smooth L1 center vote
  data: B00 ground-UV real heatmap / b00_structure_v1 checkpoint
metrics:
  predicted_instance_count: 2
  instance_tp: 1
  instance_fp: 1
  instance_fn: 1
  instance_precision: 0.5
  instance_recall: 0.5
  instance_f1: 0.5
  instance_kp_mean_error_px: 258.182456
  instance_center_mean_error_px: 3.861267
  instance_count_accuracy: 1.0
  sim2_rotation_error_deg: 90.560356
  sim2_scale_relative_error: 0.505072
  sim2_translation_error_px: 180.895847
repro:
  commit: 5de3a7d45e037a799d4dad0ae1ef3ac3cb24897e
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.evaluate_real_heatmap paths.data_root=/home/kamimura/projects/tennis-lab/.tmp/b00-alignment-eval
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    paths.output_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    real_evaluation.archive_path=synthetic_data_generation/scenes/B00/alignment/line-heatmaps/heatmaps.npz
    real_evaluation.manifest_path=synthetic_data_generation/scenes/B00/alignment/line-heatmaps/manifest.json
    real_evaluation.alignment_path=synthetic_data_generation/scenes/B00/alignment/alignment.json
    real_evaluation.device=cuda real_evaluation.preprocess.method=max decoder.threshold=0.25
    real_evaluation.preprocess.content_fraction=1.0 real_evaluation.checkpoint_path=court_alignment/augmentation_pilot/b00_structure_v1_warm20_s42/logs/version_0/checkpoints/court-alignment-epoch\\=019.ckpt
    real_evaluation.output_dir=court_alignment/real_heatmap/aug_pilot_structure_v1
artifacts:
  run_dir: knowledge/runs/run-court-align-b00-eval-aug-pilot-structure-v1
  predictions: knowledge/runs/run-court-align-b00-eval-aug-pilot-structure-v1/pred_test.npz
parents: [run-court-align-aug-pilot-structure-v1]
relations:
- {to: group-court-align-b00-real-heatmap-eval, rel: compares}
tags: [court-alignment, kp14, augmentation, b00, real-heatmap]
---

## 考察 / Findings

### 要約

structure augmentation checkpointはB00でF1=0.5。1面を対応できたが、appearance単独よりcenter/KPの精度が低く、完全成功には届かなかった。

### アーキテクチャ詳細

入力は32視点平均のB00 heatmap、decoder threshold=0.25、max preprocess、σ=2.0。参照はaccepted alignmentで独立GTではない。

### メトリクスの解釈

TP/FP/FN=1/1/1、KP誤差=258.18px、matched center誤差=3.86px、count accuracy=1.0、Sim(2)回転=90.56°、scale相対誤差=0.505、translation=180.90px。semantic countは8/2、推定centerは(124.49,178.93)/(96.91,74.34)px、fit scaleは7.240/7.333px/mだった。

### アーキテクチャ⇄メトリクスの因果考察

partial crop/dropoutが実欠損に近く、1面の対応を可能にした。一方で強度が高いと線の幾何を壊し、center誤差を増やす可能性がある（仮説）。

### 既存実験との比較

identity modelとscale単独のB00 F1=0から改善したが、appearanceのmatched center=0.59px、semantic count=7/5に対してstructureは3.86px、8/2だった。

### 次に有効な実験

dropout/false-line強度を下げ、appearanceと段階的に組み合わせる。
