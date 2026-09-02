---
id: run-court-align-b00-eval-aug-pilot-appearance-v1
type: run
title: court-align-b00-eval-aug-pilot-appearance-v1
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN KP14 + center vote
  loss: focal heatmap + masked Smooth L1 center vote
  data: B00 ground-UV real heatmap / b00_appearance_v1 checkpoint
metrics:
  predicted_instance_count: 2
  instance_tp: 1
  instance_fp: 1
  instance_fn: 1
  instance_precision: 0.5
  instance_recall: 0.5
  instance_f1: 0.5
  instance_kp_mean_error_px: 270.624847
  instance_center_mean_error_px: 0.593527
  instance_count_accuracy: 1.0
  sim2_rotation_error_deg: 90.016239
  sim2_scale_relative_error: 0.503995
  sim2_translation_error_px: 180.422647
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
    real_evaluation.preprocess.content_fraction=1.0 real_evaluation.checkpoint_path=court_alignment/augmentation_pilot/b00_appearance_v1_warm20_s42/logs/version_0/checkpoints/court-alignment-epoch\\=019.ckpt
    real_evaluation.output_dir=court_alignment/real_heatmap/aug_pilot_appearance_v1
artifacts:
  run_dir: knowledge/runs/run-court-align-b00-eval-aug-pilot-appearance-v1
  predictions: knowledge/runs/run-court-align-b00-eval-aug-pilot-appearance-v1/pred_test.npz
parents: [run-court-align-aug-pilot-appearance-v1]
relations:
- {to: group-court-align-b00-real-heatmap-eval, rel: compares}
tags: [court-alignment, kp14, augmentation, b00, real-heatmap]
---

## 考察 / Findings

### 要約

appearance augmentation checkpointはB00でF1=0.5まで改善した。2 instances中1 instanceを正しく対応し、今回の4条件で最有望だった。

### アーキテクチャ詳細

入力は32視点平均のB00 heatmap、decoder threshold=0.25、max preprocess、σ=2.0。参照はaccepted alignmentで独立GTではない。

### メトリクスの解釈

TP/FP/FN=1/1/1、KP誤差=270.62px、matched center誤差=0.59px、count accuracy=1.0、Sim(2)回転=90.02°、scale相対誤差=0.504、translation=180.42px。semantic countは7/5、推定centerは(121.54,181.21)/(101.01,173.35)px、fit scaleは7.224/2.284px/mだった。

### アーキテクチャ⇄メトリクスの因果考察

線幅・blur・振幅変動への露出が1面の対応を成立させ、centerは高精度だった。もう1面は部分線/ghost lineの影響が残ったと考えられる（仮説）。

### 既存実験との比較

identity modelとscale単独のB00 F1=0から改善した。structure単独もF1=0.5だが、appearanceのmatched center=0.59pxがstructureの3.86pxより良い。

### 次に有効な実験

appearanceを基礎に、structure要素を低確率・低強度で追加する。
