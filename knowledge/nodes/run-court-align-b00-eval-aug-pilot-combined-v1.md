---
id: run-court-align-b00-eval-aug-pilot-combined-v1
type: run
title: court-align-b00-eval-aug-pilot-combined-v1
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN KP14 + center vote
  loss: focal heatmap + masked Smooth L1 center vote
  data: B00 ground-UV real heatmap / b00_v1 checkpoint
metrics:
  predicted_instance_count: 2
  instance_tp: 0
  instance_fp: 2
  instance_fn: 2
  instance_precision: 0.0
  instance_recall: 0.0
  instance_f1: 0.0
  instance_kp_mean_error_px: 360.624451
  instance_center_mean_error_px: 360.624451
  instance_count_accuracy: 1.0
  sim2_rotation_error_deg: 180.0
  sim2_scale_relative_error: 1.0
  sim2_translation_error_px: 360.624451
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
    real_evaluation.preprocess.content_fraction=1.0 real_evaluation.checkpoint_path=court_alignment/augmentation_pilot/b00_combined_v1_warm20_s42/logs/version_0/checkpoints/court-alignment-epoch\\=019.ckpt
    real_evaluation.output_dir=court_alignment/real_heatmap/aug_pilot_combined_v1
artifacts:
  run_dir: knowledge/runs/run-court-align-b00-eval-aug-pilot-combined-v1
  predictions: knowledge/runs/run-court-align-b00-eval-aug-pilot-combined-v1/pred_test.npz
parents: [run-court-align-aug-pilot-combined-v1]
relations:
- {to: group-court-align-b00-real-heatmap-eval, rel: compares}
tags: [court-alignment, kp14, augmentation, b00, real-heatmap]
---

## 考察 / Findings

### 要約

3種のaugmentationを同時適用したcombined checkpointはB00でF1=0だった。単独のappearance/structureで得られた改善を再現しなかった。

### アーキテクチャ詳細

入力は32視点平均のB00 heatmap、decoder threshold=0.25、max preprocess、σ=2.0。参照はaccepted alignmentで独立GTではない。

### メトリクスの解釈

TP/FP/FN=0/2/2、KP/center誤差=360.62px、count accuracy=1.0、Sim(2)回転=180°、scale相対誤差=1.0。semantic countは4/3、推定centerは(102.05,80.96)/(112.32,176.23)px、fit scaleは7.167/7.281px/mだった。

### アーキテクチャ⇄メトリクスの因果考察

同時適用によりsynthetic test F1も0.6674まで低下しており、学習難易度過多がB00のcollapseに繋がったと考えられる（仮説）。

### 既存実験との比較

identity modelとscale単独のB00 F1=0で、appearance/structure単独のF1=0.5を下回った。combinedは現時点の採用候補ではない。

### 次に有効な実験

combinedを直接使わず、appearanceを固定してstructure/scaleを低確率で段階的に導入する。
