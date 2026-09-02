---
id: run-court-align-aug-pilot-structure-v1
type: run
title: court-align-aug-pilot-structure-v1
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  data: b00_structure_v1
metrics:
  instance_precision: 0.68107
  instance_recall: 0.85974
  instance_f1: 0.760046
  instance_count_accuracy: 0.605469
  instance_count_mae: 0.394531
  matched_center_mean_error_px: 2.462603
  instance_kp_mean_error_px: 100.155755
  instance_kp_pck_2px: 0.722519
  instance_kp_pck_4px: 0.722519
  sim2_translation_error_px: 50.727022
  sim2_rotation_error_deg: 25.347839
  sim2_scale_relative_error: 0.142109
repro:
  commit: 5de3a7d45e037a799d4dad0ae1ef3ac3cb24897e
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data=b00_structure_v1 data.train_samples=1024
    data.val_samples=256 data.test_samples=256 data.batch_size=16 training.steps_per_epoch=64
    training.trainer.max_epochs=20 training.learning_rate=1.0e-3 training.warmup_steps=256
    run.seed=42 run.output_dir=court_alignment/augmentation_pilot/b00_structure_v1_warm20_s42
artifacts:
  run_dir: knowledge/runs/run-court-align-aug-pilot-structure-v1
  predictions: knowledge/runs/run-court-align-aug-pilot-structure-v1/pred_test.npz
  output_dir: outputs/court_alignment/augmentation_pilot/b00_structure_v1_warm20_s42/logs/version_0
  curves: knowledge/runs/run-court-align-aug-pilot-structure-v1/curves.png
  tb_logdir: outputs/court_alignment/augmentation_pilot/b00_structure_v1_warm20_s42/logs/version_0
parents:
- run-court-align-kp14-ablation-sigma-200
relations:
- to: group-court-align-kp14-sigma-ablation
  rel: compares
tags:
- court-alignment
- kp14
- augmentation
- b00
---

## 考察 / Findings

### 要約

σ=2.0固定でline dropout・partial crop・false/ghost lineを含む `b00_structure_v1`。B00はF1=0.5で、appearanceと同率だったが、KP誤差・center誤差はappearanceより悪かった。

### アーキテクチャ詳細

4-down U-NetのKP14 heatmap + 2ch center-vote CNN。1024/256/256 samples、batch 16、64 steps/epoch、20 epochs、σ=2.0。checkpoint契約不一致のためscratch学習。

### メトリクスの解釈

synthetic testはF1=0.7600、KP誤差=100.16px、center誤差=2.46px、count accuracy=0.6055、Sim(2)回転誤差=25.35°、scale相対誤差=0.1421、test loss=0.1926。B00はTP=1/FP=1/FN=1、F1=0.5、matched center=3.86px、visible KP coverage=0.2857、Sim(2)回転誤差=90.56°だった。

### アーキテクチャ⇄メトリクスの因果考察

構造augmentationは1面の対応を成立させたが、synthetic性能とmatched centerがappearanceより悪い。仮説として、現実の欠損パターンには近づく一方、強度またはfalse-line配置が過剰で学習分布を広げすぎた。

### 既存実験との比較

identity σ=2.0のsynthetic F1=0.9974から低下した。B00 F1は従来の0から0.5へ改善し、scale単独のF1=0より有効だったが、appearanceのmatched center=0.59pxには及ばない。

### 次に有効な実験

structure強度を下げ、appearance augmentationと組み合わせる。false-line数とdropout率を独立に振る。
