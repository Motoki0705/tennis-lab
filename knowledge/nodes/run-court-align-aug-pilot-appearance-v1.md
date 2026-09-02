---
id: run-court-align-aug-pilot-appearance-v1
type: run
title: court-align-aug-pilot-appearance-v1
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  data: b00_appearance_v1
metrics:
  instance_precision: 0.751592
  instance_recall: 0.919481
  instance_f1: 0.827103
  instance_count_accuracy: 0.664062
  instance_count_mae: 0.335938
  matched_center_mean_error_px: 2.303648
  instance_kp_mean_error_px: 77.179495
  instance_kp_pck_2px: 0.786479
  instance_kp_pck_4px: 0.786479
  sim2_translation_error_px: 29.170186
  sim2_rotation_error_deg: 14.57408
  sim2_scale_relative_error: 0.081721
repro:
  commit: 5de3a7d45e037a799d4dad0ae1ef3ac3cb24897e
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data=b00_appearance_v1 data.train_samples=1024
    data.val_samples=256 data.test_samples=256 data.batch_size=16 training.steps_per_epoch=64
    training.trainer.max_epochs=20 training.learning_rate=1.0e-3 training.warmup_steps=256
    run.seed=42 run.output_dir=court_alignment/augmentation_pilot/b00_appearance_v1_warm20_s42
artifacts:
  run_dir: knowledge/runs/run-court-align-aug-pilot-appearance-v1
  predictions: knowledge/runs/run-court-align-aug-pilot-appearance-v1/pred_test.npz
  output_dir: outputs/court_alignment/augmentation_pilot/b00_appearance_v1_warm20_s42/logs/version_0
  curves: knowledge/runs/run-court-align-aug-pilot-appearance-v1/curves.png
  tb_logdir: outputs/court_alignment/augmentation_pilot/b00_appearance_v1_warm20_s42/logs/version_0
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

σ=2.0固定で線幅・blur・振幅などappearance差を導入した `b00_appearance_v1`。B00でF1=0.5まで改善し、4条件中scaleより実入力への適応が良かったが、2面同時の成功には未到達だった。

### アーキテクチャ詳細

4-down U-NetのKP14 heatmap + 2ch center-vote CNN。1024/256/256 samples、batch 16、64 steps/epoch、20 epochs、σ=2.0。既存checkpointの座標正規化メタデータが欠落していたためscratch学習。

### メトリクスの解釈

synthetic testはF1=0.8271、KP誤差=77.18px、center誤差=2.30px、count accuracy=0.6641、Sim(2)回転誤差=14.57°、scale相対誤差=0.0817、test loss=0.1529。B00はTP=1/FP=1/FN=1、precision/recall/F1=0.5、matched center=0.59px、visible KP coverage=0.25、Sim(2)回転誤差=90.02°だった。

### アーキテクチャ⇄メトリクスの因果考察

appearance augmentationにより少なくとも1面は正しく対応し、center誤差も小さい。一方、もう1面はghost/欠損の影響でinstance対応できなかったと考えられる（仮説）。

### 既存実験との比較

identity σ=2.0のsynthetic F1=0.9974に対してsyntheticは低下したが、B00 F1は従来の0から0.5へ改善した。scale単独はB00 F1=0であり、本条件の方が有望だった。

### 次に有効な実験

appearance augmentationを軸に、線dropout/ghost lineを弱く追加し、coverage gateを落とさない範囲で強度を探索する。
