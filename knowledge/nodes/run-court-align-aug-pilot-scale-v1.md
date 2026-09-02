---
id: run-court-align-aug-pilot-scale-v1
type: run
title: court-align-aug-pilot-scale-v1
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  data: b00_scale_v1
metrics:
  instance_precision: 0.866667
  instance_recall: 0.979221
  instance_f1: 0.919512
  instance_count_accuracy: 0.804688
  instance_count_mae: 0.195312
  matched_center_mean_error_px: 1.692796
  instance_kp_mean_error_px: 32.396635
  instance_kp_pck_2px: 0.910506
  instance_kp_pck_4px: 0.910749
  sim2_translation_error_px: 7.603199
  sim2_rotation_error_deg: 3.819794
  sim2_scale_relative_error: 0.021914
repro:
  commit: 5de3a7d45e037a799d4dad0ae1ef3ac3cb24897e
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data=b00_scale_v1 data.train_samples=1024
    data.val_samples=256 data.test_samples=256 data.batch_size=16 training.steps_per_epoch=64
    training.trainer.max_epochs=20 training.learning_rate=1.0e-3 training.warmup_steps=256
    run.seed=42 run.output_dir=court_alignment/augmentation_pilot/b00_scale_v1_warm20_s42
artifacts:
  run_dir: knowledge/runs/run-court-align-aug-pilot-scale-v1
  predictions: knowledge/runs/run-court-align-aug-pilot-scale-v1/pred_test.npz
  output_dir: outputs/court_alignment/augmentation_pilot/b00_scale_v1_warm20_s42/logs/version_0
  curves: knowledge/runs/run-court-align-aug-pilot-scale-v1/curves.png
  tb_logdir: outputs/court_alignment/augmentation_pilot/b00_scale_v1_warm20_s42/logs/version_0
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

σ=2.0を固定し、scale範囲拡張を中心とした `b00_scale_v1` で学習した。B00実heatmapの実測F1=0.0で、scale augmentation単独では成功に届かなかった。

### アーキテクチャ詳細

4-down U-NetのKP14 heatmap + 2ch center-vote CNN。1024/256/256 samples、batch 16、64 steps/epoch、20 epochs。σ=2.0は既知の最良値として固定し、augmentationはscale変動に限定した。既存σ=2.0 checkpointは座標正規化メタデータ欠落のため初期重みとして使用せず、scratchから学習した。

### メトリクスの解釈

synthetic testはF1=0.9195、KP誤差=32.40px、center誤差=1.69px、count accuracy=0.8047、Sim(2)回転誤差=3.82°、scale相対誤差=0.0219、test loss=0.1216。B00は2 instancesを数えられたが、TP=0/FP=2/FN=2、F1=0、KP/center誤差は未マッチpenaltyの360.62px、Sim(2)は未利用だった。

### アーキテクチャ⇄メトリクスの因果考察

観測上、scale augmentationはsynthetic性能を保ちつつ、実入力の7.17px/m（従来学習範囲3–6px/m）への外挿だけでは不十分だった。仮説として、B00の線欠損・ghost line・確率振幅の差がscale差より支配的である。

### 既存実験との比較

既存σ=2.0 identity augmentationはsynthetic F1=0.9974だった。本runはsynthetic F1が低く、B00でもbaseline同様にinstance countだけが合い、KP対応には失敗した。

### 次に有効な実験

appearanceまたはstructure augmentationを単独で強め、B00 F1とcoverage gateを比較する。scale範囲は7.2px/mを含むよう維持する。
