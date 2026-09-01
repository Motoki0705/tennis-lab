---
id: run-court-align-kp14-ablation-sigma-075
type: run
title: KP14 Ground-UV alignment σ=0.75
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-01'
status: done
config:
  model: CourtAlignmentCNN（4-down U-Net、KP14 heatmap + 2ch center vote）
  loss: focal heatmap + 0.05 masked Smooth L1 center-vote
  data: procedural multi-court ground-UV line mask（identity augmentation）
  image_size: 256
  train_samples: 4096
  val_samples: 512
  test_samples: 512
  batch_size: 16
  steps_per_epoch: 256
  max_epochs: 50
  seed: 42
  sigma_px: 0.75
  vote_radius_px: 3.0
metrics:
  instance_precision: 0.9908496732026144
  instance_recall: 1.0
  instance_f1: 0.9954038082731451
  instance_count_accuracy: 0.986328125
  instance_count_mae: 0.013671875
  matched_center_mean_error_px: 0.28317050913184155
  instance_kp_mean_error_px: 1.3291732111131034
  instance_kp_pck_2px: 0.9966905681191396
  instance_kp_pck_4px: 0.9966905681191396
  sim2_translation_error_px: 0.057765715659686946
  sim2_rotation_error_deg: 0.04846776415778165
  sim2_scale_relative_error: 0.0008015342144929823
repro:
  commit: 456a6b19e6ffbf73b9645b76f860dde14ae37906
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data.sigma_px=0.75 run.output_dir=court_alignment/ablation_sigma_075
artifacts:
  run_dir: knowledge/runs/run-court-align-kp14-ablation-sigma-075
  predictions: knowledge/runs/run-court-align-kp14-ablation-sigma-075/pred_test.npz
  log: .training_queue/logs/1788250299368402026_662390_court-align-kp14-ablation-sigma-075.log
  output_dir: outputs/court_alignment/ablation_sigma_075/logs/version_0
  curves: knowledge/runs/run-court-align-kp14-ablation-sigma-075/curves.png
  tb_logdir: outputs/court_alignment/ablation_sigma_075/logs/version_0
parents:
- run-i621-court-kp512-resume-r4
- run-i618-b00-ground-line-court-fit-v1
relations: []
tags:
- court-alignment
- kp14
- multi-court
- ground-uv
- cnn
- sigma-075
---

## 考察 / Findings

`metrics.json` 由来の test headline 値は、本文全体で小数第7位に丸めて表示する。

### 要約

固定予算の σ sweep で σ=0.75 は test F1 0.9954038、KP 1.3291732 px を達成し、4 条件中で Sim(2) residual が最小だった。一方、KP/center/test loss は σ=2.0 に及ばない。

### アーキテクチャ詳細

256×256 の ground-UV line mask から full-resolution KP14 heatmap と dense center vote を出す同一 CNN を使用した。4096/512/512 samples、batch 16、256 steps/epoch、50 epochs、seed 42、identity augmentation、loss、decoder、vote radius 3 px を全 ablation で固定し、本 run の差分は KP Gaussian target の σ=0.75 と output directory だけである。center-vote mask は固定 vote radius で作るため、sweep は KP target 幅を分離している。

### メトリクスの解釈

test count accuracy 0.9863281、F1 0.9954038、KP 1.3291732 px、PCK@2 0.9966906、center 0.2831705 px、rotation 0.0484678°、scale 0.0008015、translation 0.0577657 px だった。`diagnostic_metrics.json` の `loss` は 0.0284996（小数第7位表示）だった。TensorBoard/job log 上の best validation 表示値は epoch 41、loss 0.0276（小数第4位）/ KP 1.62 px（小数第2位）であり、後半まで改善した結果を共通の best-val/loss 規則で test した。

### アーキテクチャ⇄メトリクスの因果考察

観測として、σ=0.75 は 4 条件中で rotation・scale・translation residual が最小だが、test loss と KP localization は最良ではない。仮説として、狭い peak が decoded keypoint 間の相対幾何を鋭く制約する一方、pixel 近傍に与える教師信号が少なく、絶対 KP localization と heatmap loss の最適化を難しくした可能性がある。因果は target-width sweep の結果と整合するが、内部表現を直接測ってはいない。

### 既存実験との比較

σ=1.0 より F1、count、KP、PCK、Sim(2) residual が良いが、test loss は高い。σ=1.5/2.0 より Sim(2) residual は良く、KP/center/test loss は悪い。親の `run-i621-court-kp512-resume-r4` と `run-i618-b00-ground-line-court-fit-v1` はそれぞれ画像 KP 検出と非学習 geometry fit なので数値比較対象ではなく、本 run は ground-UV multi-court learned alignment の clean-synthetic 基準である。

### 次に有効な実験

pose residual を優先する代替 baseline として σ=0.75 を保持し、σ=2.0 と同じ real/noisy detector heatmap 入力で比較する。clean synthetic の差が入力ノイズ下でも維持されるかを検証する。
