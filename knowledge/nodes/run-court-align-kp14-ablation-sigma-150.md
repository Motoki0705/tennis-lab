---
id: run-court-align-kp14-ablation-sigma-150
type: run
title: KP14 Ground-UV alignment σ=1.5
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
  sigma_px: 1.5
  vote_radius_px: 3.0
metrics:
  instance_precision: 0.994750656167979
  instance_recall: 1.0
  instance_f1: 0.9973684210526316
  instance_count_accuracy: 0.9921875
  instance_count_mae: 0.0078125
  matched_center_mean_error_px: 0.26835523638321396
  instance_kp_mean_error_px: 1.0790055718858533
  instance_kp_pck_2px: 0.997573083287369
  instance_kp_pck_4px: 0.997573083287369
  sim2_translation_error_px: 0.07533139174670168
  sim2_rotation_error_deg: 0.07345777014910645
  sim2_scale_relative_error: 0.0009740952687413992
repro:
  commit: 456a6b19e6ffbf73b9645b76f860dde14ae37906
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data.sigma_px=1.5 run.output_dir=court_alignment/ablation_sigma_150
artifacts:
  run_dir: knowledge/runs/run-court-align-kp14-ablation-sigma-150
  predictions: knowledge/runs/run-court-align-kp14-ablation-sigma-150/pred_test.npz
  log: .training_queue/logs/1788250299515377987_662434_court-align-kp14-ablation-sigma-150.log
  output_dir: outputs/court_alignment/ablation_sigma_150/logs/version_0
  curves: knowledge/runs/run-court-align-kp14-ablation-sigma-150/curves.png
  tb_logdir: outputs/court_alignment/ablation_sigma_150/logs/version_0
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
- sigma-150
---

## 考察 / Findings

`metrics.json` 由来の test headline 値は、本文全体で小数第7位に丸めて表示する。

### 要約

σ=1.5 は test F1 0.9973684 / count accuracy 0.9921875 で σ=2.0 と同率首位となり、KP 1.0790056 px まで改善した。σ=2.0 より Sim(2) residual は小さいが、KP/center/test loss は劣る。

### アーキテクチャ詳細

全 ablation 共通の CourtAlignmentCNN は 256×256 ground-UV line mask から KP14 heatmap と center vote を出す。4096/512/512 samples、batch 16、256 steps/epoch、50 epochs、seed 42、identity augmentation、loss、decoder、vote radius 3 px を固定し、本 run では KP Gaussian target の σ だけを 1.5 にした。固定 center-vote mask は target-width 比較への混入を避けている。

### メトリクスの解釈

test count accuracy 0.9921875、F1 0.9973684、KP 1.0790056 px、PCK@2 0.9975731、center 0.2683552 px、rotation 0.0734578°、scale 0.0009741、translation 0.0753314 px だった。`diagnostic_metrics.json` の `loss` は 0.0135679（小数第7位表示）だった。TensorBoard/job log 上の best validation 表示値は epoch 42、loss 0.0191（小数第4位）/ KP 1.26 px（小数第2位）で、後半まで収束が進んだ checkpoint を共通規則で選択した。

### アーキテクチャ⇄メトリクスの因果考察

観測として、σ=1.0 から σ=1.5 へ広げると KP、center、PCK、F1、count、test loss がすべて改善し、Sim(2) residual は悪化した。仮説として、広い target が正解近傍の教師密度を増やして absolute localization と instance recovery を安定化する一方、peak の鋭さ低下が微小な相対 pose residual に不利だった可能性がある。peak shape の測定なしには因果を断定しない。

### 既存実験との比較

σ=0.75/1.0 より KP/center/F1/count/test loss が良い。σ=2.0 とは F1/count が同率で、rotation 0.0734578°、scale 0.0009741、translation 0.0753314 px は良いが、KP 1.0790056 px と center 0.2683552 px は σ=2.0 に及ばない。親 2 run の異種指標とは直接比較しない。

### 次に有効な実験

σ=2.0 を localization baseline として進めつつ、real/noisy heatmap で σ=1.5 が localization と pose residual の中間案になるかを同一 split で確認する。
