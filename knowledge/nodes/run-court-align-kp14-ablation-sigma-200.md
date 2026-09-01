---
id: run-court-align-kp14-ablation-sigma-200
type: run
title: KP14 Ground-UV alignment σ=2.0
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
  sigma_px: 2.0
  vote_radius_px: 3.0
metrics:
  instance_precision: 0.994750656167979
  instance_recall: 1.0
  instance_f1: 0.9973684210526316
  instance_count_accuracy: 0.9921875
  instance_count_mae: 0.0078125
  matched_center_mean_error_px: 0.2486753576099725
  instance_kp_mean_error_px: 0.8209754559458519
  instance_kp_pck_2px: 0.9981246552675124
  instance_kp_pck_4px: 0.9983452840595698
  sim2_translation_error_px: 0.08023348184349138
  sim2_rotation_error_deg: 0.08281236512480837
  sim2_scale_relative_error: 0.0010852774952191673
repro:
  commit: 456a6b19e6ffbf73b9645b76f860dde14ae37906
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data.sigma_px=2.0 run.output_dir=court_alignment/ablation_sigma_200
artifacts:
  run_dir: knowledge/runs/run-court-align-kp14-ablation-sigma-200
  predictions: knowledge/runs/run-court-align-kp14-ablation-sigma-200/pred_test.npz
  log: .training_queue/logs/1788250299591070475_662456_court-align-kp14-ablation-sigma-200.log
  output_dir: outputs/court_alignment/ablation_sigma_200/logs/version_0
  curves: knowledge/runs/run-court-align-kp14-ablation-sigma-200/curves.png
  tb_logdir: outputs/court_alignment/ablation_sigma_200/logs/version_0
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
- sigma-200
---

## 考察 / Findings

`metrics.json` 由来の test headline 値は、本文全体で小数第7位に丸めて表示する。

### 要約

σ=2.0 は test KP 0.8209755 px、center 0.2486754 px、`diagnostic_metrics.json` の loss 0.0105472（小数第7位表示）で 4 条件中最良となり、F1 0.9973684 / count accuracy 0.9921875 も同率首位だった。次の real/noisy heatmap 推論に使う localization baseline とする。

### アーキテクチャ詳細

4-down U-Net が 256×256 ground-UV line mask から full-resolution KP14 heatmap と center vote を予測する。4096/512/512 samples、batch 16、256 steps/epoch、50 epochs、seed 42、identity augmentation、loss、decoder、vote radius 3 px は他の ablation と同一で、本 run の差分は σ=2.0 と output directory だけである。vote radius と center-vote mask が固定なので、sweep は KP target 幅を分離する。

### メトリクスの解釈

test count accuracy 0.9921875、F1 0.9973684、KP 0.8209755 px、PCK@2 0.9981247、PCK@4 0.9983453、center 0.2486754 px、rotation 0.0828124°、scale 0.0010853、translation 0.0802335 px だった。`diagnostic_metrics.json` の `loss` は 0.0105472（小数第7位表示）だった。TensorBoard/job log 上の best validation 表示値は epoch 47、loss 0.0168（小数第4位）/ KP 1.16 px（小数第2位）で、50 epoch 近くまで改善が続いた checkpoint を共通規則で test した。

### アーキテクチャ⇄メトリクスの因果考察

観測として、σ を 1.0→1.5→2.0 と広げるほど KP/center/test loss が改善し、σ=2.0 が最大 PCK も得た一方、Sim(2) residual は sweep 中で最大だった。仮説として、広い Gaussian が正解近傍へ密な学習信号を供給し absolute localization を改善したが、peak の局所形状または subpixel decode の差が微小な相対 pose 推定に影響した可能性がある。このトレードオフは real/noisy 入力で再検証が必要である。

### 既存実験との比較

σ=0.75/1.0/1.5 のすべてより KP、center、test loss が良く、σ=1.5 と F1/count で同率である。σ=0.75 は Sim(2) rotation/scale/translation が良いため、pose residual を優先する用途では代替候補として残る。親の画像 KP14 run と B00 geometry fit は前提タスクであり、同一数値尺度の baseline ではない。

### 次に有効な実験

σ=2.0 を localization baseline として固定し、real detector 由来または line dropout・blur・false-line を加えた noisy heatmap inference を評価する。同じ split で σ=0.75 も残し、absolute localization と Sim(2) residual の選択が実入力でも一致するかを確認する。
