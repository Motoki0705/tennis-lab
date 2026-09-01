---
id: run-court-align-kp14-ablation-sigma-100
type: run
title: KP14 Ground-UV alignment σ=1.0
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
  sigma_px: 1.0
  vote_radius_px: 3.0
metrics:
  instance_precision: 0.9856957087126138
  instance_recall: 1.0
  instance_f1: 0.9927963326784546
  instance_count_accuracy: 0.978515625
  instance_count_mae: 0.021484375
  matched_center_mean_error_px: 0.2797369638637278
  instance_kp_mean_error_px: 1.4754307910216427
  instance_kp_pck_2px: 0.9963596249310535
  instance_kp_pck_4px: 0.9963596249310535
  sim2_translation_error_px: 0.06454484731448559
  sim2_rotation_error_deg: 0.054825146564884825
  sim2_scale_relative_error: 0.000841873549130016
repro:
  commit: 456a6b19e6ffbf73b9645b76f860dde14ae37906
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data.sigma_px=1.0 run.output_dir=court_alignment/ablation_sigma_100
artifacts:
  run_dir: knowledge/runs/run-court-align-kp14-ablation-sigma-100
  predictions: knowledge/runs/run-court-align-kp14-ablation-sigma-100/pred_test.npz
  log: .training_queue/logs/1788250299441571085_662412_court-align-kp14-ablation-sigma-100.log
  output_dir: outputs/court_alignment/ablation_sigma_100/logs/version_0
  curves: knowledge/runs/run-court-align-kp14-ablation-sigma-100/curves.png
  tb_logdir: outputs/court_alignment/ablation_sigma_100/logs/version_0
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
- sigma-100
---

## 考察 / Findings

`metrics.json` 由来の test headline 値は、本文全体で小数第7位に丸めて表示する。

### 要約

full-budget の σ=1.0 は pilot から大幅に改善したが、固定 sweep 内では test F1 0.9927963、KP 1.4754308 px で、採用指標を首位にできなかった。用途別候補として σ=0.75 と σ=2.0 に挟まれ、本 sweep では支配的な選択肢ではない。

### アーキテクチャ詳細

σ=0.75/1.5/2.0 と同じ CourtAlignmentCNN、procedural multi-court ground-UV data、focal heatmap + masked center-vote loss を使用した。4096/512/512 samples、batch 16、256 steps/epoch、50 epochs、seed 42、vote radius 3 px を固定し、差分は σ=1.0 と output directory だけである。center-vote mask が σ に依存しないため、KP Gaussian target 幅だけを比較できる。

### メトリクスの解釈

test count accuracy 0.9785156、F1 0.9927963、KP 1.4754308 px、PCK@2 0.9963596、center 0.2797370 px、rotation 0.0548251°、scale 0.0008419、translation 0.0645448 px だった。`diagnostic_metrics.json` の `loss` は 0.0207129（小数第7位表示）だった。TensorBoard/job log 上の best validation 表示値は epoch 40、loss 0.0237（小数第4位）/ KP 1.53 px（小数第2位）で、共通の best-val/loss checkpoint 選択後の test 値である。

### アーキテクチャ⇄メトリクスの因果考察

観測として、σ=0.75 より test loss は低いが KP/F1/count/Sim(2) residual は悪く、σ=1.5/2.0 より KP/F1/count/loss も悪い。仮説として、σ=1.0 は狭い target の sparsity を十分に解消せず、広い target がもたらす localization 学習の安定化も得られない中間点になった可能性がある。これは候補選択の説明仮説であり、勾配密度の直接計測は未実施である。

### 既存実験との比較

同じ σ の小予算 pilot は F1 0.2162162 / KP 308.5652202 px、本 run は F1 0.9927963 / KP 1.4754308 px だが、sample 数・batch・step・epoch が同時に異なるため、改善を単一要因へ帰属させない。親 2 run は画像 KP14 と ground-line geometry の前提を提供するが、dataset/metric contract が違うため数値比較しない。

### 次に有効な実験

clean-synthetic baseline としては σ=2.0 を優先し、σ=1.0 の追加反復は行わない。target 幅の効果をより細かく確認する必要が生じた場合のみ、1.5–2.0 間を同じ固定予算で追加する。
