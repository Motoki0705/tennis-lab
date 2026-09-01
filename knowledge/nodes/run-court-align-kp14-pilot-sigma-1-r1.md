---
id: run-court-align-kp14-pilot-sigma-1-r1
type: run
title: KP14 Ground-UV alignment pilot（σ=1.0、小予算）
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-01'
status: done
config:
  model: CourtAlignmentCNN（4-down U-Net、KP14 heatmap + 2ch center vote）
  loss: focal heatmap + 0.05 masked Smooth L1 center-vote
  data: procedural multi-court ground-UV line mask（identity augmentation）
  image_size: 256
  train_samples: 128
  val_samples: 32
  test_samples: 32
  batch_size: 4
  steps_per_epoch: 32
  max_epochs: 20
  seed: 42
  sigma_px: 1.0
  vote_radius_px: 3.0
metrics:
  instance_precision: 0.1875
  instance_recall: 0.2553191489361702
  instance_f1: 0.21621621621621623
  instance_count_accuracy: 0.46875
  instance_count_mae: 0.53125
  matched_center_mean_error_px: 3.5027733544508615
  instance_kp_mean_error_px: 308.5652202441589
  instance_kp_pck_2px: 0.14446529080675422
  instance_kp_pck_4px: 0.14446529080675422
  sim2_translation_error_px: 268.59656496682857
  sim2_rotation_error_deg: 134.0852520459274
  sim2_scale_relative_error: 0.7455048737956788
repro:
  commit: 456a6b19e6ffbf73b9645b76f860dde14ae37906
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data.train_samples=128 data.val_samples=32
    data.test_samples=32 data.batch_size=4 data.num_workers=4 training.steps_per_epoch=32
    training.trainer.max_epochs=20 training.trainer.log_every_n_steps=8 run.output_dir=court_alignment/pilot_sigma_1_r1
artifacts:
  run_dir: knowledge/runs/run-court-align-kp14-pilot-sigma-1-r1
  predictions: knowledge/runs/run-court-align-kp14-pilot-sigma-1-r1/pred_test.npz
  log: .training_queue/logs/1788249772365915349_604689_court-align-kp14-pilot-sigma-1-r1.log
  output_dir: outputs/court_alignment/pilot_sigma_1_r1/logs/version_0
  curves: knowledge/runs/run-court-align-kp14-pilot-sigma-1-r1/curves.png
  tb_logdir: outputs/court_alignment/pilot_sigma_1_r1/logs/version_0
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
- pilot
- sigma-100
---

## 考察 / Findings

`metrics.json` 由来の test headline 値は、本文全体で小数第7位に丸めて表示する。

### 要約

σ=1.0 を小予算で確認した pilot は instance F1 0.2162162、KP 誤差 308.5652202 px に留まり、明確に未学習だった。σ 選択には使用できない。

### アーキテクチャ詳細

256×256 の single-channel ground-UV line mask を入力し、4-down U-Net が 14 semantic keypoint heatmap と 2 channel の center vote を同じ解像度で出力する。1–2 面を含む procedural multi-court data、focal heatmap loss と masked center-vote loss、σ=1.0、vote radius 3 px は通常構成と同じだが、本 run だけは 128/32/32 samples、batch 4、32 steps/epoch、20 epochs の小予算である。

### メトリクスの解釈

観測値は precision 0.1875000、recall 0.2553191、F1 0.2162162、count accuracy 0.4687500、KP 308.5652202 px、center 3.5027734 px、Sim(2) rotation 134.0852520°、scale 0.7455049、translation 268.5965650 px だった。`diagnostic_metrics.json` の `loss` は 0.6301270（小数第7位表示）だった。instance/KP/Sim(2) の全系統が実用域に達しておらず、収束カーブも小予算 run の学習状態を示す参考に限られる。

### アーキテクチャ⇄メトリクスの因果考察

観測として、同じ σ=1.0 の full-budget run より学習 sample、step、epoch がすべて少なく、予測 instance と KP 対応が崩れている。仮説として、疎な Gaussian target を学習するのに必要な反復数へ達しなかったことが主因である。ただし複数の予算要因を同時に変えているため、どの要因が支配的かはこの run だけでは断定できず、σ の良否も分離できない。

### 既存実験との比較

`run-i621-court-kp512-resume-r4` は画像上の単一 court KP14 検出、`run-i618-b00-ground-line-court-fit-v1` は detector line を ground plane に集約する幾何 fit であり、本 run は両者を ground-UV 上の learned multi-court instance alignment へ接続する試行である。入力・split・指標が異なるため、それらの数値とは直接比較しない。

### 次に有効な実験

4096/512/512 samples、batch 16、256 steps/epoch、50 epochs、seed 42 を固定した σ sweep で比較する。pilot の数値を σ=1.0 の候補評価へ混ぜない。
