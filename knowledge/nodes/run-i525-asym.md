---
id: run-i525-asym
type: run
title: i525_asym_200ep
issue: 535
provider: claude
session: d22b7d68-7d91-4a6f-862d-434085e5d2d9
date: '2026-06-19'
status: done
config:
  model: multiview_axial_split_asym
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.700435
  position_error_std_m: 0.628913
  position_error_median_m: 0.504851
  angular_error_deg: 19.93961
  angular_error_std_deg: 20.894035
  angular_error_median_deg: 14.197411
  x_error_m: 0.375438
  y_error_m: 0.520758
  z_error_m: 0.041689
  position_accuracy: 0.491645
  angle_accuracy: 0.516981
  position_accuracy_0.5m: 0.491645
  position_accuracy_1m: 0.822103
  position_accuracy_2m: 0.938324
  angle_accuracy_10deg: 0.380994
  angle_accuracy_15deg: 0.516981
  angle_accuracy_30deg: 0.825228
repro:
  commit: 6d24b4dccad36a73bd56a526602757e16bac0275
  branch: exp/i525-asym
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split_asym loss=canonical_rot data=multiview_sequence training.trainer.max_epochs=200
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i525-asym
  predictions: knowledge/runs/run-i525-asym/pred_test.npz
  log: .training_queue/logs/1781875485504708764_626801_i525_asym_200ep.log
  curves: knowledge/runs/run-i525-asym/curves.png
parents:
- run-i518-exp10
- run-i521-ex10-vel
relations:
- to: run-i525-shared-match-dim
  rel: compares
- to: run-i525-shared-match-layers
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- asymmetric
---

## 考察 / Findings

### 要約
分離 rotation trunk を深層化した非対称容量アーキ（200ep 収束）。「深くすれば回転が改善」の仮説は支持されず、回転・位置とも EX10 や 8 倍小さい parameff に劣る。

### アーキテクチャ詳細
`multiview_axial_split_asym` + `canonical_rot`：pose trunk 6 層、rotation trunk を `rot_num_task_layers=10` で深層化、`hidden_dim 512`、約 103M params。`max_epochs=200`。

### メトリクスの解釈
位置 `0.700m` / 回転 `19.94°`。回転は EX10 (`9.98°`) より悪化し、8 倍小さい parameff (`15.55°`, 9.9M) にも劣る。位置 `0.700m` も EX10 (`0.238m`) / parameff (`0.569m`) に劣る。

### アーキテクチャ⇄メトリクスの因果考察
要因の候補:(1) #525 で効いたのは trunk の**幅**で**深さ**ではない—深層化は容量軸として不適か最適化困難、(2) 103M は 200ep でも相対的に未収束（大型ほど要 epoch）、(3) rotation trunk 深層化が aux/canonical の共有学習を不安定化。

### 既存実験との比較
親 [[run-i518-exp10]] / [[run-i521-ex10-vel]] に対し回転・位置とも悪化。param-matched 共有 trunk [[run-i525-shared-match-dim]] / [[run-i525-shared-match-layers]] と対照（`compares`）。

### 次に有効な実験
非対称化するなら深さでなく**幅**（rotation trunk を広く）を試す、または大型モデルは epoch を増やす。
