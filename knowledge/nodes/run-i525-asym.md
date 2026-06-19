---
id: run-i525-asym
type: run
title: i525_asym
issue: 525
provider: claude
session: d22b7d68-7d91-4a6f-862d-434085e5d2d9
date: '2026-06-19'
status: done
config:
  model: multiview_axial_split_asym
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 1.682742
  position_error_std_m: 0.982201
  position_error_median_m: 1.529504
  angular_error_deg: 72.075562
  angular_error_std_deg: 50.017761
  angular_error_median_deg: 61.737881
  x_error_m: 0.680131
  y_error_m: 1.404864
  z_error_m: 0.075512
  position_accuracy: 0.080564
  angle_accuracy: 0.133703
  position_accuracy_0.5m: 0.080564
  position_accuracy_1m: 0.251846
  position_accuracy_2m: 0.687236
  angle_accuracy_10deg: 0.089597
  angle_accuracy_15deg: 0.133703
  angle_accuracy_30deg: 0.265955
repro:
  commit: 6d24b4dccad36a73bd56a526602757e16bac0275
  branch: exp/i525-asym
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split_asym loss=canonical_rot data=multiview_sequence training.trainer.max_epochs=8
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i525-asym
  predictions: knowledge/runs/run-i525-asym/pred_test.npz
  log: .training_queue/logs/1781862734397001991_577747_i525_asym.log
parents:
- run-i518-exp10
- run-i521-ex10-vel
relations:
- {to: run-i525-shared-match-dim, rel: compares}
- {to: run-i525-shared-match-layers, rel: compares}
tags: [plcs, canonical, split-trunk, asymmetric, issue-533-pipeline]
---

## 考察 / Findings

#525 follow-up: **非対称容量アーキテクチャ**。#525 の知見＝「回転は trunk 容量(幅)で改善するが、
位置は容量ではなく trunk 分離に依存する（shared では幅を増やしても ~0.84m で頭打ち）」を踏まえ、
容量を**非対称配分**する。`multiview_axial_split_asym`: pose trunk は実績の 6 層を維持し、
**分離された rotation trunk だけ 10 層に深化**（hidden_dim 512 共通, 計 ~103M）。
`PLCSMultiViewAxialSplitModel` に後方互換の `rot_num_task_layers`（既定=num_task_layers）を追加して実現。
#525 の depth-collapse（shared-match-layers: 29.8°/1.62m）は**共有** trunk で 2 タスクが混ざった効果なので、
rotation trunk が分離されている本構成では位置を汚染しない、というのが設計上の予測。

⚠️ **本ノードは issue #533 のログ基盤(repro バンドル + test 推論保存)を end-to-end 検証するための
bounded run（max_epochs=8）**であり、metrics は収束値ではない（EX10/比較対象は 200ep）。
位置 1.68m / 角度 72.1° は 8ep 時点の値で、フル学習との直接比較は不可。
本格評価は `knowledge/runs/run-i525-asym/repro.sh` で full epoch 再学習して取得すること。

- 検証できたこと: 非対称 split（rotation 深化）構成が学習・test 推論・npz 保存まで通ること、
  後方互換の `rot_num_task_layers` 追加で対称構成(78.1M)も不変であること。
- 次: repro.sh で full epoch 化し、回転が EX10(9.98°)を下回り位置(~0.24m)を維持するかを検証する。
