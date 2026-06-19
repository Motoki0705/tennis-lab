---
id: run-i525-parameff
type: run
title: i525_parameff
issue: 525
provider: claude
session: d22b7d68-7d91-4a6f-862d-434085e5d2d9
date: '2026-06-19'
status: done
config:
  model: multiview_axial_split_eff
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 1.728365
  position_error_std_m: 1.182573
  position_error_median_m: 1.420856
  angular_error_deg: 74.462891
  angular_error_std_deg: 52.704441
  angular_error_median_deg: 66.722878
  x_error_m: 0.848323
  y_error_m: 1.354288
  z_error_m: 0.091677
  position_accuracy: 0.077113
  angle_accuracy: 0.159768
  position_accuracy_0.5m: 0.077113
  position_accuracy_1m: 0.278876
  position_accuracy_2m: 0.711043
  angle_accuracy_10deg: 0.128856
  angle_accuracy_15deg: 0.159768
  angle_accuracy_30deg: 0.263253
repro:
  commit: caa677e3995f4bafadf5f354813b1944c83bef1c
  branch: exp/i525-parameff
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split_eff loss=canonical_rot data=multiview_sequence training.trainer.max_epochs=8
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i525-parameff
  predictions: knowledge/runs/run-i525-parameff/pred_test.npz
  log: .training_queue/logs/1781862734379484842_577732_i525_parameff.log
parents:
- run-i518-exp10
relations:
- {to: run-i525-shared-6l, rel: compares}
- {to: run-i521-ex10-vel, rel: compares}
tags: [plcs, canonical, split-trunk, parameter-efficiency, issue-533-pipeline]
---

## 考察 / Findings

#525 follow-up: **分離 trunk アーキテクチャのパラメータ効率上限探索**。EX10 split (78M,
9.98°/0.238m) を幅・深さともに縮小した `multiview_axial_split_eff`
(hidden_dim 512→256, num_heads 8→4, num_task_layers 6→3 = **約9.9M, EX10比 ~12.7%**)。
「分離の優位が真にアーキテクチャ由来なら、容量を大きく削っても位置精度は崩れにくいはず」が仮説。

⚠️ **本ノードは issue #533 のログ基盤(repro バンドル + test 推論保存)を end-to-end 検証するための
bounded run（max_epochs=8）**であり、metrics は収束値ではない（EX10 は 200ep）。
位置 1.73m / 角度 74.5° は 8ep 時点の値で、フル学習との直接比較は不可。
本格比較は `knowledge/runs/run-i525-parameff/repro.sh` で full epoch 再学習して取得すること。

- 検証できたこと: 縮小 split 構成が学習・test 推論・npz 保存まで通ること、test split=100 scene
  の予測が `pred_test.npz`（pred/target position・rotation + mask, scene_ids）として保存されること。
- 次: repro.sh で full epoch 化し、`run-i518-exp10` / `run-i525-shared-6l`(39.3M) と
  同条件で効率フロンティアを比較する。
