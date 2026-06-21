---
id: run-i541-parameff-medcap
type: run
title: i541_parameff_medcap
issue: 536
provider: claude
session: d22b7d68-7d91-4a6f-862d-434085e5d2d9
date: '2026-06-20'
status: done
config:
  model: multiview_axial_split_medcap
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.571368
  position_error_std_m: 0.50124
  position_error_median_m: 0.424379
  angular_error_deg: 17.748798
  angular_error_std_deg: 18.633047
  angular_error_median_deg: 13.230637
  x_error_m: 0.238456
  y_error_m: 0.472779
  z_error_m: 0.049666
  position_accuracy: 0.582902
  angle_accuracy: 0.5679
  position_accuracy_0.5m: 0.582902
  position_accuracy_1m: 0.874026
  position_accuracy_2m: 0.973207
  angle_accuracy_10deg: 0.381985
  angle_accuracy_15deg: 0.5679
  angle_accuracy_30deg: 0.860531
repro:
  commit: 6399aa6f6848994957412eebabf4a4330c95cd15
  branch: feat/issue-533-experiment-log-format
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split_medcap loss=canonical_rot data=multiview_sequence
    training.trainer.max_epochs=200 run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i541-parameff-medcap
  predictions: knowledge/runs/run-i541-parameff-medcap/pred_test.npz
  log: .training_queue/logs/1781927073241453654_759989_i541_parameff_medcap.log
  curves: knowledge/runs/run-i541-parameff-medcap/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_0
parents:
- run-i525-parameff
- run-i518-exp10
relations:
- to: run-i541-parameff-deeppose
  rel: compares
- to: run-i525-parameff
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- parameter-efficiency
- width
- efficiency-frontier
---

## 考察 / Findings

### 要約
eff と EX10 の中間容量を主に幅で埋めた構成（200ep, 28.9M）。幅増しは回転を改善せず、より少容量の深さ振り deeppose に完敗。

### アーキテクチャ詳細
`multiview_axial_split_medcap` + `canonical_rot`：`hidden_dim 384` / `num_heads 6` / `num_task_layers 4`、約 28.9M params。`max_epochs=200`。

### メトリクスの解釈
位置 `0.571m` / 回転 `17.75°`。幅 256→384・約 3 倍のパラメータ（9.9M→28.9M）を投じても eff ベースライン（[[run-i525-parameff]]: `15.55°/0.569m`）とほぼ同じか、むしろ悪化。「~30M で回転ギャップを半分埋める」狙いは外れた。

### アーキテクチャ⇄メトリクスの因果考察
幅を足しても回転は伸びない＝回転の主因は幅ではない。より少ない 19.5M で深さを振った [[run-i541-parameff-deeppose]]（`9.55°/0.202m`）が、より多い 28.9M で幅を振った本構成を回転で約 2 倍・位置で約 2.8 倍引き離す。同予算では深さ ≫ 幅が明確。

### 既存実験との比較
深さ振りの [[run-i541-parameff-deeppose]] と対（`compares`）—#536 内で「回転の主因は幅でなく深さ」を最もクリーンに示すペア。親 [[run-i525-parameff]] とも比較（`compares`）。

### 次に有効な実験
縮小 split の容量は幅ではなく深さに割り当てる。中間容量を狙うなら medcap（幅広・浅）ではなく deeppose 路線（幅維持・深層）を採る。
