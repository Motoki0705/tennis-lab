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

eff(9.9M)と EX10(78M)の中間容量を、主に**幅**で埋めた構成(`multiview_axial_split_medcap`: hidden_dim 384 / num_heads 6 / num_task_layers 4、約 28.9M params)の 200epoch 結果。**位置 0.571m / 回転 17.75°**。

- **幅増しは回転を改善しない**: 幅 256→384・約 3 倍のパラメータ(9.9M→28.9M)を投じても、回転 17.75°/位置 0.571m は eff ベースライン(`run-i525-parameff`: 15.55°/0.569m)と**ほぼ同じか、むしろ悪化**。当初の「~30M で回転ギャップを半分埋める」狙いは外れた。
- **深さ(deeppose)に完敗**: より少ない 19.5M で深さを振った `run-i541-parameff-deeppose`(9.55°/0.202m)が、より多い 28.9M で幅を振った本構成(17.75°/0.571m)を回転で約 2 倍・位置で約 2.8 倍引き離す。**同程度の予算では深さ ≫ 幅**が明確。
- deeppose(depth)と medcap(width)の対比は、#536 内で「回転の主因は幅ではなく深さ」を最もクリーンに示すペア(→ compares)。

次の示唆: 縮小 split の容量は幅ではなく深さに割り当てる。中間容量を狙うなら medcap(幅広・浅)ではなく deeppose 路線(幅維持・深層)を採る。
