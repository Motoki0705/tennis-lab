---
id: run-i541-parameff-longtrain
type: run
title: i541_parameff_longtrain
issue: 536
provider: claude
session: d22b7d68-7d91-4a6f-862d-434085e5d2d9
date: '2026-06-20'
status: done
config:
  model: multiview_axial_split_eff
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.655458
  position_error_std_m: 0.478922
  position_error_median_m: 0.559786
  angular_error_deg: 24.145535
  angular_error_std_deg: 19.600937
  angular_error_median_deg: 18.576324
  x_error_m: 0.271781
  y_error_m: 0.541901
  z_error_m: 0.050541
  position_accuracy: 0.460341
  angle_accuracy: 0.406612
  position_accuracy_0.5m: 0.460341
  position_accuracy_1m: 0.81092
  position_accuracy_2m: 0.975221
  angle_accuracy_10deg: 0.260428
  angle_accuracy_15deg: 0.406612
  angle_accuracy_30deg: 0.697609
repro:
  commit: 6399aa6f6848994957412eebabf4a4330c95cd15
  branch: feat/issue-533-experiment-log-format
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split_eff loss=canonical_rot data=multiview_sequence training.trainer.max_epochs=400
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i541-parameff-longtrain
  predictions: knowledge/runs/run-i541-parameff-longtrain/pred_test.npz
  log: .training_queue/logs/1781927074922005793_760015_i541_parameff_longtrain.log
parents:
- run-i525-parameff
relations:
- to: run-i525-parameff
  rel: contradicts
tags:
- plcs
- canonical
- split-trunk
- parameter-efficiency
- long-training
---

## 考察 / Findings

eff(`multiview_axial_split_eff`, 256幅 / 4heads / 3層, 9.9M)を **epoch だけ 200→400 に倍増**した構成の結果。**位置 0.655m / 回転 24.15°**。

- **長期学習は逆効果**: 同一構成の `run-i525-parameff`(200ep)は 15.55°/0.569m。epoch を倍にした本ランは回転 15.55°→**24.15°**、位置 0.569→**0.655m** と**明確に悪化**した。「容量不足のモデルは長く回せば追いつく」という仮説は**反証**(→ contradicts)。
- **解釈**: 9.9M の under-capacity split を 400ep 回すと、汎化が崩れる(過学習)か最適化が不安定化する。位置・回転とも 200ep 時点が良く、追加 epoch は test 精度を毀損する。容量側のボトルネックは epoch では埋まらない。
- **示唆**: #536 の効率改善は「学習を延ばす」ではなく「**容量配分(深さ)**」で得るべき。回転を伸ばしたいなら longtrain ではなく deeppose(深さ振り)が正解。小容量 split のデフォルト学習量は 200ep を上限の目安とする。
