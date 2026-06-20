---
id: run-i525-parameff
type: run
title: i525_parameff_200ep
issue: 536
provider: claude
session: d22b7d68-7d91-4a6f-862d-434085e5d2d9
date: '2026-06-19'
status: done
config:
  model: multiview_axial_split_eff
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.569017
  position_error_std_m: 0.438843
  position_error_median_m: 0.466388
  angular_error_deg: 15.549188
  angular_error_std_deg: 20.593863
  angular_error_median_deg: 10.43282
  x_error_m: 0.273752
  y_error_m: 0.43476
  z_error_m: 0.056929
  position_accuracy: 0.541499
  angle_accuracy: 0.657817
  position_accuracy_0.5m: 0.541499
  position_accuracy_1m: 0.875172
  position_accuracy_2m: 0.990472
  angle_accuracy_10deg: 0.481153
  angle_accuracy_15deg: 0.657817
  angle_accuracy_30deg: 0.895907
repro:
  commit: caa677e3995f4bafadf5f354813b1944c83bef1c
  branch: exp/i525-parameff
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split_eff loss=canonical_rot data=multiview_sequence training.trainer.max_epochs=200
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i525-parameff
  predictions: knowledge/runs/run-i525-parameff/pred_test.npz
  log: .training_queue/logs/1781875485488211550_626786_i525_parameff_200ep.log
parents:
- run-i518-exp10
relations:
- to: run-i525-shared-6l
  rel: compares
- to: run-i521-ex10-vel
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- parameter-efficiency
---

## 考察 / Findings

200epoch まで収束させた縮小 split trunk(`multiview_axial_split_eff`: hidden_dim 256 / num_heads 4 / num_task_layers 3、約 9.9M params ≒ EX10(78M)の 12.7%)の確定結果。位置 0.569m / 回転 15.55°。

- **位置はアーキ由来で極めてパラメータ効率が高い**: 9.9M の縮小 split が、78M クラスの共有 trunk 群すべて(shared-6l 0.836m / shared-match-dim 0.848m / shared-match-layers 1.617m — いずれも位置 ~0.84m で頭打ち)を 0.569m で明確に上回る。位置精度の優位は容量ではなく split-trunk というアーキテクチャに起因し、8 分の 1 以下のパラメータでも維持される。
- **回転は容量を要する**: 一方で回転 15.55° は shared-6l(15.27°)とほぼ同等にとどまり、EX10 split(9.98°)や shared-match-dim(12.22°)には届かない。これは #525 の「幅(hidden_dim)が回転に効く」という所見と整合し、縮小により num_heads/hidden_dim を削った代償が回転側に集中して現れている。
- **効率フロンティアの結論**: 「位置は安価(アーキ由来)・回転は容量依存」。split-trunk の構造的利得で位置は小容量でも頭打ちを突破できるが、回転を EX10 水準へ詰めるには幅/ヘッド数の確保が必要。

次の示唆: 位置を犠牲にせず回転だけを底上げする中間点として、split 構造を保ったまま hidden_dim のみ EX10 寄りに戻す(num_layers は縮小維持)スイープが、効率フロンティアの最良トレードオフ候補になる。
