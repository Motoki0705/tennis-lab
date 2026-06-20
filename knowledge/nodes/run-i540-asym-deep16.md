---
id: run-i540-asym-deep16
type: run
title: i540_asym_deep16
issue: 535
provider: claude
session: d22b7d68-7d91-4a6f-862d-434085e5d2d9
date: '2026-06-20'
status: done
config:
  model: multiview_axial_split_asym_deep16
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.207075
  position_error_std_m: 0.182874
  position_error_median_m: 0.153868
  angular_error_deg: 8.396061
  angular_error_std_deg: 7.641904
  angular_error_median_deg: 6.27077
  x_error_m: 0.076728
  y_error_m: 0.171573
  z_error_m: 0.034041
  position_accuracy: 0.950193
  angle_accuracy: 0.839364
  position_accuracy_0.5m: 0.950193
  position_accuracy_1m: 0.991129
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.689944
  angle_accuracy_15deg: 0.839364
  angle_accuracy_30deg: 0.984475
repro:
  commit: 6399aa6f6848994957412eebabf4a4330c95cd15
  branch: feat/issue-533-experiment-log-format
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.plcs.scripts.train
    model=multiview_axial_split_asym_deep16 loss=canonical_rot data=multiview_sequence
    training.trainer.max_epochs=200 run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i540-asym-deep16
  predictions: knowledge/runs/run-i540-asym-deep16/pred_test.npz
  log: .training_queue/logs/1781927908633902412_762455_i540_asym_deep16.log
parents:
- run-i525-asym
- run-i518-exp10
relations:
- to: run-i525-asym
  rel: contradicts
- to: run-i518-exp10
  rel: compares
- to: run-i541-parameff-deeppose
  rel: confirms
tags:
- plcs
- canonical
- split-trunk
- asymmetric
- depth
- capacity-frontier
---

## 考察 / Findings

rotation trunk の深さを**極限まで**振った非対称構成(`multiview_axial_split_asym_deep16`: pose trunk 6 層・**rotation trunk 16 層**、hidden_dim 512 / num_heads 8、約 78.1M params ≒ EX10 と同等予算)の 200epoch 収束結果。**位置 0.207m / 回転 8.40°** で、本実験群の絶対最良。

- **EX10(split, 78M)を両指標で上回る**: 回転 8.40°(< EX10 9.98°)・位置 0.207m(< EX10 0.238m)。同等パラメータ予算で EX10 を超えたのは初。中央値も回転 6.27°・位置 0.154m と良好で、外れ値依存ではない(angular_std 7.64°)。
- **#535 の負の結論(深化は無効)を覆す**: `run-i525-asym`(rot=10, 103M)は 19.94° と大きく劣化したが、より小さい 78.1M で rot=16 まで深めた本構成は 8.40° に達した。つまり「分離 rotation trunk の深化は回転に効く」が成立する。`run-i525-asym` が負だった主因は深さそのものではなく、**103M という過大容量が 200ep で未収束/最適化困難**だった可能性が高い(同ノードの考察 (2) と整合)。深さは「学習可能なサイズ envelope に収まっている限り」回転の主レバーになる。
- **幅より深さ**: 同じ #535 で幅を広げた `run-i540-asym-wide`(768幅, 172M)は 12.27° に留まり、しかも resume を要した。深さ(78M で 8.40°)は幅(172M で 12.27°)より安価かつ高精度で、回転改善の効率が高い。

次の示唆: deep16 が新ベースライン候補。pose trunk は 6 層のまま rotation trunk 深化のみで EX10 を超えられたので、(1) rotation 深さの最適点(12/16/20 層)を sweep、(2) hidden_dim を 512→384 に絞っても深さで回転を維持できるか(deeppose 系の知見と接続)を確認したい。
