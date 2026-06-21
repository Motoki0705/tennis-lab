---
id: run-i535-asym-deep16-rerun
type: run
title: i535_asym_deep16_rerun
issue: 535
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-21'
status: done
config:
  model: multiview_axial_split_asym_deep16
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.252409
  position_error_std_m: 0.273971
  position_error_median_m: 0.173325
  angular_error_deg: 10.410616
  angular_error_std_deg: 10.222599
  angular_error_median_deg: 8.136292
  x_error_m: 0.10378
  y_error_m: 0.206081
  z_error_m: 0.03868
  position_accuracy: 0.901963
  angle_accuracy: 0.791273
  position_accuracy_0.5m: 0.901963
  position_accuracy_1m: 0.974953
  position_accuracy_2m: 0.996217
  angle_accuracy_10deg: 0.606614
  angle_accuracy_15deg: 0.791273
  angle_accuracy_30deg: 0.960668
repro:
  commit: 6d24b4dccad36a73bd56a526602757e16bac0275
  branch: exp/i525-asym
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split_asym_deep16 loss=canonical_rot
    data=multiview_sequence data.batch_size=4 training.trainer.max_epochs=200 run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i535-asym-deep16-rerun
  predictions: knowledge/runs/run-i535-asym-deep16-rerun/pred_test.npz
  log: .training_queue/logs/1781994475574624725_1061676_i535_asym_deep16_rerun.log
parents:
- run-i518-exp10
- run-i525-asym
relations:
- to: run-i540-asym-deep16
  rel: supersedes
- to: run-i518-exp10
  rel: compares
- to: run-i535-asym-wide-rerun
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- asymmetric
- depth
- capacity-frontier
- valid-rerun
---

## 考察 / Findings

正しい worktree(`exp/i525-asym`, commit `6d24b4d`, cwd=`/home/kamimura/projects/wt/i525-asym`)で再実行した、**有効**な非対称深さ計測。rotation trunk 16 層 / pose trunk 6 層、hidden 512 / heads 8 ＝ **142.3M params**(起動ログで確認)。no-op だった旧 `run-i540-asym-deep16` の 78.1M(＝EX10 再学習)とは**別物**で、本ノードがそれを supersede する。200ep フル学習(early stop なし)。

- **test: 回転 10.41°(median 8.14°)/ 位置 0.252m(median 0.173m)**。
- **EX10(78.1M, 9.98°/0.238m)に届かない。** train ang は 8.16°(train loss 0.06)まで収束したのに val/test は 10.41° ＝ 固定小データ(≈800 窓/ep)では rotation trunk を 6→16 に深めた余剰容量(+64M)を活かせず、わずかに過学習。
- rot 深さ系列(幅512): EX10 6層 9.98° < **deep16 16層 10.41°** < asym 10層 19.94°。**非単調**で、深さで EX10 を超えるという #535 当初仮説は**不成立**。
- **当初結論の撤回**: 旧 `run-i540-asym-deep16`(8.40°)は作業ディレクトリ取り違えで `rot_num_task_layers` が no-op になり対称 EX10 を再学習しただけ。「深さが回転の主レバー」は撤回。
- batch=4(VRAM 16GB 制約。基準 EX10 は batch 8 ＝ 軽微な交絡)。容量が効くかの公平判定は **#539**(chunked・データリッチ・適正バッチ)へ。
