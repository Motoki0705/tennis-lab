---
id: run-i560-nocanon-rs-s4-h4
type: run
title: i545_nocanon_rs_s4_h4
issue: 560
provider: codex
session: 019ef7e7-03a8-7e53-9c92-21cdcb326a79
date: '2026-06-26'
status: done
config:
  model: multiview_axial_split
  loss: no_canonical
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.402243
  position_error_std_m: 0.378259
  position_error_median_m: 0.297362
  angular_error_deg: 9.490103
  angular_error_std_deg: 10.790317
  angular_error_median_deg: 6.347322
  x_error_m: 0.173854
  y_error_m: 0.322147
  z_error_m: 0.047841
  position_accuracy: 0.721902
  angle_accuracy: 0.824207
  position_accuracy_0.5m: 0.721902
  position_accuracy_1m: 0.955457
  position_accuracy_2m: 0.993171
  angle_accuracy_10deg: 0.679677
  angle_accuracy_15deg: 0.824207
  angle_accuracy_30deg: 0.960782
repro:
  commit: ed6eef1b4fbfa5431d1ce40f3a010e90115b09fb
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=4
    model.num_task_layers=4 model.predict_canonical_pose=false data=chunked_multiview_sequence_bs8
    data.batch_size=8 training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256]
    loss=no_canonical loss.rotation_weight=0.5 +loss.angle_weight=1.0 training.trainer.max_epochs=200
    training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i560-nocanon-rs-s4-h4
  predictions: knowledge/runs/run-i560-nocanon-rs-s4-h4/pred_test.npz
  log: .training_queue/logs/1782307133839957697_639017_i545_nocanon_rs_s4_h4.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_17
  curves: knowledge/runs/run-i560-nocanon-rs-s4-h4/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_17
parents:
- run-i560-nocanon-s4-h4
relations:
- to: run-i560-nocanon-s4-h4
  rel: compares
- to: run-i545-s4-h4
  rel: compares
tags:
- plcs
- no-canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- loss-tuning
- rot-strong
---

## 考察 / Findings

### 要約
S=4/H=4 balanced で canonical head 無し ＋ rot-strong。回転 9.49°/位置 0.402m。回転は戻るが、**位置は strict 0.185m から 0.402m へ再悪化し canonical_rot baseline（0.337m）よりも悪い**。共有が深い balanced は rot-strong で両立できない。

### アーキテクチャ詳細
`multiview_axial_split` H=4/S=4。`predict_canonical_pose=false`, `loss=no_canonical` ＋ `rotation_weight=0.5` `+angle_weight=1.0`。strict 版 [[run-i560-nocanon-s4-h4]] との差は rotation/angle weight のみ。

### メトリクスの解釈
回転 mean 9.49 / median 6.35、角@15=0.824。位置 mean 0.402 / median 0.297、位置@0.5m=0.722。Wave B 内で位置は s0_h12 を除き最悪級、回転も S=5/H=2・S=6/H=0 に劣る。

### アーキテクチャ⇄メトリクスの因果考察
共有 H=4 が大きいため、rotation を 0.5 に戻すと共有 trunk が強く rotation に引かれ位置が大きく痩せる（0.185→0.402m）。回転も浅共有 S=5/H=2（7.54°）より伸びない＝深い共有は rot/pos 双方に中途半端（仮説）。criterion #2/#3 の合成: 共有が深いほど位置の再悪化が大きく、残存する feature 競合コストも最大。

### 既存実験との比較
- canonical_rot baseline [[run-i545-s4-h4]]（8.23°/0.337m）に対し回転・位置とも劣る＝balanced では canonical head 除去＋rot-strong に利点なし。
- strict [[run-i560-nocanon-s4-h4]]（0.185m）からの位置再悪化幅（+0.217m）は H 系列で最大。

### 次に有効な実験
- balanced × rot-strong は不利。balanced で両立を狙うなら rotation_weight を 0.5 未満に下げ位置との中間点を探るか、配分を fully separate 寄りにする。
