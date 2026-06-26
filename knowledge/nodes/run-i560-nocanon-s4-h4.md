---
id: run-i560-nocanon-s4-h4
type: run
title: i545_nocanon_s4_h4
issue: 560
provider: codex
session: 019ef7e7-03a8-7e53-9c92-21cdcb326a79
date: '2026-06-25'
status: done
config:
  model: multiview_axial_split
  loss: no_canonical
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.184639
  position_error_std_m: 0.162646
  position_error_median_m: 0.137753
  angular_error_deg: 51.911728
  angular_error_std_deg: 58.429848
  angular_error_median_deg: 21.644306
  x_error_m: 0.079684
  y_error_m: 0.142286
  z_error_m: 0.030728
  position_accuracy: 0.943922
  angle_accuracy: 0.388447
  position_accuracy_0.5m: 0.943922
  position_accuracy_1m: 0.995417
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.276049
  angle_accuracy_15deg: 0.388447
  angle_accuracy_30deg: 0.57254
repro:
  commit: ed6eef1b4fbfa5431d1ce40f3a010e90115b09fb
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=4
    model.num_task_layers=4 model.predict_canonical_pose=false data=chunked_multiview_sequence_bs8
    data.batch_size=8 training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256]
    loss=no_canonical training.trainer.max_epochs=200 training.early_stopping.enabled=false
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i560-nocanon-s4-h4
  predictions: knowledge/runs/run-i560-nocanon-s4-h4/pred_test.npz
  log: .training_queue/logs/1782307133770982559_638957_i545_nocanon_s4_h4.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_14
  curves: knowledge/runs/run-i560-nocanon-s4-h4/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_14
parents:
- run-i545-s4-h4
relations:
- to: run-i545-s4-h4
  rel: compares
- to: run-i560-nocanon-rs-s4-h4
  rel: compares
tags:
- plcs
- no-canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- loss-tuning
- strict-no-canonical
---

## 考察 / Findings

### 要約
S=4/H=4 balanced で strict no-canonical。位置 0.337m→**0.185m に大回復**（baseline [[run-i545-s4-h4]]、閾値 0.27m 以下）一方、回転は 51.91° に崩壊。H=2 の [[run-i560-nocanon-s5-h2]] と同じく、canonical/rotation 教師を外すと共有 trunk が位置に回り位置が戻る。

### アーキテクチャ詳細
`multiview_axial_split` H=4/S=4。`predict_canonical_pose=false`, `loss=no_canonical`（rotation≈0.02）。#545 の同 H/S balanced 最良点 [[run-i545-s4-h4]] と容量・データ同一。

### メトリクスの解釈
位置 mean 0.185 / median 0.138、位置@0.5m=0.944。回転 mean 51.91 / median 21.64、角@15=0.388 と崩壊。

### アーキテクチャ⇄メトリクスの因果考察
共有 H=4 が大きいほど canonical_rot 下では位置が痩せていたが、rotation/canonical 教師を外すと位置が 0.18m へ回復。共有が深いほど strict 化の位置改善幅も大きい（s4_h4: -45%, s5_h2: -47%）。回転崩壊は rotation≈0.02 ゆえ。criterion #1 を支持。

### 既存実験との比較
- baseline [[run-i545-s4-h4]]（8.23°/0.337m, #545 回転最良）から位置大回復・回転崩壊。
- Wave B [[run-i560-nocanon-rs-s4-h4]] で rotation を戻すと位置 0.402m に再悪化（canonical_rot 0.337m より更に悪い）かつ回転 9.49°。balanced は rot-strong でも位置が戻らず、共有 trunk 競合の残存コストを最も強く受ける。

### 次に有効な実験
- balanced は位置・回転両立が最も難しい配分。両立を狙うなら fully separate（[[run-i560-nocanon-rs-s6-h0]]）側に寄せるのが筋。
