---
id: run-i560-nocanon-s6-h0
type: run
title: i545_nocanon_s6_h0
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
  position_error_m: 0.161719
  position_error_std_m: 0.195093
  position_error_median_m: 0.115864
  angular_error_deg: 14.658255
  angular_error_std_deg: 13.974393
  angular_error_median_deg: 11.3133
  x_error_m: 0.061784
  y_error_m: 0.131987
  z_error_m: 0.030145
  position_accuracy: 0.973754
  angle_accuracy: 0.630532
  position_accuracy_0.5m: 0.973754
  position_accuracy_1m: 0.98953
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.442343
  angle_accuracy_15deg: 0.630532
  angle_accuracy_30deg: 0.89732
repro:
  commit: a7bd9a7b4e3bb88a51fe536840a9b44b18d941ec
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=0
    model.num_task_layers=6 model.predict_canonical_pose=false data=chunked_multiview_sequence_bs8
    data.batch_size=8 training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256]
    loss=no_canonical training.trainer.max_epochs=200 training.early_stopping.enabled=false
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i560-nocanon-s6-h0
  predictions: knowledge/runs/run-i560-nocanon-s6-h0/pred_test.npz
  log: .training_queue/logs/1782307133735453817_638927_i545_nocanon_s6_h0.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_12
  curves: knowledge/runs/run-i560-nocanon-s6-h0/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_12
parents:
- run-i545-s6-h0
relations:
- to: run-i545-s6-h0
  rel: compares
- to: run-i560-nocanon-rs-s6-h0
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
S=6/H=0 fully separate で canonical pose head を消し（`predict_canonical_pose=false`, `loss=no_canonical`, rotation 弱め ≈0.02）位置・回転の二項に絞った Wave A 端点。位置 0.162m（baseline [[run-i545-s6-h0]] 0.186m を更新）だが、回転は 14.66° と弱い（rotation supervision がほぼ off のため）。

### アーキテクチャ詳細
`multiview_axial_split` H=0/S=6。`model.predict_canonical_pose=false` で canonical pose 出力ヘッドを除去し、`loss=no_canonical`（pose naturalness aux なし・rotation_weight≈0.02 の弱 rotation + position の二項）。data/batch/seq/early-stop は #545 と共通。

### メトリクスの解釈
位置 mean 0.162 / median 0.116、位置@0.5m=0.974 で位置は最良級。回転 mean 14.66 / median 11.31、角@15=0.631 / @10=0.442 と弱い。rotation を 0.02 までほぼ切ったので、完全分岐でも回転が育たない。

### アーキテクチャ⇄メトリクスの因果考察
fully separate なので位置と回転は別 trunk。rotation_weight≈0.02 では rot 分岐が十分な教師を受けず回転が伸びない一方、位置分岐は干渉なく最適化され 0.162m に到達（仮説）。canonical head 除去自体は位置をむしろ僅かに改善（0.186→0.162）。

### 既存実験との比較
- baseline [[run-i545-s6-h0]]（canonical_rot, 8.96°/0.186m）に対し位置改善・回転大幅劣化。回転劣化は rotation 弱設定が主因。
- Wave B の [[run-i560-nocanon-rs-s6-h0]]（rotation_weight=0.5 に戻す）では回転が 8.49° に回復し位置 0.207m。回転は supervision 強度律速で、canonical head の有無ではない。

### 次に有効な実験
- fully separate なら rotation を強めても位置が崩れにくいはずで、[[run-i560-nocanon-rs-s6-h0]] が実際に両立寄り。posw を足して位置下限を狙う余地。
