---
id: run-i560-nocanon-s5-h2
type: run
title: i545_nocanon_s5_h2
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
  position_error_m: 0.181147
  position_error_std_m: 0.129647
  position_error_median_m: 0.148799
  angular_error_deg: 49.217514
  angular_error_std_deg: 55.302536
  angular_error_median_deg: 24.192024
  x_error_m: 0.079006
  y_error_m: 0.140189
  z_error_m: 0.032788
  position_accuracy: 0.976325
  angle_accuracy: 0.366083
  position_accuracy_0.5m: 0.976325
  position_accuracy_1m: 1.0
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.260607
  angle_accuracy_15deg: 0.366083
  angle_accuracy_30deg: 0.565884
repro:
  commit: ed6eef1b4fbfa5431d1ce40f3a010e90115b09fb
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=2
    model.num_task_layers=5 model.predict_canonical_pose=false data=chunked_multiview_sequence_bs8
    data.batch_size=8 training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256]
    loss=no_canonical training.trainer.max_epochs=200 training.early_stopping.enabled=false
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i560-nocanon-s5-h2
  predictions: knowledge/runs/run-i560-nocanon-s5-h2/pred_test.npz
  log: .training_queue/logs/1782307133753706151_638942_i545_nocanon_s5_h2.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_13
  curves: knowledge/runs/run-i560-nocanon-s5-h2/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_13
parents:
- run-i545-s5-h2
relations:
- to: run-i545-s5-h2
  rel: compares
- to: run-i560-nocanon-rs-s5-h2
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
S=5/H=2（少量共有）で strict no-canonical。**位置が 0.342m→0.181m に大回復**（baseline [[run-i545-s5-h2]]、判定閾値 0.27m を大きく下回る）一方、回転は 49.22° に崩壊。canonical head を消し rotation をほぼ切ると、共有 trunk H=2 が位置に専念でき位置が戻る——という #560 の核心を示す代表点。

### アーキテクチャ詳細
`multiview_axial_split` H=2/S=5。`predict_canonical_pose=false`, `loss=no_canonical`（rotation≈0.02）。#545 の同 H/S [[run-i545-s5-h2]] と容量・データ同一で、差は loss/head のみ。

### メトリクスの解釈
位置 mean 0.181 / median 0.149、位置@0.5m=0.976（baseline 0.342m/0.819 から大幅改善）。回転 mean 49.22 / median 24.19、角@15=0.366 と崩壊。

### アーキテクチャ⇄メトリクスの因果考察
canonical_rot では rotation/canonical 系の教師が共有 trunk H=2 を引っ張り位置を痩せさせていた。これらを外す（rotation≈0.02）と共有 trunk が位置最適化に専念し位置が 0.18m へ回復（仮説）。裏返しに rotation 教師がほぼ無いので回転は崩壊。→ H>0 の位置劣化は **canonical head の存在ではなく rotation 系教師の共有 trunk 競合** が主因、という criterion #1 を支持。

### 既存実験との比較
- baseline [[run-i545-s5-h2]]（8.87°/0.342m）から位置 -47%・回転崩壊。
- Wave B [[run-i560-nocanon-rs-s5-h2]]（rotation を 0.5 に戻す）では位置が 0.332m に再悪化し回転 7.54°。**strict で位置改善 → rot-strong で再悪化** ＝ criterion #2（rotation supervision 強度が H>0 位置劣化の主要因）を裏づける対。

### 次に有効な実験
- 位置・回転の両立は H=2 では loss 配分のみでは達成困難。rotation_weight を 0.02 と 0.5 の間で振り、位置 0.27m 以下を保てる回転上限を探る。
