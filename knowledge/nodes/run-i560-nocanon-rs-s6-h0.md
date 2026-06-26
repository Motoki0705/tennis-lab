---
id: run-i560-nocanon-rs-s6-h0
type: run
title: i545_nocanon_rs_s6_h0
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
  position_error_m: 0.20671
  position_error_std_m: 0.192729
  position_error_median_m: 0.154232
  angular_error_deg: 8.487839
  angular_error_std_deg: 8.148999
  angular_error_median_deg: 6.274662
  x_error_m: 0.084356
  y_error_m: 0.164728
  z_error_m: 0.035684
  position_accuracy: 0.95941
  angle_accuracy: 0.857903
  position_accuracy_0.5m: 0.95941
  position_accuracy_1m: 0.990374
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.706852
  angle_accuracy_15deg: 0.857903
  angle_accuracy_30deg: 0.970382
repro:
  commit: ed6eef1b4fbfa5431d1ce40f3a010e90115b09fb
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=0
    model.num_task_layers=6 model.predict_canonical_pose=false data=chunked_multiview_sequence_bs8
    data.batch_size=8 training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256]
    loss=no_canonical loss.rotation_weight=0.5 +loss.angle_weight=1.0 training.trainer.max_epochs=200
    training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i560-nocanon-rs-s6-h0
  predictions: knowledge/runs/run-i560-nocanon-rs-s6-h0/pred_test.npz
  log: .training_queue/logs/1782307133805460326_638987_i545_nocanon_rs_s6_h0.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_15
  curves: knowledge/runs/run-i560-nocanon-rs-s6-h0/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_15
parents:
- run-i560-nocanon-s6-h0
relations:
- to: run-i560-nocanon-s6-h0
  rel: compares
- to: run-i545-s6-h0
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
S=6/H=0 fully separate で canonical head を消したまま rotation supervision を `canonical_rot` 相当（`rotation_weight=0.5`, `angle_weight=1.0`）に戻した Wave B 端点。回転 8.49°/位置 0.207m で**両立**。canonical pose head 無しでも回転は十分育つ（head 不要を示す）。

### アーキテクチャ詳細
`multiview_axial_split` H=0/S=6。`predict_canonical_pose=false`, `loss=no_canonical` ＋ `rotation_weight=0.5` `+angle_weight=1.0`。strict 版 [[run-i560-nocanon-s6-h0]] との差は rotation/angle weight のみ。

### メトリクスの解釈
回転 mean 8.49 / median 6.27、角@15=0.858（strict 0.631 から回復）。位置 mean 0.207 / median 0.154、位置@0.5m=0.959。strict（14.66°/0.162m）比で回転大改善・位置微悪化。

### アーキテクチャ⇄メトリクスの因果考察
完全分岐では rotation を強めても rot 分岐内で完結し位置分岐への干渉が小さい。よって rotation 0.02→0.5 で回転が 14.66→8.49° と回復しつつ位置は 0.162→0.207m の小悪化に留まる（仮説）。canonical pose head を消しても rotation 教師さえ十分なら回転は育つ＝**回転は head ではなく supervision 強度律速**。

### 既存実験との比較
- baseline [[run-i545-s6-h0]]（canonical_rot, 8.96°/0.186m）と同等水準（回転やや良 8.49<8.96、位置やや悪 0.207>0.186）。canonical head の有無は fully separate では大差なし。
- strict [[run-i560-nocanon-s6-h0]] から回転を取り戻した対。共有構成（[[run-i560-nocanon-rs-s4-h4]] 0.402m）と違い位置劣化が小さいのが完全分岐の利点。

### 次に有効な実験
- fully separate × rot-strong に position_weight 増量（#545 の posw 知見）を重ね、回転 8.5°・位置 0.17m 級の同時達成を狙う。
