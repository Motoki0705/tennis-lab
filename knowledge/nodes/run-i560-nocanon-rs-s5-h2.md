---
id: run-i560-nocanon-rs-s5-h2
type: run
title: i545_nocanon_rs_s5_h2
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
  position_error_m: 0.331787
  position_error_std_m: 0.306287
  position_error_median_m: 0.241582
  angular_error_deg: 7.538045
  angular_error_std_deg: 7.279468
  angular_error_median_deg: 5.50921
  x_error_m: 0.12126
  y_error_m: 0.272784
  z_error_m: 0.044502
  position_accuracy: 0.807556
  angle_accuracy: 0.888919
  position_accuracy_0.5m: 0.807556
  position_accuracy_1m: 0.971347
  position_accuracy_2m: 0.993803
  angle_accuracy_10deg: 0.748317
  angle_accuracy_15deg: 0.888919
  angle_accuracy_30deg: 0.981275
repro:
  commit: ed6eef1b4fbfa5431d1ce40f3a010e90115b09fb
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=2
    model.num_task_layers=5 model.predict_canonical_pose=false data=chunked_multiview_sequence_bs8
    data.batch_size=8 training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256]
    loss=no_canonical loss.rotation_weight=0.5 +loss.angle_weight=1.0 training.trainer.max_epochs=200
    training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i560-nocanon-rs-s5-h2
  predictions: knowledge/runs/run-i560-nocanon-rs-s5-h2/pred_test.npz
  log: .training_queue/logs/1782307133822773982_639002_i545_nocanon_rs_s5_h2.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_16
  curves: knowledge/runs/run-i560-nocanon-rs-s5-h2/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_16
parents:
- run-i560-nocanon-s5-h2
relations:
- to: run-i560-nocanon-s5-h2
  rel: compares
- to: run-i545-s5-h2
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
- rotation-best
---

## 考察 / Findings

### 要約
S=5/H=2（少量共有）で canonical head を消したまま rot-strong（`rotation_weight=0.5`, `angle_weight=1.0`）。回転 **7.54° / median 5.51°** で **#545+#560 全体の回転ベスト**（従来最良 [[run-i545-s4-h4]] 8.23° を更新）。ただし位置は 0.332m と凡庸（strict 0.181m から再悪化）。回転フロンティアの代表点。

### アーキテクチャ詳細
`multiview_axial_split` H=2/S=5。`predict_canonical_pose=false`, `loss=no_canonical` ＋ `rotation_weight=0.5` `+angle_weight=1.0`。strict 版 [[run-i560-nocanon-s5-h2]] との差は rotation/angle weight のみ。

### メトリクスの解釈
回転 mean 7.54 / median 5.51、角@15=0.889 / @10=0.748 でいずれも研究内最高。位置 mean 0.332 / median 0.242、位置@0.5m=0.808。

### アーキテクチャ⇄メトリクスの因果考察
浅い共有 H=2 ＋ 厚い分岐 S=5 に強い rotation 教師を与えると、回転 readout が最大化され 7.54° に到達（仮説: canonical head を外したことで rotation 教師がより素直に効く）。同時に共有 trunk が rotation に引かれ位置は 0.181→0.332m に再悪化＝**rotation supervision 強度と位置が共有 trunk 上でトレードオフ**。

### 既存実験との比較
- 回転で #545 balanced 最良 [[run-i545-s4-h4]]（8.23°/0.337m）を更新（7.54<8.23）。位置はほぼ同等（0.332≈0.337）。
- strict [[run-i560-nocanon-s5-h2]]（49.22°/0.181m）から回転を取り戻すと位置が再悪化する対＝criterion #2 の最も鮮明な例。
- canonical_rot baseline [[run-i545-s5-h2]]（8.87°/0.342m）より回転良・位置同等。canonical head 無し＋rot-strong が回転に最適。

### 次に有効な実験
- 回転最優先なら本構成。位置も要るなら fully separate（[[run-i560-nocanon-rs-s6-h0]]）と用途で使い分け。H=2 で位置を 0.27m に戻せる rotation_weight の中間点探索。
