---
id: run-i545-s4-h4-auxoff-posw8
type: run
title: i545_s4_h4_auxoff_posw8
issue: 545
provider: codex
session: 019ef7e7-03a8-7e53-9c92-21cdcb326a79
date: '2026-06-25'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.478371
  position_error_std_m: 0.360238
  position_error_median_m: 0.375713
  angular_error_deg: 53.260281
  angular_error_std_deg: 58.555275
  angular_error_median_deg: 24.758848
  x_error_m: 0.188828
  y_error_m: 0.399779
  z_error_m: 0.045276
  position_accuracy: 0.666581
  angle_accuracy: 0.363549
  position_accuracy_0.5m: 0.666581
  position_accuracy_1m: 0.898754
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.254272
  angle_accuracy_15deg: 0.363549
  angle_accuracy_30deg: 0.557754
repro:
  commit: 02024548fa0bb35e732b5d1fef92d77281a20b9a
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=4
    model.num_task_layers=4 data=chunked_multiview_sequence_bs8 data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256] loss=canonical_rot
    loss.position_weight=8.0 loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0
    loss.torsion_angle_weight=0.0 loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0
    training.trainer.max_epochs=200 training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i545-s4-h4-auxoff-posw8
  predictions: knowledge/runs/run-i545-s4-h4-auxoff-posw8/pred_test.npz
  log: .training_queue/logs/1782305635231131893_634524_i545_s4_h4_auxoff_posw8.log
  curves: knowledge/runs/run-i545-s4-h4-auxoff-posw8/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_9
parents:
- run-i545-s4-h4
relations:
- to: run-i545-s4-h4-auxoff
  rel: compares
- to: run-i545-s4-h4-posw8
  rel: compares
- to: run-i545-s6-h0-auxoff-posw8
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- loss-tuning
- aux-off
- position-weight
- collapse
---

## 考察 / Findings

### 要約
S=4/H=4 に aux 全 OFF ＋ `position_weight=8`。回転 53.26°/位置 0.478m と **本ウェーブ最悪の崩壊**。aux を外しても共有 trunk + posw8 の破綻は救えず、むしろ位置も最悪化。

### アーキテクチャ詳細
[[run-i545-s4-h4]] 同一構成に対し、aux 5 項=0 かつ `position_weight=8.0`（[[run-i545-s4-h4-auxoff]] と [[run-i545-s4-h4-posw8]] の合成条件）。

### メトリクスの解釈
回転 mean 53.26 / median 24.76、位置 mean 0.478 / median 0.376、角@15=0.364、位置@0.5m=0.667。全指標で batch A 最低。

### アーキテクチャ⇄メトリクスの因果考察
[[run-i545-s4-h4-posw8]] と同様、共有 trunk で posw8 が回転を破壊。さらに aux を外したことで pose 系の補助正則が無くなり、回転崩壊後の幾何整合が一層損なわれ位置も最悪化したと推測（仮説）。

### 既存実験との比較
- [[run-i545-s4-h4-posw8]]（aux あり posw8, 52.27°/0.368m）よりさらに悪い＝aux 除去は posw8 崩壊を悪化させる。
- 対照的に fully separate [[run-i545-s6-h0-auxoff-posw8]]（同 aux-off+posw8, 8.46°/0.166m）は最良位置＝崩壊は配分依存であって aux 除去依存ではない。

### 次に有効な実験
- 共有構成では posw8 を放棄。aux-off の効用は posw≤4 でのみ評価する。
