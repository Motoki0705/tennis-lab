---
id: run-i545-s5-h2-auxoff-posw8
type: run
title: i545_s5_h2_auxoff_posw8
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
  position_error_m: 0.45626
  position_error_std_m: 0.455844
  position_error_median_m: 0.364374
  angular_error_deg: 35.29166
  angular_error_std_deg: 40.858578
  angular_error_median_deg: 19.737347
  x_error_m: 0.170455
  y_error_m: 0.381798
  z_error_m: 0.048788
  position_accuracy: 0.699828
  angle_accuracy: 0.395976
  position_accuracy_0.5m: 0.699828
  position_accuracy_1m: 0.939953
  position_accuracy_2m: 0.979576
  angle_accuracy_10deg: 0.288702
  angle_accuracy_15deg: 0.395976
  angle_accuracy_30deg: 0.620645
repro:
  commit: 02024548fa0bb35e732b5d1fef92d77281a20b9a
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=2
    model.num_task_layers=5 data=chunked_multiview_sequence_bs8 data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256] loss=canonical_rot
    loss.position_weight=8.0 loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0
    loss.torsion_angle_weight=0.0 loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0
    training.trainer.max_epochs=200 training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i545-s5-h2-auxoff-posw8
  predictions: knowledge/runs/run-i545-s5-h2-auxoff-posw8/pred_test.npz
  log: .training_queue/logs/1782305635254381729_634539_i545_s5_h2_auxoff_posw8.log
  curves: knowledge/runs/run-i545-s5-h2-auxoff-posw8/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_10
parents:
- run-i545-s5-h2
relations:
- to: run-i545-s5-h2
  rel: compares
- to: run-i545-s4-h4-auxoff-posw8
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
S=5/H=2（#545 が「少量共有を戻した再確認点」として挙げた配分）＋ aux OFF ＋ `position_weight=8`。回転 35.29°/位置 0.456m で **崩壊（ただし H=4 より軽い）**。「少量共有 S=5/H=2 で回転改善」狙いは posw8 条件では成立せず、浅い共有 H=2 でも posw8 は回転を壊す。

### アーキテクチャ詳細
`multiview_axial_split` H=2/S=5（77.845M）。aux 5 項=0、`position_weight=8.0`。issue #545 が次手に挙げた「少量共有を戻した S=5/H=2 seed 再確認」に batch A の aux-off+posw8 処理を適用した構成。

### メトリクスの解釈
回転 mean 35.29 / median 19.74（baseline [[run-i545-s5-h2]] 8.87° から大崩壊）、位置 0.456 / 0.364、角@15=0.396、位置@0.5m=0.700。

### アーキテクチャ⇄メトリクスの因果考察
H=2 と共有が浅くても posw8 は共有勾配を位置支配にし回転を壊す。崩壊度は H=4（52–53°）> H=2（35°）> H=0（崩壊せず）で、**共有深さ H が大きいほど posw8 崩壊が重い**という単調関係を示す（仮説）。

### 既存実験との比較
- baseline [[run-i545-s5-h2]]（posw2, aux あり, 8.87°/0.342m）から大幅劣化。
- [[run-i545-s4-h4-auxoff-posw8]]（H=4, 53.26°）より軽症、[[run-i545-s6-h0-auxoff-posw8]]（H=0, 8.46°）は無傷。→ H に対する崩壊の単調性を裏づける中間点。

### 次に有効な実験
- S=5/H=2 の回転改善は posw8 ではなく posw≤4 ＋別 loss（rotation/angle 再重み）で狙う（#560 の `nocanon_rs_s5_h2` が 7.54° で有望）。
