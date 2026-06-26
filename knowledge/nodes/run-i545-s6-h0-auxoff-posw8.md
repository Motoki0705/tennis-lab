---
id: run-i545-s6-h0-auxoff-posw8
type: run
title: i545_s6_h0_auxoff_posw8
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
  position_error_m: 0.165563
  position_error_std_m: 0.150383
  position_error_median_m: 0.12174
  angular_error_deg: 8.45724
  angular_error_std_deg: 7.375236
  angular_error_median_deg: 6.583411
  x_error_m: 0.073695
  y_error_m: 0.122157
  z_error_m: 0.037072
  position_accuracy: 0.967776
  angle_accuracy: 0.850882
  position_accuracy_0.5m: 0.967776
  position_accuracy_1m: 0.991977
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.694072
  angle_accuracy_15deg: 0.850882
  angle_accuracy_30deg: 0.978517
repro:
  commit: a7bd9a7b4e3bb88a51fe536840a9b44b18d941ec
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=0
    model.num_task_layers=6 data=chunked_multiview_sequence_bs8 data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256] loss=canonical_rot
    loss.position_weight=8.0 loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0
    loss.torsion_angle_weight=0.0 loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0
    training.trainer.max_epochs=200 training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i545-s6-h0-auxoff-posw8
  predictions: knowledge/runs/run-i545-s6-h0-auxoff-posw8/pred_test.npz
  log: .training_queue/logs/1782305635279397115_634478_i545_s6_h0_auxoff_posw8.log
  curves: knowledge/runs/run-i545-s6-h0-auxoff-posw8/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_11
parents:
- run-i545-s6-h0
relations:
- to: run-i545-s6-h0
  rel: compares
- to: run-i545-s4-h4-posw8
  rel: compares
- to: run-i545-s4-h4-posw4
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
- position-best
---

## 考察 / Findings

### 要約
S=6/H=0 fully separate ＋ aux OFF ＋ `position_weight=8`。回転 8.46°/位置 **0.166m** で、**#545 全体の位置ベスト**（従来最良 [[run-i545-s6-h0]] の 0.186m を更新）。回転も無傷。同じ aux-off+posw8 が共有構成を崩壊させたのと対照的に、完全分岐では位置 8× が安全かつ有効。

### アーキテクチャ詳細
`multiview_axial_split` H=0/S=6（=EX10 相当, 77.845M）。aux 5 項=0、`position_weight=8.0`。rot/pose が trunk を共有しないため、position weight は pose(位置)分岐の勾配にのみ作用する。

### メトリクスの解釈
位置 mean 0.166 / median 0.122（baseline s6-h0: 0.186/0.158 を更新）、位置@0.5m=0.968。回転 mean 8.46 / median 6.58（baseline 8.96 から微改善）、角@15=0.851（baseline 0.840 から改善）。回転・位置・各精度すべてで s6-h0 を上回るか同等で、崩壊なし。

### アーキテクチャ⇄メトリクスの因果考察
fully separate では rot trunk と pose trunk が独立。`position_weight=8` は pose 分岐のみを強化し rot 分岐に干渉しないため、回転 supervision が飢えず崩壊しない。位置は強い重みでさらに最適化され 0.166m に到達（仮説: 共有が無い＝勾配競合が無いことが posw8 を安全にする）。これは [[run-i545-s4-h4-posw8]] 等の共有構成崩壊と表裏一体で、**「重い position weight は完全分岐とだけ両立」**を実証する。

### 既存実験との比較
- baseline [[run-i545-s6-h0]]（posw2, aux あり, 8.96°/0.186m）を回転・位置とも更新。
- 共有構成の posw8（[[run-i545-s4-h4-posw8]] 52°、[[run-i545-s5-h2-auxoff-posw8]] 35°）が全滅する中、唯一健全。
- balanced の位置最良 [[run-i545-s4-h4-posw4]]（0.258m）より大幅に良い位置。位置最良は依然 fully separate。

### 次に有効な実験
- s6-h0 ＋ posw を 4–8 で振り位置の下限を探る。
- 回転は balanced（8.0–8.2°）に僅差で劣る（8.46°）ため、s6-h0 ＋回転側 loss 強化（#560 `nocanon_rs` 系）で回転も詰められるか検証。
