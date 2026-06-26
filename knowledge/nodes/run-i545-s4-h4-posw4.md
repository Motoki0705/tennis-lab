---
id: run-i545-s4-h4-posw4
type: run
title: i545_s4_h4_posw4
issue: 545
provider: codex
session: 019ef7e7-03a8-7e53-9c92-21cdcb326a79
date: '2026-06-24'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.257621
  position_error_std_m: 0.180859
  position_error_median_m: 0.210396
  angular_error_deg: 8.009185
  angular_error_std_deg: 7.678938
  angular_error_median_deg: 6.089533
  x_error_m: 0.106789
  y_error_m: 0.202751
  z_error_m: 0.043039
  position_accuracy: 0.911214
  angle_accuracy: 0.857025
  position_accuracy_0.5m: 0.911214
  position_accuracy_1m: 0.993293
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.722758
  angle_accuracy_15deg: 0.857025
  angle_accuracy_30deg: 0.983584
repro:
  commit: 8ce84fc04df94fba6a0489133a67914859ed5dbf
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=4
    model.num_task_layers=4 data=chunked_multiview_sequence_bs8 data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256] loss=canonical_rot
    loss.position_weight=4.0 training.trainer.max_epochs=200 training.early_stopping.enabled=false
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i545-s4-h4-posw4
  predictions: knowledge/runs/run-i545-s4-h4-posw4/pred_test.npz
  log: .training_queue/logs/1782305635167127164_634479_i545_s4_h4_posw4.log
  curves: knowledge/runs/run-i545-s4-h4-posw4/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_6
parents:
- run-i545-s4-h4
relations:
- to: run-i545-s4-h4
  rel: compares
- to: run-i545-s4-h4-posw8
  rel: compares
- to: run-i545-s4-h4-auxoff
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- loss-tuning
- position-weight
---

## 考察 / Findings

### 要約
S=4/H=4 balanced に `position_weight` を canonical_rot 既定 2.0 → 4.0 へ倍増。回転 8.01°（median 6.09°）/位置 0.258m で、**回転・位置とも balanced baseline [[run-i545-s4-h4]]（8.23°/0.337m）を同時に上回る**。canonical_rot の枠内で位置を中庸に増やすのが #545 balanced 点の最も素直な改善。

### アーキテクチャ詳細
`multiview_axial_split` H=4/S=4（=[[run-i545-s4-h4]], 77.845M）と同一容量・同一データ（`chunked_multiview_sequence_bs8`, eff batch=8, `seq_len_range=[64,256]`, early-stop OFF / 200ep 完遂）。唯一の差は `loss.position_weight` 2.0→4.0。rotation/angle/canonical aux の各 weight は不変。

### メトリクスの解釈
回転 mean 8.01 / median 6.09、位置 mean 0.258 / median 0.210。角@15=0.857（baseline 0.871 から微減）に対し位置@0.5m=0.911（baseline 0.828 から大幅改善）。位置 mean は 0.337→0.258（約 -23%）、回転 mean も 8.23→8.01 と僅かに改善し崩壊・過学習なし。

### アーキテクチャ⇄メトリクスの因果考察
balanced 配分では rot/pose が共有 trunk H=4 を介して勾配を競合する。position_weight を 2→4 と中庸に上げると共有 trunk が位置寄りに最適化され位置が大きく改善する一方、回転を壊さない閾値以下に収まる（仮説: 4× は競合を位置側へ傾けるが rotation supervision を飢えさせない）。同じ balanced で 8× にした [[run-i545-s4-h4-posw8]] が回転 52° に崩壊するのと表裏一体。

### 既存実験との比較
- baseline [[run-i545-s4-h4]]（posw2, 8.23°/0.337m）を回転・位置とも上回る。
- aux-off の [[run-i545-s4-h4-auxoff]]（8.16°/0.304m）より位置が良い（0.258<0.304）＝位置改善は aux 除去より posw 増量の方が効く。両者は独立レバーで併用余地あり。
- 位置単独では fully separate [[run-i545-s6-h0]]（0.186m）/[[run-i545-s6-h0-auxoff-posw8]]（0.166m）に及ばないが、balanced で回転を保ったまま位置を 0.26m まで詰められる点が価値。

### 次に有効な実験
- posw を 4 と 8 の間（5–6）で振り、回転崩壊の閾値と位置改善の上限を特定する。
- balanced + posw4 に rotation/angle 再重み付け（#560 の `nocanon_rs` 系）を重ね、回転をさらに押し下げられるか検証。
