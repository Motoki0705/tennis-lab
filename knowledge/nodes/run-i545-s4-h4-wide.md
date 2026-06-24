---
id: run-i545-s4-h4-wide
type: run
title: i545_s4_h4_wide
issue: 545
provider: codex
session: fabaa2ad-e86a-4a2b-b14b-6ce71d148c0a
date: '2026-06-24'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.360726
  position_error_std_m: 0.345223
  position_error_median_m: 0.270025
  angular_error_deg: 8.309699
  angular_error_std_deg: 7.62322
  angular_error_median_deg: 6.418861
  x_error_m: 0.142923
  y_error_m: 0.295563
  z_error_m: 0.047402
  position_accuracy: 0.82632
  angle_accuracy: 0.855257
  position_accuracy_0.5m: 0.82632
  position_accuracy_1m: 0.958705
  position_accuracy_2m: 0.990354
  angle_accuracy_10deg: 0.70123
  angle_accuracy_15deg: 0.855257
  angle_accuracy_30deg: 0.980166
repro:
  commit: 674818c567169bd3bee4bab17dd417a7308fdcc6
  branch: feat/training-queue-auto-prune-ckpt
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.hidden_dim=768
    model.num_heads=12 model.num_layers=4 model.num_task_layers=4 data.batch_size=2
    training.trainer.accumulate_grad_batches=4 data=chunked_multiview_sequence_bs8
    data.seq_len_range=[64,256] loss=canonical_rot training.trainer.max_epochs=200
    training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i545-s4-h4-wide
  predictions: knowledge/runs/run-i545-s4-h4-wide/pred_test.npz
  log: .training_queue/logs/1782217908061887471_292125_i545_s4_h4_wide.log
  curves: knowledge/runs/run-i545-s4-h4-wide/curves.png
parents:
- run-i545-s4-h4
relations:
- to: run-i545-s6-h0
  rel: compares
- to: run-i539-wide-chunked
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- wide
---

## 考察 / Findings

### 要約
前回最良だった S=4/H=4 balanced 配分を hidden_dim=768 / heads=12 に広げた follow-up。params は 171.6M。回転は 8.31°で narrow balanced の 8.23°とほぼ同等だが、位置は 0.361m で改善せず、むしろ narrow の 0.337m より悪化した。

### アーキテクチャ詳細
`multiview_axial_split`（`hidden_dim=768`, `num_heads=12`, `num_layers=4`, `num_task_layers=4`）。H/S 配分は [[run-i545-s4-h4]] と同じ balanced だが、幅を増やして 171.6M params にした構成。VRAM 対応のため physical batch=2 / accumulate=4 で effective batch=8 を維持。data/loss/epoch/early-stop 条件は #545 共通。

### メトリクスの解釈
test 回転 mean 8.31°/median 6.42°、位置 mean 0.361m/median 0.270m。角度精度 `@15°=0.855`、位置精度 `@0.5m=0.826`。narrow balanced と比べ、回転 mean は +0.08°、位置 mean は +0.024m で、容量追加による改善は見られない。

### アーキテクチャ⇄メトリクスの因果考察
balanced 配分のまま幅だけ増やしても位置律速は解けなかった。位置の律速は単純な総容量ではなく、pose 側 trunk の独立深さ、batch dynamics、または loss/head の設計にある可能性が高い（仮説）。effective batch は同じだが physical batch が 8 から 2 に下がっており、BatchNorm 等は無いとしても optimizer step あたりの勾配統計やメモリ制約由来の挙動差は残る。

### 既存実験との比較
親の [[run-i545-s4-h4]]（8.23°/0.337m）を上回れなかった。[[run-i545-s6-h0]]（8.96°/0.186m）と比べると回転は良いが、位置は大幅に悪い。[[run-i539-wide-chunked]]（228.7M, 10.33°/0.206m）より回転は良いが、位置は劣る。したがって「balanced に幅を足せば回転と位置が同時改善する」という次手仮説は否定寄り。

### 次に有効な実験
wide 化の方向を続けるなら、balanced ではなく S=6/H=0 側を wide にして位置最良を保ったまま回転を改善できるかを確認する方が有効。別案として、S=4/H=4 では head/loss 調整を先に試し、単純な hidden_dim 拡大は優先度を下げる。
