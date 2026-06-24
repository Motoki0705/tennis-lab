---
id: run-i545-s6-h0
type: run
title: i545_s6_h0
issue: 545
provider: codex
session: fabaa2ad-e86a-4a2b-b14b-6ce71d148c0a
date: '2026-06-23'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.186016
  position_error_std_m: 0.137102
  position_error_median_m: 0.158319
  angular_error_deg: 8.960709
  angular_error_std_deg: 11.506378
  angular_error_median_deg: 6.014357
  x_error_m: 0.077715
  y_error_m: 0.143441
  z_error_m: 0.037327
  position_accuracy: 0.966738
  angle_accuracy: 0.839746
  position_accuracy_0.5m: 0.966738
  position_accuracy_1m: 0.997027
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.710904
  angle_accuracy_15deg: 0.839746
  angle_accuracy_30deg: 0.967467
repro:
  commit: 674818c567169bd3bee4bab17dd417a7308fdcc6
  branch: feat/training-queue-auto-prune-ckpt
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=0
    model.num_task_layers=6 data.batch_size=8 training.trainer.accumulate_grad_batches=1
    data=chunked_multiview_sequence_bs8 data.seq_len_range=[64,256] loss=canonical_rot
    training.trainer.max_epochs=200 training.early_stopping.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i545-s6-h0
  predictions: knowledge/runs/run-i545-s6-h0/pred_test.npz
  log: .training_queue/logs/1782217908028700708_292095_i545_s6_h0.log
  curves: knowledge/runs/run-i545-s6-h0/curves.png
parents:
- run-i518-exp10
relations:
- to: run-i545-s4-h4
  rel: compares
- to: run-i539-ex10-chunked
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
- param-matched
- endpoint
---

## 考察 / Findings

### 要約
H+2S=12 の fully separate 端点（S=6/H=0, EX10 相当）を、#545 と同一の data-rich / no-early-stop / 200ep プロトコルで取り直した run。回転は 8.96°（median 6.01°）で S=4/H=4 より悪いが、位置は 0.186m（median 0.158m）でスイープ全体および wide balanced を大きく上回った。

### アーキテクチャ詳細
`multiview_axial_split`（`num_layers=0`=共有 trunk H、`num_task_layers=6`=rot/pose 各分岐 trunk S）。forward 総層適用 = H + 2S = 12 で、#545 の S=5..1 と同じく 77.845M params。共有 trunk を完全に消し、rot/pose trunk を最初から分離する fully separate 構成。`data=chunked_multiview_sequence_bs8`、effective batch=8、`seq_len_range=[64,256]`、`loss=canonical_rot`、early-stop OFF / 200ep 完遂。

### メトリクスの解釈
test 回転 mean 8.96°/median 6.01°、位置 mean 0.186m/median 0.158m。角度精度 `@15°=0.840` は S=4/H=4 の 0.871 を下回る一方、位置精度 `@0.5m=0.967` は既存 S=5..1 の最良 0.828 を大幅に上回る。#545 の端点欠落を埋めた結果、回転最適と位置最適が分離した。

### アーキテクチャ⇄メトリクスの因果考察
観測として、位置は共有 trunk を入れない fully separate で最良になった。位置推定では rot/pose の特徴競合を早期から切ること、あるいは pose 側 trunk に十分な深さを残すことが効いている可能性が高い（仮説）。一方で回転は S=4/H=4 より悪く、低レベル multiview/temporal 表現を一定量共有することが回転には有利だったと解釈できる。

### 既存実験との比較
[[run-i545-s4-h4]] に対し、回転は 8.96° > 8.23° で悪化、位置は 0.186m < 0.337m で大幅改善。したがって「balanced が全指標最良」という前回結論は更新され、**回転最良は balanced、位置最良は fully separate**。[[run-i539-ex10-chunked]]（early-stop あり, 15.84°/0.542m）とは同じ EX10 系でも本 run が大幅に良く、#539 の early-stop 交絡をさらに強く裏づける。[[run-i539-wide-chunked]]（228.7M, 10.33°/0.206m）より小さい 77.8M でも、回転・位置とも本 run が上回った。

### 次に有効な実験
回転最良の balanced と位置最良の fully separate が分かれたため、次は multi-objective の選び方を明確化する。実験としては、S=6/H=0 を基準に回転側だけ共有を少量戻す S=5/H=2 の再現性確認、または loss weight / head capacity を調整して fully separate の回転を改善できるかを試す価値がある。
