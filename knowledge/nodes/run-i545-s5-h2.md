---
id: run-i545-s5-h2
type: run
title: i545_s5_h2
issue: 545
provider: codex
session: 019eed5b-bfed-7080-95ca-a09921dece32
date: '2026-06-23'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.342071
  position_error_std_m: 0.250758
  position_error_median_m: 0.295139
  angular_error_deg: 8.86616
  angular_error_std_deg: 8.572008
  angular_error_median_deg: 6.713905
  x_error_m: 0.136783
  y_error_m: 0.274002
  z_error_m: 0.047038
  position_accuracy: 0.818661
  angle_accuracy: 0.83505
  position_accuracy_0.5m: 0.818661
  position_accuracy_1m: 0.978526
  position_accuracy_2m: 0.99843
  angle_accuracy_10deg: 0.678513
  angle_accuracy_15deg: 0.83505
  angle_accuracy_30deg: 0.969986
repro:
  commit: 674818c567169bd3bee4bab17dd417a7308fdcc6
  branch: feat/training-queue-auto-prune-ckpt
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=2
    model.num_task_layers=5 data=chunked_multiview_sequence_bs8 data.batch_size=8
    data.seq_len_range=[64,256] training.trainer.accumulate_grad_batches=1 loss=canonical_rot
    training.trainer.max_epochs=200 training.early_stopping.enabled=false run.gpus=1
    run.resume=/home/kamimura/projects/wt/i545-prune/outputs/plcs/plcs_multiview_axial_split/logs/version_0/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-i545-s5-h2
  predictions: knowledge/runs/run-i545-s5-h2/pred_test.npz
  log: .training_queue/logs/1782137727219335489_7930_i545_s5_h2_resume.log
  curves: knowledge/runs/run-i545-s5-h2/curves.png
parents:
- run-i518-exp10
relations:
- to: run-i545-s4-h4
  rel: compares
- to: run-i539-ex10-chunked
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- chunked
- data-rich
- trunk-allocation
- param-matched
---

## 考察 / Findings

### 要約
パラメータ一定スイープ（77.845M）の **最分岐寄り端（S=5/H=2）**。回転 8.87°/位置 0.342m で、回転はスイープ 5 本中**最悪**。「分岐を厚くするほど良い（fully separate 最適）」説に反する観測。

### アーキテクチャ詳細
`multiview_axial_split`（`num_layers=2`=共有 H=2、`num_task_layers=5`=分岐 S=5）。H+2S=12、77.845M（EX10 と delta 0.000M）。本 run は **WSL2 swap crash で原ラン（version_0）が中断 → last.ckpt から resume（version_7）** して 200ep 完遂（codex セッションが resume を実行）。他はスイープ共通条件。

### メトリクスの解釈
test 回転 mean 8.87°/median 6.71°、位置 mean 0.342m/median 0.295m。位置精度 `@0.5m=0.819` は良好だが、回転は `@10°=0.679 / @15°=0.835` でスイープ最低。curves は resume 後区間（version_7）のみ可視化され、前半（version_0）は別 event のため連続曲線にはならない点に注意。

### アーキテクチャ⇄メトリクスの因果考察
共有 H=2 が薄すぎて低レベルの multiview 特徴学習が rot/pose 両分岐へ重複・分散し、回転 readout の効率が落ちた可能性（仮説）。最分岐構成にもかかわらず最良でない＝「分離が競合を断つから fully separate が最適」（#518）は data-rich・同容量では単純には成立しない。

### 既存実験との比較
親 [[run-i518-exp10]]（EX10 = fully separate, S=6）の延長線上の最分岐端点。バランス点 [[run-i545-s4-h4]]（8.23°/0.337m）に回転で明確に劣る（8.87>8.23）。早期終了版 [[run-i539-ex10-chunked]]（15.84°）は大幅に上回る。

### 次に有効な実験
未取得の S=6/H=0 端点を同条件で取得し、S=6→5→4 の回転カーブが端で悪化（＝中間最適）するかを確定する。
