---
id: run-i535-asym-wide-rerun
type: run
title: i535_asym_wide_rerun
issue: 535
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-21'
status: done
config:
  model: multiview_axial_split_wide
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.894287
  position_error_std_m: 0.512892
  position_error_median_m: 0.808025
  angular_error_deg: 60.562759
  angular_error_std_deg: 52.051292
  angular_error_median_deg: 40.625477
  x_error_m: 0.441681
  y_error_m: 0.676823
  z_error_m: 0.05689
  position_accuracy: 0.256967
  angle_accuracy: 0.192595
  position_accuracy_0.5m: 0.256967
  position_accuracy_1m: 0.646212
  position_accuracy_2m: 0.9633
  angle_accuracy_10deg: 0.127633
  angle_accuracy_15deg: 0.192595
  angle_accuracy_30deg: 0.387282
repro:
  commit: 6d24b4dccad36a73bd56a526602757e16bac0275
  branch: exp/i525-asym
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split_wide loss=canonical_rot
    data=multiview_sequence data.batch_size=2 training.trainer.max_epochs=200 run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i535-asym-wide-rerun
  predictions: knowledge/runs/run-i535-asym-wide-rerun/pred_test.npz
  log: .training_queue/logs/1781994475592903754_1061691_i535_asym_wide_rerun.log
  curves: knowledge/runs/run-i535-asym-wide-rerun/curves.png
parents:
- run-i518-exp10
- run-i525-asym
relations:
- to: run-i540-asym-wide
  rel: supersedes
- to: run-i535-asym-deep16-rerun
  rel: compares
- to: run-i518-exp10
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- asymmetric
- width
- capacity-frontier
- valid-rerun
- under-converged
---

## 考察 / Findings

### 要約
正しい worktree で再実行した**有効**な非対称幅+深さ計測（rot10/pose6, hidden768, 228.7M）。batch2（VRAM 制約）+ 過大容量 × 固定小データで最適化に失敗し、回転 60.56°（ほぼランダム）。「幅が無効」ではなく**データ不足・未収束**の交絡。

### アーキテクチャ詳細
`multiview_axial_split_wide` + `canonical_rot` + `data=multiview_sequence`、200ep 設定。`hidden_dim 768`/`num_heads 12`、rotation trunk **10 層** / pose trunk **6 層** ＝ **228.7M params**（起動ログで確認）。`exp/i525-asym` worktree（commit `6d24b4d`）。VRAM 16GB の上限により物理 `data.batch_size=2`（基準 EX10 は 8）。no-op だった旧 [[run-i540-asym-wide]]（対称 768=172M）とは別物。

### メトリクスの解釈
test 回転 `60.56°`（median `40.63°`）/ 位置 `0.894m` ＝ ほぼ学習破綻（回転はランダム水準）。curves.png は **ep92 で early-stop**し、train ang すら `40.10°`（train loss `0.61`）止まりで val も追随せず ＝ **未収束（最適化失敗）**。過学習ではなく fit 自体に失敗している。

### アーキテクチャ⇄メトリクスの因果考察
228.7M を batch=2 で回したため、過小バッチで勾配が高分散になり、過大容量 × 固定小データ（≈800 窓）で最適化に失敗したと解釈（仮説）。**「容量（幅）が無効」の証拠ではない**点に注意。@Motoki0705 の「データ不足・未収束で精度が出ない」という指摘と整合する。

### 既存実験との比較
no-op だった [[run-i540-asym-wide]]（12.27°, 対称768）を `supersedes`（ただし本ランは未収束のため数値の直接比較は不可）。[[run-i518-exp10]]・[[run-i535-asym-deep16-rerun]] とも、未収束ゆえ公平比較にならない。

### 次に有効な実験
データ拡充（chunked）+ 勾配累積で effective batch を 8 に戻して再評価 → #539 [[run-i539-wide-chunked]]。そこでは wide が **10.33°/0.206m（位置は全ラン最良）**に転じ、本ランの崩壊が容量不足ではなく未収束（データ＋バッチ）由来だったことが裏付けられた。
