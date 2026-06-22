---
id: run-i535-asym-deep16-rerun
type: run
title: i535_asym_deep16_rerun
issue: 535
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-21'
status: done
config:
  model: multiview_axial_split_asym_deep16
  loss: canonical_rot
  data: multiview_sequence
metrics:
  position_error_m: 0.252409
  position_error_std_m: 0.273971
  position_error_median_m: 0.173325
  angular_error_deg: 10.410616
  angular_error_std_deg: 10.222599
  angular_error_median_deg: 8.136292
  x_error_m: 0.10378
  y_error_m: 0.206081
  z_error_m: 0.03868
  position_accuracy: 0.901963
  angle_accuracy: 0.791273
  position_accuracy_0.5m: 0.901963
  position_accuracy_1m: 0.974953
  position_accuracy_2m: 0.996217
  angle_accuracy_10deg: 0.606614
  angle_accuracy_15deg: 0.791273
  angle_accuracy_30deg: 0.960668
repro:
  commit: 6d24b4dccad36a73bd56a526602757e16bac0275
  branch: exp/i525-asym
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split_asym_deep16 loss=canonical_rot
    data=multiview_sequence data.batch_size=4 training.trainer.max_epochs=200 run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i535-asym-deep16-rerun
  predictions: knowledge/runs/run-i535-asym-deep16-rerun/pred_test.npz
  log: .training_queue/logs/1781994475574624725_1061676_i535_asym_deep16_rerun.log
  curves: knowledge/runs/run-i535-asym-deep16-rerun/curves.png
parents:
- run-i518-exp10
- run-i525-asym
relations:
- to: run-i540-asym-deep16
  rel: supersedes
- to: run-i518-exp10
  rel: compares
- to: run-i535-asym-wide-rerun
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- asymmetric
- depth
- capacity-frontier
- valid-rerun
---

## 考察 / Findings

### 要約
正しい worktree で再実行した**有効**な非対称深さ計測（rot16/pose6, 142.3M）。回転 10.41°/位置 0.252m で、EX10 の ~1.8倍の容量を投じても EX10（9.98°/0.238m）に届かない。#535 当初の「深さが回転の主レバー」は no-op バグ由来で撤回。

### アーキテクチャ詳細
`multiview_axial_split_asym_deep16` + `canonical_rot` + `data=multiview_sequence`、200ep。rotation trunk **16 層** / pose trunk **6 層**、`hidden_dim 512`/`num_heads 8` ＝ **142.3M params**（起動ログで確認）。`rot_num_task_layers` を honor する `exp/i525-asym` worktree（commit `6d24b4d`, cwd=`wt/i525-asym`）で実行。VRAM 制約で `data.batch_size=4`（基準 EX10 は 8）。no-op だった旧 [[run-i540-asym-deep16]]（78.1M=対称 EX10 相当）とは別物。

### メトリクスの解釈
test 回転 `10.41°`（median `8.14°`）/ 位置 `0.252m`（median `0.173m`）。curves.png は train/val とも 200ep を完走し右肩下がり・崩壊なし（full 収束）。ただし train ang `8.16°`（train loss `0.06`）に対し val/test `10.41°` と乖離 ＝ 軽い過学習。

### アーキテクチャ⇄メトリクスの因果考察
固定小データ（≈800 窓/ep を 200ep 反復＝記憶可能）では、rotation trunk を 6→16 に深めた +64M の余剰容量を汎化に活かせず過学習方向に出たと解釈（仮説）。rot 深さ系列（幅512）は EX10 6層 `9.98°` < deep16 16層 `10.41°` < asym 10層 `19.94°` と**非単調**で、「深さで EX10 を超える」当初仮説は不成立。

### 既存実験との比較
[[run-i518-exp10]]（EX10, 78.1M, 9.98°/0.238m）に両指標で僅かに劣る（`compares`）。幅振りの [[run-i535-asym-wide-rerun]]（228.7M, 60.56°）よりは大幅に良いが、wide が batch2 で未収束のため公平な対比ではない。no-op だった [[run-i540-asym-deep16]] を `supersedes`。

### 次に有効な実験
固定小データでは容量増の効果が見えないため、データ拡充（chunked）+ 適正バッチで再評価 → #539 [[run-i539-deep16-chunked]]。そこでも deep16 は本群最下位で、深さ偏重は data-rich でも非推奨と判明した。
