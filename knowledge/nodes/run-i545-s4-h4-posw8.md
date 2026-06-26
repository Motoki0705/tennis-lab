---
id: run-i545-s4-h4-posw8
type: run
title: i545_s4_h4_posw8
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
  position_error_m: 0.367864
  position_error_std_m: 0.258331
  position_error_median_m: 0.314441
  angular_error_deg: 52.269588
  angular_error_std_deg: 59.069241
  angular_error_median_deg: 22.104662
  x_error_m: 0.142855
  y_error_m: 0.304756
  z_error_m: 0.04532
  position_accuracy: 0.796941
  angle_accuracy: 0.377782
  position_accuracy_0.5m: 0.796941
  position_accuracy_1m: 0.972541
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.273086
  angle_accuracy_15deg: 0.377782
  angle_accuracy_30deg: 0.579838
repro:
  commit: 02024548fa0bb35e732b5d1fef92d77281a20b9a
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=4
    model.num_task_layers=4 data=chunked_multiview_sequence_bs8 data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data.seq_len_range=[64,256] loss=canonical_rot
    loss.position_weight=8.0 training.trainer.max_epochs=200 training.early_stopping.enabled=false
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i545-s4-h4-posw8
  predictions: knowledge/runs/run-i545-s4-h4-posw8/pred_test.npz
  log: .training_queue/logs/1782305635188217814_634494_i545_s4_h4_posw8.log
  curves: knowledge/runs/run-i545-s4-h4-posw8/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_7
parents:
- run-i545-s4-h4
relations:
- to: run-i545-s4-h4
  rel: compares
- to: run-i545-s4-h4-posw4
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
- position-weight
- collapse
---

## 考察 / Findings

### 要約
S=4/H=4 balanced に `position_weight` を 2.0 → 8.0 へ 4 倍増。回転が **52.27°/median 22.10° に崩壊**（baseline 8.23°）し、位置も 0.368m と悪化。共有 trunk 構成で位置 8× は回転 supervision を飢えさせ学習を破壊する負例。

### アーキテクチャ詳細
[[run-i545-s4-h4]] と同一（H=4/S=4, 77.845M, 同データ/プロトコル）。差は `loss.position_weight` 2.0→8.0 のみ。

### メトリクスの解釈
回転 mean 52.27 / median 22.10、角@15=0.378 / @10=0.273 と大幅劣化。位置 mean 0.368 / median 0.314 で baseline 0.337 よりむしろ悪い。位置を強く効かせたのに位置も悪化＝最適化が破綻している（観測: test 値が崩壊レンジ）。

### アーキテクチャ⇄メトリクスの因果考察
共有 trunk H=4 を rot/pose が共用するため、position を 8× にすると共有勾配が位置に支配され rotation 表現が崩れる。回転が崩れると canonical/角度系の整合も失われ、結果として位置の幾何整合も劣化したと推測（仮説）。同じ 8× でも fully separate の [[run-i545-s6-h0-auxoff-posw8]] は崩壊しない（rot 専用 trunk が保護される）。→ **「重い position weight は共有 trunk と両立しない」**を示す。

### 既存実験との比較
- [[run-i545-s4-h4-posw4]]（posw4, 8.01°/0.258m）は健全。崩壊閾値は 4 < x ≤ 8 の間。
- [[run-i545-s5-h2-auxoff-posw8]]（H=2, 35.29°）も崩壊だが H=4 ほど重くない＝共有が浅いほど崩壊が軽い傾向。
- [[run-i545-s6-h0-auxoff-posw8]]（H=0, 8.46°/0.166m）は無傷＝崩壊は共有 trunk 依存。

### 次に有効な実験
- posw を 4 から段階的に上げ崩壊閾値（5,6）を特定。共有構成では posw≤4 を実用上限と仮置きし、heavy posw は fully separate でのみ用いる。
