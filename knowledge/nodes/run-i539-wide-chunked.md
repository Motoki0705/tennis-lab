---
id: run-i539-wide-chunked
type: run
title: i539_wide_chunked
issue: 539
provider: claude
session: 8722d9dc-5894-4536-8e54-d03e3e34949a
date: '2026-06-21'
status: done
config:
  model: multiview_axial_split_wide
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.205996
  position_error_std_m: 0.196673
  position_error_median_m: 0.139112
  angular_error_deg: 10.330529
  angular_error_std_deg: 11.399992
  angular_error_median_deg: 7.61869
  x_error_m: 0.080536
  y_error_m: 0.167964
  z_error_m: 0.039252
  position_accuracy: 0.921283
  angle_accuracy: 0.786782
  position_accuracy_0.5m: 0.921283
  position_accuracy_1m: 0.992914
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.634695
  angle_accuracy_15deg: 0.786782
  angle_accuracy_30deg: 0.957678
repro:
  commit: d407e54cdb903d7082aa4011b2a6f8cb0426c7cc
  branch: exp/i525-asym
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split_wide data.batch_size=2
    training.trainer.accumulate_grad_batches=4 data=chunked_multiview_sequence_bs8
    data.seq_len_range=[64,256] loss=canonical_rot training.trainer.max_epochs=200
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i539-wide-chunked
  predictions: knowledge/runs/run-i539-wide-chunked/pred_test.npz
  log: .training_queue/logs/1782037138499188850_185302_i539_wide_chunked.log
  curves: knowledge/runs/run-i539-wide-chunked/curves.png
parents:
- run-i535-asym-wide-rerun
- run-i518-exp10
relations:
- to: run-i535-asym-wide-rerun
  rel: supersedes
- to: run-i539-deep16-chunked
  rel: compares
- to: run-i539-ex10-chunked
  rel: compares
- to: run-i518-exp10
  rel: compares
tags:
- plcs
- canonical
- split-trunk
- asymmetric
- width
- chunked
- data-rich
- capacity-frontier
- best
---

## 考察 / Findings

### 要約
**本群最良。** wide（hidden768, rot10/pose6, 228.7M）を chunked + 勾配累積（effective batch=8）で学習し、回転 10.33°/**位置 0.206m（全ラン最良）**。固定データの崩壊（60.56°）から回復し、@Motoki0705 の「データをスケールすれば wide が効く」仮説を支持。

### アーキテクチャ詳細
`multiview_axial_split_wide` + `canonical_rot` + `data=chunked_multiview_sequence_bs8`。物理 `data.batch_size=2` × `accumulate_grad_batches=4` ＝ **effective batch 8**（本 PR で追加した勾配累積機能を使用）。`data.seq_len_range=[64,256]`、val/test は固定 `scene_dir`。`exp/i525-asym` worktree（commit `6d24b4d`）。

### メトリクスの解釈
test 回転 `10.33°`（median `7.62°`）/ 位置 `0.206m`（median `0.139m`）。curves.png: **full 200ep 完走**（early-stop なし）、train ang `5.54°`（本群最良）/ val `10.60°`、val は train に追随し崩壊・過学習なく最後まで右肩下がり。位置は固定 EX10（0.238m）・固定 deep16（0.252m）を上回り全ラン最良。

### アーキテクチャ⇄メトリクスの因果考察
data-rich（記憶不能な新規データの継続供給）+ effective batch 8 による勾配安定で、228.7M の大容量が初めて活き最後まで収束し続けたと解釈（仮説）。固定データでの崩壊（[[run-i535-asym-wide-rerun]] 60.56°）が容量過剰ではなく**データ枯渇 + batch2 の未収束**由来だったことが裏付けられる。

### 既存実験との比較
固定データ [[run-i535-asym-wide-rerun]]（60.56°, 未収束）を `supersedes`。同一 chunked 条件で [[run-i539-ex10-chunked]]（78.1M, 15.84°）・[[run-i539-deep16-chunked]]（142.3M, 19.11°）を大きく上回り **data-rich では容量が効き 幅 ≫ 深さ**。[[run-i518-exp10]]（固定 EX10）とも位置で上回る。留意: ex10/deep16 は ep86–95 で early-stop しており本比較は wide 有利方向の交絡を含むが、wide が full 学習で位置最良に達した事実は頑健。

### 次に有効な実験
ex10/deep16 を early-stop 緩和で full 学習させ公平比較を確定（#539 Phase2）。容量/データ量スケーリング曲線で wide のさらなる上積み余地と頭打ち点を特定する。
