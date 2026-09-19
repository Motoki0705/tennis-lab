---
id: run-slcs-full-real-rgb-no-ball-smooth-start-failure-v1
type: run
title: 'SLCS全体版60epoch開始: validation RGB archive CRCエラーで0更新停止'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: failed
config:
  model: SLCSFusionModel hidden128/shared4/DINO-downsample2
  loss: ball_position_smoothness_weight=0
  data: slcs/real_rgb_v1
  planned_epochs: 60
metrics:
  optimizer_updates: 0
  exit_code: 1
repro:
  commit: e454473c28e5a591dc892586e8575e1b29daa6cb
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tasks.slcs.scripts.train --config-name train_real_rgb loss.ball_position_smoothness_weight=0.0
    run.output_dir=slcs/train/real_rgb_no_ball_smooth/s42-takeover-001
artifacts:
  run_dir: knowledge/runs/run-slcs-full-real-rgb-no-ball-smooth-start-failure-v1
  log: knowledge/runs/run-slcs-full-real-rgb-no-ball-smooth-start-failure-v1/queue.log
  output_dir: outputs/slcs/train/real_rgb_no_ball_smooth/s42-takeover-001
parents: [run-slcs-real-rgb-cpu-smoke-v1, run-slcs-pilot-no-ball-smooth-eval-v1]
tags: [slcs, real-rgb, full-dataset, failed, crc]
---

## 考察 / Findings

### 要約

全61clip版の60epoch学習を共有queueへ投入したが、steps計数時のval dataset構築で停止した。
`BadZipFile: Bad CRC-32 for file 'tokens.npy'`が発生し、optimizer更新は0。

### アーキテクチャ詳細

`train_real_rgb`からball平滑化weightだけ0を指定。batch16、workers2、warmup200、seed42。
checkpointsやCRC検査を省略するfallbackはない。失敗時readerはNPZ全pathを例外に付けておらず、
最初のlogだけでは対象camera/clipの特定はできない。

### メトリクスの解釈

精度評価・学習曲線はなく、成功したCPU smokeとは異なる新しい読込時点での失敗。
queue log、開始commit、command、resolved configを保存し、後の成功で取り消さない。

### アーキテクチャ⇄メトリクスの因果考察

ZIPが返した読込検証エラーは観測事実だが、永続file破損か一時的読込異常かはこのlogでは未確定。
新しいlossやGPUモデル演算が原因とする根拠はない。媒体・hardwareの原因探索を本作業の条件にはしない。

### 既存実験との比較

同datasetの本番loader計数とCPU1batch smokeは先に成功している。今回の失敗はその記録を否定せず、
同時に過去成功だけで今回の失敗を無視しない。

### 次に有効な実験

全173 RGB NPZを一度だけread-only CRC点検し、readerへclip/camera/path付きerrorとcause chainを追加する。
壊れた配列を黙って使わず、点検結果を保存して新出力先で同じ学習条件を再実行する。
