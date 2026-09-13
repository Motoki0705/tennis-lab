---
id: run-court-storage-ondemand-v1
type: run
title: オンデマンド計測ハーネス初期化失敗
provider: codex
session: 01a0985c-eb87-7310-9657-5411e2818d4e
date: '2026-09-13'
status: failed
config:
  model: DINOv3 ViT-B/16 + transformer + DPT + LoRA
  loss: kp_seg_line default
  data: B00, batch8, pose_safe, bf16, compile=false
metrics: {}
repro:
  commit: 21199b3e29746c9d6360f287462f1ca400865690
  branch: codex/court-storage-ondemand
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/benchmark_court_ondemand.py
    --repo /home/kamimura/projects/tennis-lab --nht-worker /home/kamimura/projects/tennis-lab/.claude/worktrees/nht-court-storage-ondemand/scripts/resident_render_worker.py
    --nht-python /home/kamimura/.local/share/uv/tools/nht/bin/python --output outputs/storage-ondemand/benchmark-v1
artifacts:
  run_dir: knowledge/runs/run-court-storage-ondemand-v1
  nht_patch: knowledge/runs/run-court-storage-views-v1/nht-resident.patch
parents: []
relations: []
tags:
- court
- storage
- ondemand
- throughput
---

## 考察 / Findings

### 要約
実験起動時のimport errorで終了。速度や精度の測定値はない。

### アーキテクチャ詳細
登録関数の名称を誤って参照した。学習は開始していない。

### メトリクスの解釈
測定未実施。

### アーキテクチャ⇄メトリクスの因果考察
3DGS方式の問題ではなく、実験ハーネスの初期化ミス。

### 既存実験との比較
比較不可。

### 次に有効な実験
importを修正して再実行。
