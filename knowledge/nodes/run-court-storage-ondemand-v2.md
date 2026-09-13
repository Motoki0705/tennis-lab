---
id: run-court-storage-ondemand-v2
type: run
title: 常駐描画一致確認・collate失敗
provider: codex
session: 01a0985c-eb87-7310-9657-5411e2818d4e
date: '2026-09-13'
status: failed
config:
  model: DINOv3 ViT-B/16 + transformer + DPT + LoRA
  loss: kp_seg_line default
  data: B00, batch8, pose_safe, bf16, compile=false
metrics:
  rerender_float_mae: 0.0
repro:
  commit: 21199b3e29746c9d6360f287462f1ca400865690
  branch: codex/court-storage-ondemand
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/benchmark_court_ondemand.py
    --repo /home/kamimura/projects/tennis-lab --nht-worker /home/kamimura/projects/tennis-lab/.claude/worktrees/nht-court-storage-ondemand/scripts/resident_render_worker.py
    --nht-python /home/kamimura/.local/share/uv/tools/nht/bin/python --output outputs/storage-ondemand/benchmark-v2
artifacts:
  run_dir: knowledge/runs/run-court-storage-ondemand-v2
  metrics: knowledge/runs/run-court-storage-ondemand-v2/metrics.json
  nht_patch: knowledge/runs/run-court-storage-views-v1/nht-resident.patch
parents:
- run-court-storage-ondemand-v1
relations: []
tags:
- court
- storage
- ondemand
- throughput
---

## 考察 / Findings

### 要約
常駐レンダラーと保存RGBの完全一致を確認。collateの必須bundle引数欠落で学習ベンチマークは未完了。

### アーキテクチャ詳細
B00、959×539、8枚。NHTの公開ファイル境界の内側でcheckpointとshaderを1回だけロードする実験worker。

### メトリクスの解釈
RGB float MAE=0。render_batchesは常駐化後の描画時間であり、学習速度ではない。

### アーキテクチャ⇄メトリクスの因果考察
CLI起動と毎画像の重みロードを除去できる。

### 既存実験との比較
v1のimportを修正。学習速度はv3以降で評価。

### 次に有効な実験
collate引数を修正し実モデルforward/backwardへ接続する。
