---
id: run-court-storage-views-v1
type: run
title: 4シーンの新規カメラ・低解像度オンデマンド描画
provider: codex
session: 01a0985c-eb87-7310-9657-5411e2818d4e
date: '2026-09-13'
status: done
config:
  model: resident NHT
  data: B00-B03, 8 stratified training views each, 5cm local-X offset, original/256px
  loss: not a training run
metrics:
  B00_original_float_mae: 0.0
  B01_original_float_mae: 0.0
  B02_original_float_mae: 0.0
  B03_original_float_mae: 0.0
repro:
  commit: 21199b3e29746c9d6360f287462f1ca400865690
  branch: codex/court-storage-ondemand
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/probe_court_views.py
    --repo /home/kamimura/projects/tennis-lab --nht-worker /home/kamimura/projects/tennis-lab/.claude/worktrees/nht-court-storage-ondemand/scripts/resident_render_worker.py
    --nht-python /home/kamimura/.local/share/uv/tools/nht/bin/python --output outputs/storage-ondemand/views-v1
artifacts:
  run_dir: knowledge/runs/run-court-storage-views-v1
  metrics: knowledge/runs/run-court-storage-views-v1/metrics.json
  nht_patch: knowledge/runs/run-court-storage-views-v1/nht-resident.patch
parents:
- run-court-storage-ondemand-v4
relations: []
tags:
- court
- ondemand
- rendering
---

## 考察 / Findings

### 要約
4シーンすべてで既存カメラのRGBが完全一致し、新規カメラからも有限画像と可視KPが得られた。

### アーキテクチャ詳細
各sceneのtrain sampleを等間隔に8件選択。カメラlocal-Xへ5 cm平行移動した新規提案と、その幅256px版を描画。各group4回、初回を除く3回でwarm性能を見る。既存のV3投影・renderer visibility関数を使用する。

### メトリクスの解釈
新規カメラは元画像と異なり、全32提案で可視KPが正数だった。小解像度は描画時間が短い。解像度間の画素一致・dense target品質・学習精度は測定していない。

### アーキテクチャ⇄メトリクスの因果考察
画像容量はtemporary bufferに制限され、camera要求とlabel geometryのみ増える。ただし5cm提案はカメラ多様性の動作確認であり、大きな分布拡張や未観測領域の品質を示さない。

### 既存実験との比較
v4は固定の64カメラで速度を比較した。本runでは各sceneの別カメラと新規提案を検証。

### 次に有効な実験
productionのSfM境界・target-court binding・release gateを含む新規サンプラへ接続し、SEG/LINE生成、独立testを含む学習評価を行う。
