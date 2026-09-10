---
id: run-b01-court-sfm-bounds-gpu-mismatch
type: run
title: B01描画検証：混在GPUのtinycudann互換性エラー
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: failed
config:
  scene: B01
  CUDA_VISIBLE_DEVICES: unset
metrics: {}
repro:
  commit: 96b98a17ee067d9956cafcfb65448d527910349b
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python outputs/b01-court-bounds/render_compare.py
artifacts:
  run_dir: knowledge/runs/run-b01-court-sfm-bounds-gpu-mismatch
parents: []
relations: []
tags:
- court
- synthetic-data
- sfm
- camera-sampling
---

## 考察 / Findings

### 要約
描画を開始できず失敗。複数GPUのcompute capabilityを検出し、tinycudannが非対応の75を選択した。

### メトリクスの解釈
描画品質に関する結果はない。

### 次に有効な実験
キューの資源予約を維持したまま、既存の本番設定と同じ `CUDA_VISIBLE_DEVICES=0` を指定して再実行する。
