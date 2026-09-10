---
id: run-b01-court-shapes-only
type: run
title: B01：形状追加だけでは位置分布が狭まり、描画検証を中断
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: failed
config:
  scene: B01
  shapes:
  - circle
  - ellipse
  - rectangle
  - superellipse
  spatial_coverage_cell_m: null
metrics:
  proposals: 2232
  groups: 46
  occupied_1m_xy_cells: 632
  baseline_occupied_1m_xy_cells: 747
  outside_sfm_hull_count: 0
  pre_render_accepted: 2025
repro:
  commit: 96b98a17ee067d9956cafcfb65448d527910349b
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python outputs/b01-court-shapes/full_render_check.py
artifacts:
  run_dir: knowledge/runs/run-b01-court-shapes-only
parents:
- run-b01-court-sfm-bounds-full
relations: []
tags:
- court
- synthetic-data
- camera-sampling
- spatial-coverage
---

## 考察 / Findings

### 要約
長方形とスーパー楕円を追加しただけの案では、B01の占有1m XY区画が747から632へ減少した。位置分布を広げる目的を満たさないため、開始済みのGPU検証をqueue経由でキャンセルした。`failed` は意図的中断を表し、レンダラーの異常終了や画質評価の結果ではない。

### アーキテクチャ詳細
rectangleのL1支持関数、superellipse（p=4）のL(4/3)支持関数でSfM凸包に全形状を収めた。サンプリングは既存の3D弧長方式。軌道選択は旧来の型付きパラメータ値の多様性だけを使用した。

### メトリクスの解釈
1m区画数は全候補カメラ中心のXY座標を同じcourt-000座標系でfloorして重複を除いた数。GPUの全視点検証は完了していないので、描画採用率や品質改善は主張しない。

### アーキテクチャ⇄メトリクスの因果考察
形状の種類を増やしても、選択されたカメラ位置が広がるとは限らない。元の選択器は形状パラメータの種類・回数を評価しており、実際の位置の重複を評価していなかった。

### 既存実験との比較
前回の円・楕円のみの2,288候補・747区画に対して、この案は2,232候補・632区画。外周外0は維持した。

### 次に有効な実験
共通XYグリッドで未使用区画を優先しつつ、形状の採用回数を均等化する。空間カバーだけを優先する案では長方形18・スーパー楕円7・円1・楕円1に偏ったため、形状バランスを空間カバーより先に評価する。
