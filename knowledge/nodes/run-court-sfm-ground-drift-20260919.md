---
id: run-court-sfm-ground-drift-20260919
type: run
title: B00のSfM観測区間による地面高さ不整合の診断
provider: codex
date: '2026-09-19'
status: done
session: 01a0b44d-2642-7040-8df3-0cad69ce7cd8
config:
  model: saved_b00_colmap_sparse_model
  loss: no_optimization_diagnostic_only
  data: four_chronological_quarters_shared_ground_cells
  device: cpu
  primary_cell_size_m: 0.5
  sensitivity_cell_size_m: 1.0
metrics:
  frames: 491
  sparse_points: 217407
  primary_shared_cells: 33
  primary_q3_minus_q1_median_height_m: 0.03186454484424611
  sensitivity_shared_cells: 101
  sensitivity_q3_minus_q1_median_height_m: 0.030186141668640296
artifacts:
  output_dir: paper/court_robustness/evidence/sfm_drift
  report: paper/court_robustness/report.pdf
  measurements: paper/court_robustness/evidence/sfm_drift/measurements.json
parents:
- run-court-supplied-photos-paper-20260918
relations: []
tags:
- court-detection
- sfm
- drift
- diagnostic
---

## 考察 / Findings

### 要約
B00の保存SfMトラックを撮影区間で分け、共通地面セルの高さに時間依存の偏りを観測した。診断手順・図・解釈の正本は論文第6節と `artifacts.measurements`。

### アーキテクチャ詳細
新規学習・再構成は行っていない。既存RANSAC点群と地面・コート座標を共有し、COLMAPの公開形式から観測区間を抽出した。対応点は全単射・座標・RGB一致で結び付けた。

### メトリクスの解釈
高さ差は全区間で支持される同じ地面セル同士の比較で、絶対ドリフト誤差ではない。格子幅を変えた感度確認も記録した。学習を伴わないため収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
SfMドリフトと整合する兆候という仮説であり、地面の凹凸・特徴点誤差・三角測量誤差との因果分離はできていない。カメラ軌跡の始終点差は実移動を含むため根拠に使わない。

### 既存実験との比較
親runのレンダリングと教師の投影一致を補う幾何品質の診断である。検出器の性能を再測定・更新した実験ではない。

### 次に有効な実験
独立な地面基準と再訪対応による検証、およびSfM段階への長距離構造制約の導入比較。
