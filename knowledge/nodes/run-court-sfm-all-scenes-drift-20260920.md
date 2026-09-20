---
id: run-court-sfm-all-scenes-drift-20260920
type: run
title: B00〜B03の共通条件によるSfM地面高さ不整合の診断
provider: codex
date: '2026-09-20'
status: done
session: 01a0b44d-2642-7040-8df3-0cad69ce7cd8
config:
  model: saved_b00_b03_colmap_sparse_models
  loss: no_optimization_diagnostic_only
  data: chronological_registered_frame_halves_shared_ground_cells
  device: cpu
  primary_cell_size_m: 0.5
  sensitivity_cell_size_m: 1.0
  minimum_points_per_cell_per_period: 5
metrics:
  scenes: 4
  sparse_points: 428265
  b00_primary_shared_cells: 617
  b01_primary_shared_cells: 246
  b02_primary_shared_cells: 69
  b03_primary_shared_cells: 6
  b00_later_minus_earlier_median_height_m: 0.008820320916167522
  b01_later_minus_earlier_median_height_m: 0.0008863718554193487
  b02_later_minus_earlier_median_height_m: -0.013775496912700358
  b03_later_minus_earlier_median_height_m: 0.015520147622269187
artifacts:
  output_dir: paper/court_robustness/evidence/sfm_drift
  measurements: paper/court_robustness/evidence/sfm_drift/measurements.json
  report: paper/court_robustness/report.pdf
parents:
- run-court-sfm-ground-drift-20260919
relations: []
tags:
- court-detection
- sfm
- drift
- diagnostic
---

## 考察 / Findings

### 要約
全4シーンを同じ条件で診断した。シーン別の支持数・分布・感度と、既存ドリフト研究との関係は論文第2・6節を正本とする。

### アーキテクチャ詳細
新規学習・再構成は行っていない。登録画像の欠番を保持して撮影順に分割し、全シーンで保存コートの共通平面を高さ基準とした。全SfMトラックと公開点群の座標・RGB・全単射を照合した。

### メトリクスの解釈
主要値は前半と後半の共通セルにおける高さ中央値の差であり、絶対ドリフト誤差ではない。B01は偏りが小さく、B03は支持6セルで感度が大きい。学習を伴わないため収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
ドリフトと整合する不整合を調べる診断だが、地面の凹凸や特徴点・三角測量誤差との因果分離はできない。全シーンでドリフトを確証したとは解釈しない。

### 既存実験との比較
親runの4区間・0.5 m格子ではB01〜B03に共通セルがなく、全シーンを同一の2区間比較へ変更した。4区間診断も未成立をnullとして保持する。B00の比較対象点・セルも変わるため、旧3.2 cmとの差を精度改善と解釈しない。

### 次に有効な実験
再訪で十分な地面観測が重なる撮影と独立な地面基準を用意し、構造制約・ループ閉じ込みによる再構成の改善を比較する。
