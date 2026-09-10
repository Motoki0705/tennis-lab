---
id: run-court-b03-canonical-sfm
type: run
title: B03：現行alignmentでSfM制約付きCourt dataset正式公開
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: done
config:
  scene: B03
  hfov_degrees:
  - 75.0
  - 110.0
  axis_ratios:
  - 0.35
  - 0.5
  - 0.65
  - 0.8
  - 1.0
  sfm_complex_center_on_hull: false
  sfm_boundary_expansion_percent: 5.0
  sfm_boundary_margin_m: 0.5
metrics:
  proposal_count: 2256
  accepted_frame_count: 2093
  rejected_frame_count: 163
  accepted_fraction: 0.9277482269503546
  trajectory_group_count: 40
  maximum_adjacent_step_m: 1.035211661119462
  split_frame_counts:
    test: 198
    train: 1669
    validation: 226
  split_group_counts:
    test: 4
    train: 32
    validation: 4
  coverage_counts:
    full: 53
    near_full: 598
    partial: 1442
  renderer_visible_points_by_class:
    far_doubles_left: 1694
    far_doubles_right: 1695
    near_doubles_left: 124
    near_doubles_right: 121
    far_singles_left: 1754
    near_singles_left: 120
    far_singles_right: 1748
    near_singles_right: 120
    far_service_left: 1845
    far_service_right: 1846
    near_service_left: 966
    near_service_right: 958
    far_service_t: 2089
    near_service_t: 1081
  split_leakage_count: 0
  court_sample_counts:
    court-000: 2093
  split_court_sample_counts:
    test:
      court-000: 198
    train:
      court-000: 1669
    validation:
      court-000: 226
  minimum_expanded_hull_clearance_m: 0.9626028587621107
  accepted_court_count: 1
repro:
  commit: 5589f44671ecd85578cc93f62db131da54ec9bfe
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python outputs/court-b00-b03-generation/run_requested.py
    outputs/court-b00-b03-generation/B03-final-requested.yaml
artifacts:
  run_dir: knowledge/runs/run-court-b03-canonical-sfm
  output_dir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B03/datasets/court
  preview: knowledge/runs/run-court-b03-canonical-sfm/B03-preview.jpg
  log: knowledge/runs/run-court-b03-canonical-sfm/run.log
parents:
- run-court-b03-wide-cancelled
relations: []
tags:
- court
- synthetic-data
- canonical
- sfm-bounds
---

## 考察 / Findings

### 要約
B03を2093枚で正式公開し、alignmentとCourt ownerの再読込検証が成功。全採用カメラ位置をSfM hullへ照合し、5%拡大した外周から0.5m以上内側にあることを確認した。

### アーキテクチャ詳細
現行production alignmentのground-line-map.npzはsemantic_ground_line_correspondences_v14。SfM/3DGSは再学習せず再利用した。circle/ellipse/rectangle/superellipseを混合し、個別コートと複合中心の軌道、高さ、軸比、注視点jitterを変化させた。シーン別設定は同梱YAMLに固定。

### メトリクスの解釈
2256候補のうち2093枚を採用。release条件の最低2,000枚・採用率90%・coverage条件は変更していない。minimum_expanded_hull_clearance_mは全採用カメラの水平位置に対する最小境界距離。形状全体の包含は解析的support計算と単体テストでも検証する。

### アーキテクチャ⇄メトリクスの因果考察
画角を75〜110度へ広げ、SfM境界を広げず画像内のコート点を増やした。B02ではSfM範囲の非対称性に合わせて複合軌道中心をhull頂点平均へ移した。幾何的適合性はアーティファクトの完全排除を保証しない。

### 既存実験との比較
親runで得た失敗理由に対処したCourt-only再生成。各runの設定・結果・失敗理由は親nodeを参照。

### 次に有効な実験
生成データを用いた下流court detection/pose推定の精度と実画像への汎化を評価する。今回の生成検証では下流モデルの精度は測定していない。
