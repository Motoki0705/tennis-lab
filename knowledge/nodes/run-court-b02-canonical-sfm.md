---
id: run-court-b02-canonical-sfm
type: run
title: B02：現行alignmentでSfM制約付きCourt dataset正式公開
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: done
config:
  scene: B02
  hfov_degrees:
  - 75.0
  - 110.0
  axis_ratios:
  - 0.35
  - 0.5
  - 0.65
  - 0.8
  - 1.0
  sfm_complex_center_on_hull: true
  sfm_boundary_expansion_percent: 5.0
  sfm_boundary_margin_m: 0.5
metrics:
  proposal_count: 2232
  accepted_frame_count: 2112
  rejected_frame_count: 120
  accepted_fraction: 0.946236559139785
  trajectory_group_count: 40
  maximum_adjacent_step_m: 1.0495597706104847
  split_frame_counts:
    test: 241
    train: 1681
    validation: 190
  split_group_counts:
    test: 4
    train: 32
    validation: 4
  coverage_counts:
    full: 134
    near_full: 477
    partial: 1501
  renderer_visible_points_by_class:
    far_doubles_left: 1692
    far_doubles_right: 1715
    near_doubles_left: 218
    near_doubles_right: 217
    far_singles_left: 1750
    near_singles_left: 225
    far_singles_right: 1780
    near_singles_right: 220
    far_service_left: 1868
    far_service_right: 1880
    near_service_left: 953
    near_service_right: 983
    far_service_t: 2093
    near_service_t: 1179
  split_leakage_count: 0
  court_sample_counts:
    court-000: 2112
  split_court_sample_counts:
    test:
      court-000: 241
    train:
      court-000: 1681
    validation:
      court-000: 190
  minimum_expanded_hull_clearance_m: 0.9689974556616683
  accepted_court_count: 1
repro:
  commit: 5589f44671ecd85578cc93f62db131da54ec9bfe
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python outputs/court-b00-b03-generation/run_requested.py
    outputs/court-b00-b03-generation/B02-final-requested.yaml
artifacts:
  run_dir: knowledge/runs/run-court-b02-canonical-sfm
  output_dir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B02/datasets/court
  preview: knowledge/runs/run-court-b02-canonical-sfm/B02-preview.jpg
  log: knowledge/runs/run-court-b02-canonical-sfm/run.log
parents:
- run-court-b02-wide-no-full
relations: []
tags:
- court
- synthetic-data
- canonical
- sfm-bounds
---

## 考察 / Findings

### 要約
B02を2112枚で正式公開し、alignmentとCourt ownerの再読込検証が成功。全採用カメラ位置をSfM hullへ照合し、5%拡大した外周から0.5m以上内側にあることを確認した。

### アーキテクチャ詳細
現行production alignmentのground-line-map.npzはsemantic_ground_line_correspondences_v14。SfM/3DGSは再学習せず再利用した。circle/ellipse/rectangle/superellipseを混合し、個別コートと複合中心の軌道、高さ、軸比、注視点jitterを変化させた。シーン別設定は同梱YAMLに固定。

### メトリクスの解釈
2232候補のうち2112枚を採用。release条件の最低2,000枚・採用率90%・coverage条件は変更していない。minimum_expanded_hull_clearance_mは全採用カメラの水平位置に対する最小境界距離。形状全体の包含は解析的support計算と単体テストでも検証する。

### アーキテクチャ⇄メトリクスの因果考察
画角を75〜110度へ広げ、SfM境界を広げず画像内のコート点を増やした。B02ではSfM範囲の非対称性に合わせて複合軌道中心をhull頂点平均へ移した。幾何的適合性はアーティファクトの完全排除を保証しない。

### 既存実験との比較
親runで得た失敗理由に対処したCourt-only再生成。各runの設定・結果・失敗理由は親nodeを参照。

### 次に有効な実験
生成データを用いた下流court detection/pose推定の精度と実画像への汎化を評価する。今回の生成検証では下流モデルの精度は測定していない。
