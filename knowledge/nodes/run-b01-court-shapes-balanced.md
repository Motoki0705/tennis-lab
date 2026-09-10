---
id: run-b01-court-shapes-balanced
type: run
title: B01：長方形・スーパー楕円と空間カバー選択の全視点検証
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: done
config:
  scene: B01
  shapes:
  - circle
  - ellipse
  - rectangle
  - superellipse
  boundary_margin_m: 0.5
  spatial_coverage_cell_m: 1.0
  hfov_degrees:
  - 45
  - 90
  look_at_jitter_radius_m: 1.0
metrics:
  proposals: 2264
  rendered_frames: 2176
  accepted_frames: 2176
  accepted_fraction: 0.9611307420494699
  trajectory_groups: 29
  occupied_1m_xy_cells: 1124
  baseline_occupied_1m_xy_cells: 747
  train_occupied_1m_xy_cells: 1042
  baseline_train_occupied_1m_xy_cells: 685
  outside_sfm_hull_count: 0
  minimum_boundary_clearance_m: 0.6938329607856488
repro:
  commit: 96b98a17ee067d9956cafcfb65448d527910349b
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python outputs/b01-court-shapes-balanced/full_render_check.py
artifacts:
  run_dir: knowledge/runs/run-b01-court-shapes-balanced
  output_dir: /home/kamimura/projects/tennis-lab-worktrees/court-sfm-bounded/outputs/b01-court-shapes-balanced
parents:
- run-b01-court-shapes-only
relations:
- to: run-b01-court-sfm-bounds-full
  rel: compares
tags:
- court
- synthetic-data
- camera-sampling
- spatial-coverage
---

## 考察 / Findings

### 要約
円・楕円に長方形とスーパー楕円を加え、形状バランスと未使用XY区画を評価する選択器を使用した。B01の占有1m区画は747から1,124へ50.5%増加し、SfM外周外0を維持。全2,176視点をGPU描画し、2,176視点が採用条件を通過した（全2,264候補中96.11%）。

### アーキテクチャ詳細
平面形状と支持関数をshapes.pyに集約。rectangleはL-infinity境界/L1支持関数、superellipseはp=4境界/L(4/3)支持関数で全周をSfM凸包内へ制約する。角も離散サンプルに依存せず包含される。共通座標系のXYグリッドを使用し、型付き値の新規性、形状回数のバランス、未使用区画数を順に評価する。円7・楕円7・長方形8・スーパー楕円7を選択。半径倍率、軸比、高さ、注視点の乱数設定は前回から維持した。
候補増加に伴う計算量に対応するため、残り軌道グループの最低コストを候補ごとに再ソートせず、反復ごとに一度計算するようにした。予約するフレーム数の意味は変えていない。

### メトリクスの解釈
占有区画数は全候補のカメラ中心をcourt-000座標系の1m XYグリッドへ写し重複除去した数であり、画像の可視面積や下流の推定精度ではない。比較元は2,288候補、今回2,264候補なので枚数増による見かけの改善ではない。描画前はコート点不足54・near/far不定34を除外。描画後の追加除外はなかった。全29軌道、full/near_full/partial、14KP全クラスを確認した。

### アーキテクチャ⇄メトリクスの因果考察
形状だけ増やす案は区画数632へ減少した。実際の位置の重複を減らす選択が必要だった。空間カバーのみを優先した案は長方形へ偏ったため、形状回数のバランスを先に評価し、円・楕円のバリエーションも維持した。

### 既存実験との比較
比較先run-b01-court-sfm-bounds-fullは円20・楕円27、747区画、採用2,101/2,288。今回の形状群は少ない候補数で位置分布を広げ、採用率も上がった。レンダリング条件は同じpublic NHT export、GPU0、HFOV45〜90度。

### 次に有効な実験
下流の実動画評価で、位置分布の広がりがpose推定へ与える効果を測る。SfM凸包は水平方向の位置制約であり、未観測方向や高さに対する画質保証ではない。ネットや遠景の残存artifactは別途評価する。

B01の手動確認済みgeometryを座標検証用adapterで明示的に解釈した。通常のcanonical alignment/dataset ownerの公開を行った結果ではない。GPU検証は共有training queueを使用した。
