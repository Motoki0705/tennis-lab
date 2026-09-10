---
id: run-b01-court-sfm-bounds
type: run
title: B01：SfM外周制約で描画改善、狭い画角では採用率不足
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: done
config:
  scene: B01
  dataset_selector: sfm_bounded
  hfov_degrees:
  - 35
  - 75
  boundary_margin_m: 0.5
  look_at_jitter_radius_m: 1.0
metrics:
  baseline_outside_count: 3769
  bounded_outside_count: 0
  bounded_proposals: 2288
  bounded_groups: 47
  bounded_pre_render_accepted: 2015
  bounded_pre_render_acceptance_fraction: 0.8806818181818182
  baseline_rendered_count: 96
  bounded_rendered_count: 188
  baseline_mean_alpha_below_0_5_fraction: 0.0038976677029708465
  bounded_mean_alpha_below_0_5_fraction: 0.001971137565459158
repro:
  commit: 96b98a17ee067d9956cafcfb65448d527910349b
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python outputs/b01-court-bounds/render_compare.py
artifacts:
  run_dir: knowledge/runs/run-b01-court-sfm-bounds
parents:
- run-b01-court-sfm-bounds-gpu-mismatch
relations: []
tags:
- court
- synthetic-data
- sfm
- camera-sampling
---

## 考察 / Findings

### 要約
全周の円・楕円をSfMカメラXY凸包の内側へ収めると、B01で外周外の候補が3,769/4,800から0/2,288になった。代表画像では大きな背景の引き伸ばしが減ったが、従来HFOVのままでは幾何採用率が88.1%で本番基準90%を満たさない。

### アーキテクチャ詳細
各コート/複合中心の局所平面で凸包の半平面を構成し、楕円の支持関数から全周を含む半径上限を求める。高さの3段階と上下変動は保持。注視点は中心半径1mの円盤から面積一様にサンプリングする。
B01は手動確認済みgeometry形式であり、この実験は座標のみを明示的に採用した。canonical alignment/dataset ownerへの変換・公開は行っていない。両計画では既存の全軌道shard割当が旧計画で600枚上限を満たさないため最大800枚を指定した。

### メトリクスの解釈
各軌道4視点を品質に依存せず選択し、旧96枚・新188枚を実GPUで描画。alpha欠損率は画像の引き伸ばしを直接測らない。画像目視と併せて解釈する。背景・ネットの残存artifactも確認した。

### アーキテクチャ⇄メトリクスの因果考察
仮説：外周外への外挿を抑えたことでRGBの大崩れが減った。一方、内側の小半径では狭い画角にコート点が入りにくくなり、semantic coverage不足が増えた。

### 既存実験との比較
同じB01のv3既定軌道と比較。生成件数・半径分布が変わるため対応画素の品質比較や下流精度改善を意味しない。

### 次に有効な実験
外周制約を維持してHFOVを45〜90度へ広げ、全ての幾何採用視点を描画して最終採用率を検証する。
