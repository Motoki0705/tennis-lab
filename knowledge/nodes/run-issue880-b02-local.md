---
id: run-issue880-b02-local
type: run
title: B02 Binary/Semanticアライメント目視比較
issue: 880
provider: codex
session: 01a09588-636e-7560-b584-5451ff831df9
date: '2026-09-13'
status: done
config:
  model: Issue876 completed last.ckpt, binary + 12-class semantic head
  loss: six-type raster support, equal type weights
  data: B02
  translation_radius_metres: 2.0
  maximum_iterations: 400
metrics:
  binary_mean_placement_change_m: 0.05227057390296896
  binary_holdout_support_before: 0.256976705044508
  binary_holdout_support_after: 0.27277902017037076
  semantic_mean_placement_change_m: 0.08660750416778375
  semantic_holdout_support_before: 0.08940378751140088
  semantic_holdout_support_after: 0.08515449590049684
repro:
  commit: 6679ae4307b1d7ddd1c81d753a0effe7e02ffacd
  branch: feat/issue-880-semantic-alignment
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-880-semantic-alignment
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.synthetic_data_generation.scripts.compare_semantic_alignment
    profile=b02 roots.project_root=/home/kamimura/projects/tennis-lab comparison.checkpoint=court_detection/mixed-source/semantic-line-frozen-b00-b03-b8-e20-s42/logs/version_0/checkpoints/last.ckpt
    comparison.output=issue880/b02-last-v1
artifacts:
  run_dir: knowledge/runs/run-issue880-b02-local
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue880/b02-last-v1
  comparison: assets/alignment/issue880/B02/local-search/comparison.json
  binary_projection: assets/alignment/issue880/B02/local-search/binary-projection.png
  semantic_projection: assets/alignment/issue880/B02/local-search/semantic-projection.png
parents: []
relations: []
tags:
- court
- alignment
- semantic-line
- visual-comparison
---

## 考察 / Findings

### 要約
既存配置に大きな縦ずれがあり、±2mの有界探索では両方式とも修正できない。

### アーキテクチャ詳細
学習終了後の同一checkpointからBinaryとSemanticを推論。SemanticはFar/Near・左右を確率加算し背景＋6種類へ統合する。既存の地面・尺度・コート数を固定し、fitカメラだけで位置・yawを最適化した。重み付き地面投影、探索条件は両方式で共通。

### メトリクスの解釈
frontmatterのplacement_changeは初期配置からの移動量で、正解に対する誤差ではない。supportは各方式内のbefore/afterのみ解釈し、異なる目的関数の数値を直接比較しない。これは推論比較であり学習曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
この結果だけでは手法の能力差と探索範囲の不足を分離できないため、両方式を同じ±8m条件で追加実験した。

### 既存実験との比較
既存alignmentと同一checkpointのBinaryを並列表示して目視した。モデル学習時のデータにB00〜B03が含まれるため、未知シーンへの汎化を示す評価ではない。分離したholdoutはアライメント最適化からの分離であり、学習データからの分離ではない。

### 次に有効な実験
ライン種別の誤認と、特にサービスライン・センター線・センターマークの再現率を改善してから同条件で再比較する。今回はSemanticのproduction既定化を行わない。
