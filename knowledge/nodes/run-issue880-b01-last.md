---
id: run-issue880-b01-last
type: run
title: B01 Binary/Semanticアライメント目視比較
issue: 880
provider: codex
session: 01a09588-636e-7560-b584-5451ff831df9
date: '2026-09-13'
status: done
config:
  model: Issue876 completed last.ckpt, binary + 12-class semantic head
  loss: six-type raster support, equal type weights
  data: B01
  translation_radius_metres: 2.0
  maximum_iterations: 2000
metrics:
  binary_mean_placement_change_m: 0.041931349988044146
  binary_holdout_support_before: 0.7508809516827265
  binary_holdout_support_after: 0.821399464375443
  semantic_mean_placement_change_m: 0.03936058192507185
  semantic_holdout_support_before: 0.31406880718552405
  semantic_holdout_support_after: 0.33992737976627213
repro:
  commit: cfc211b87ad3a1027082f5a9f9092956f09eee11
  branch: feat/issue-880-semantic-alignment
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-880-semantic-alignment
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.synthetic_data_generation.scripts.compare_semantic_alignment
    profile=b01 roots.project_root=/home/kamimura/projects/tennis-lab comparison.checkpoint=court_detection/mixed-source/semantic-line-frozen-b00-b03-b8-e20-s42/logs/version_0/checkpoints/last.ckpt
    comparison.output=issue880/b01-last-v1 comparison.maximum_iterations=2000
artifacts:
  run_dir: knowledge/runs/run-issue880-b01-last
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue880/b01-last-v1
  comparison: assets/alignment/issue880/B01/comparison.json
  binary_projection: assets/alignment/issue880/B01/binary-projection.png
  semantic_projection: assets/alignment/issue880/B01/semantic-projection.png
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
人手確定済み3面の配置から比較。双方とも微調整に留まり、Semantic優位とは判断できない。

### アーキテクチャ詳細
学習終了後の同一checkpointからBinaryとSemanticを推論。SemanticはFar/Near・左右を確率加算し背景＋6種類へ統合する。既存の地面・尺度・コート数を固定し、fitカメラだけで位置・yawを最適化した。重み付き地面投影、探索条件は両方式で共通。

### メトリクスの解釈
frontmatterのplacement_changeは初期配置からの移動量で、正解に対する誤差ではない。supportは各方式内のbefore/afterのみ解釈し、異なる目的関数の数値を直接比較しない。これは推論比較であり学習曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
人手確定時の尺度をplaneとheatmapの双方に反映し、同じmetric frameで比較した。遠い外側コートも含め大きな改善差はない。

### 既存実験との比較
既存alignmentと同一checkpointのBinaryを並列表示して目視した。モデル学習時のデータにB00〜B03が含まれるため、未知シーンへの汎化を示す評価ではない。分離したholdoutはアライメント最適化からの分離であり、学習データからの分離ではない。

### 次に有効な実験
ライン種別の誤認と、特にサービスライン・センター線・センターマークの再現率を改善してから同条件で再比較する。今回はSemanticのproduction既定化を行わない。
