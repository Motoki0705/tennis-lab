---
id: run-issue880-b03-last
type: run
title: B03 Binary/Semanticアライメント目視比較
issue: 880
provider: codex
session: 01a09588-636e-7560-b584-5451ff831df9
date: '2026-09-13'
status: done
config:
  model: Issue876 completed last.ckpt, binary + 12-class semantic head
  loss: six-type raster support, equal type weights
  data: B03
  translation_radius_metres: 2.0
  maximum_iterations: 2000
metrics:
  binary_mean_placement_change_m: 0.05141524314277764
  binary_holdout_support_before: 0.44623084863026935
  binary_holdout_support_after: 0.48095494757095975
  semantic_mean_placement_change_m: 0.03843684965848107
  semantic_holdout_support_before: 0.17720882905026278
  semantic_holdout_support_after: 0.20331555884331465
repro:
  commit: 6679ae4307b1d7ddd1c81d753a0effe7e02ffacd
  branch: feat/issue-880-semantic-alignment
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-880-semantic-alignment
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.synthetic_data_generation.scripts.compare_semantic_alignment
    profile=b03 roots.project_root=/home/kamimura/projects/tennis-lab comparison.checkpoint=court_detection/mixed-source/semantic-line-frozen-b00-b03-b8-e20-s42/logs/version_0/checkpoints/last.ckpt
    comparison.output=issue880/b03-last-v1 comparison.maximum_iterations=2000
artifacts:
  run_dir: knowledge/runs/run-issue880-b03-last
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue880/b03-last-v1
  comparison: assets/alignment/issue880/B03/comparison.json
  binary_projection: assets/alignment/issue880/B03/binary-projection.png
  semantic_projection: assets/alignment/issue880/B03/semantic-projection.png
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
両方式の配置はほぼ一致し、Semanticを既定にするだけの目視上の改善差はない。

### アーキテクチャ詳細
学習終了後の同一checkpointからBinaryとSemanticを推論。SemanticはFar/Near・左右を確率加算し背景＋6種類へ統合する。既存の地面・尺度・コート数を固定し、fitカメラだけで位置・yawを最適化した。重み付き地面投影、探索条件は両方式で共通。

### メトリクスの解釈
frontmatterのplacement_changeは初期配置からの移動量で、正解に対する誤差ではない。supportは各方式内のbefore/afterのみ解釈し、異なる目的関数の数値を直接比較しない。これは推論比較であり学習曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
暗い画像と審判台の遮蔽があるholdoutも確認。Semanticの内部サービス線・センター線の投影は弱い。

### 既存実験との比較
既存alignmentと同一checkpointのBinaryを並列表示して目視した。モデル学習時のデータにB00〜B03が含まれるため、未知シーンへの汎化を示す評価ではない。分離したholdoutはアライメント最適化からの分離であり、学習データからの分離ではない。

### 次に有効な実験
ライン種別の誤認と、特にサービスライン・センター線・センターマークの再現率を改善してから同条件で再比較する。今回はSemanticのproduction既定化を行わない。
