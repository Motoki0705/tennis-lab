---
id: run-court-b00-alignment-v14
type: run
title: B00：現行alignment公開後の実行中コード更新による停止
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: failed
config:
  scene: B00
  hfov_degrees:
  - 75.0
  - 110.0
  sfm_boundary_expansion_percent: 5.0
metrics:
  alignment_published: true
  old_court_dataset_removed: true
  new_court_dataset_published: false
repro:
  commit: ad5554ab0cfdc433408b5762f3eb8792a30e6698
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python outputs/court-b00-b03-generation/run_requested.py
    outputs/court-b00-b03-generation/B00-wide-requested.yaml
artifacts:
  run_dir: knowledge/runs/run-court-b00-alignment-v14
  log: knowledge/runs/run-court-b00-alignment-v14/run.log
parents:
- run-court-b00-narrow-cancelled
relations: []
tags:
- court
- alignment
- runtime
---

## 考察 / Findings

### 要約
alignmentの推定・採用・公開は完了し、旧Court datasetは削除された。続くCourt生成は、実行開始後に追加した設定フィールドと旧メモリ内クラスの不一致によるAttributeErrorで停止した。

### アーキテクチャ詳細
production alignmentを使用。対応データはsemantic_ground_line_correspondences_v14。起動後にsfm_complex_center_on_hullを追加し、後からlazy importされたselection側だけが新しいコードになった。

### メトリクスの解釈
dataset失敗はalignment不採用ではない。新alignmentのCOMPLETEDを確認したため、Court-only suffixで再実行可能。

### アーキテクチャ⇄メトリクスの因果考察
長時間の実行中に同じworktreeのruntimeソースを変更した運用ミス。固定commitの単独再実行では同じ混在状態を再現しない。以後は生成終了までruntimeコードを固定する。

### 既存実験との比較
前のB00試行と異なり、今回でalignment更新と旧dataset削除まで完了した。

### 次に有効な実験
新alignmentで幾何的実現可能性を確認し、同一の固定コードからCourt-only生成を実行する。
