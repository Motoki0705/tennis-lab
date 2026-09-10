---
id: run-court-b03-wide-cancelled
type: run
title: B03：広画角設定のcoverage検証
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: failed
config:
  scene: B03
  hfov_degrees:
  - 75.0
  - 110.0
  axis_ratios:
  - 0.65
  - 0.8
  - 1.0
  sfm_boundary_expansion_percent: 5.0
metrics:
  dataset_published: false
repro:
  commit: ad5554ab0cfdc433408b5762f3eb8792a30e6698
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python outputs/court-b00-b03-generation/run_requested.py
    outputs/court-b00-b03-generation/B03-wide-requested.yaml
artifacts:
  run_dir: knowledge/runs/run-court-b03-wide-cancelled
  log: knowledge/runs/run-court-b03-wide-cancelled/run.log
parents:
- run-court-b03-alignment-v14
relations: []
tags:
- court
- synthetic-data
- coverage
---

## 考察 / Findings

### 要約
B02のfull coverage不足を受け、実行途中で共有queueから意図的に取り消した。その後のCPU検証ではこのB03設定にはfull候補が5枚存在したため、B02と同じ失敗が確定していたわけではない。軸比を追加した設定ではfull候補53枚となり、より余裕のある設定で再実行する。

### アーキテクチャ詳細
SfM convex hullの5%拡大と0.5m内側marginを固定し、circle/ellipse/rectangle/superellipseを生成。alignment v14を再利用したCourt-only suffix。

### メトリクスの解釈
CPU投影のcoverageは画面内幾何形状の分類で、描画後のalpha/depth可視性とは別。

### アーキテクチャ⇄メトリクスの因果考察
B02のSfM範囲はコート中心に対し非対称。コート中心の閉軌道だけでは狭い側の境界に制限され、観測済みの広い側を十分利用できない。B03には同じ制約はなかった。

### 既存実験との比較
狭画角に対して有効候補数は改善するが、全体像の存在を別に検証する必要がある。

### 次に有効な実験
全release条件の幾何的実現可能性を描画前に確認する。B02は複合軌道中心をSfM hull頂点平均へ移し、個別軌道と注視点はコート中心に保持する。
