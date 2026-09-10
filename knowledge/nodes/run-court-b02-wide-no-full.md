---
id: run-court-b02-wide-no-full
type: run
title: B02：広画角設定のcoverage検証
provider: codex
session: 01a088da-0867-7e43-99b5-415e56dc292d
date: '2026-09-10'
status: failed
config:
  scene: B02
  hfov_degrees:
  - 75.0
  - 110.0
  axis_ratios:
  - 0.65
  - 0.8
  - 1.0
  sfm_boundary_expansion_percent: 5.0
metrics:
  pre_render_valid: 2128
  proposals: 2232
  full_coverage: 0
repro:
  commit: ad5554ab0cfdc433408b5762f3eb8792a30e6698
  branch: codex/court-sfm-bounded
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. .venv/bin/python outputs/court-b00-b03-generation/run_requested.py
    outputs/court-b00-b03-generation/B02-wide-requested.yaml
artifacts:
  run_dir: knowledge/runs/run-court-b02-wide-no-full
  log: knowledge/runs/run-court-b02-wide-no-full/run.log
parents:
- run-court-b02-narrow-fov
relations: []
tags:
- court
- synthetic-data
- coverage
---

## 考察 / Findings

### 要約
75〜110度で採用候補数は増えたが、描画後のrelease gateがfull coverage不足で正式公開を拒否した。全体が映る幾何候補自体が0だった。画角・採用枚数だけの事前検証では不十分だった。

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
