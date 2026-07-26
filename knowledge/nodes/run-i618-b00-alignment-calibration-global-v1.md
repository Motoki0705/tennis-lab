---
id: run-i618-b00-alignment-calibration-global-v1
type: run
title: B00 alignment global subset calibration（失敗）
issue: 618
provider: codex
date: '2026-07-25'
status: failed
config:
  model: ckpt/court_detection/line/court-detection-epoch19.ckpt
  loss: metric ITF line-template residual
  data: B00 fit groups only
  stability: unrestricted multi-court rediscovery on three 8-group subsets
metrics:
  fit_accepted_view_fraction: 0.9944903581267218
  weighted_inlier_fraction: 0.9651286913818714
  distance_weighted_q95_m: 0.2097498846601784
  maximum_subset_centre_shift_m: 9.07534255536229
  maximum_subset_orientation_difference_deg: 89.93695527263324
artifacts:
  calibration: data/tennis/3dgs_alignment/b00-default-v1/calibration/b00-court-alignment-calibration-v1-078a71d385ece130.json
  report: .codex-loop/C05_ALIGNMENT_ACCEPTANCE.md
parents:
- run-i618-b00-ground-line-court-fit-v1
relations: []
tags:
- 3dgs
- alignment
- calibration
- fit-only
- negative
---

## 考察 / Findings

### 要約

fit residual、coverage、point support は pass したが、partial subset で global
multi-court search が約90°の偽候補を再選択し、stability gate を棄却した。holdout は
未推論のまま停止した。

### アーキテクチャ詳細

C04 と同じ line model、projection、metric templateを使い、3つの8-group subsetごとに
2面をglobal再探索して、full-fit `court-0` にcentreが近いcandidateを照合した。

### メトリクスの解釈

aggregate weighted inlierは0.9651、q95は0.2097 mだった一方、subset centre shiftは
最大9.08 m、orientation差は最大89.94°となった。これは frozen gate
0.5 m / 1°を明確に破る。

### アーキテクチャ⇄メトリクスの因果考察

観測上、partial evidenceではglobal探索の候補集合自体に対象courtの正しい局所解が
入らなかった。仮説として、これはfixed physical clusterのstabilityではなく
court再選択能力を測る設計になっていたためである。

### 既存実験との比較

親 `run-i618-b00-ground-line-court-fit-v1` は全fit evidenceで2面を分離できた。本runは
subset evidenceで同じglobal selectionを再要求すると不安定になる負の結果を追加した。

### 次に有効な実験

gate値は変えず、full-fitで選択済みのphysical cluster近傍だけをlocal refitしてsubset
stabilityを測り直す。
