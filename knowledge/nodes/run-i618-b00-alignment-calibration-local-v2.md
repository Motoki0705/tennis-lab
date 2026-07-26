---
id: run-i618-b00-alignment-calibration-local-v2
type: run
title: B00 alignment physical-cluster local calibration
issue: 618
provider: codex
date: '2026-07-25'
status: done
config:
  model: ckpt/court_detection/line/court-detection-epoch19.ckpt
  loss: metric ITF line-template residual
  data: B00 fit groups only
  stability: frozen court-0 cluster local refit on three 8-group subsets
metrics:
  fit_accepted_view_fraction: 0.9944903581267218
  weighted_inlier_fraction: 0.9651286913818714
  distance_weighted_q95_m: 0.2097498846601784
  template_coverage_fraction: 1.0
  maximum_subset_centre_shift_m: 0.05307551869615782
  maximum_subset_orientation_difference_deg: 0.11844190269708861
  maximum_subset_relative_scale_difference: 0.004453370398004797
  point_support_count: 15099
  point_support_rms_m: 0.04758324569185451
artifacts:
  calibration: data/tennis/3dgs_alignment/b00-default-v1/calibration/b00-court-alignment-calibration-v2-c953326ffb0825fb.json
  report: .codex-loop/C05_ALIGNMENT_ACCEPTANCE.md
parents:
- run-i618-b00-alignment-calibration-global-v1
relations: []
tags:
- 3dgs
- alignment
- calibration
- stability
- fit-only
---

## 考察 / Findings

### 要約

選択済み `court-0` cluster を固定したlocal subset refitでは全fit calibration gateを
passし、holdout gateをfingerprint `c953…` として凍結できた。

### アーキテクチャ詳細

前runとの差分はstability測定だけで、各subsetの探索をreference centre ±2 m、
orientation ±5°、scale ±3%へ限定した。最終transform、line threshold 0.25 m、
acceptance gate値は変更していない。

### メトリクスの解釈

最大centre drift 0.0531 m、orientation 0.118°、scale 0.445%で、0.5 m / 1° / 2%
gateを十分下回った。point cloudは15,099 support points、RMS 0.0476 m、1 m grid
coverage 1.0だった。

### アーキテクチャ⇄メトリクスの因果考察

physical court instanceを固定したことで、partial evidenceが別court/直交構造を選ぶ
自由度だけを除き、同じcluster内でのgeometry変動を測定できた。

### 既存実験との比較

親 `run-i618-b00-alignment-calibration-global-v1` の9.08 m / 89.94° driftは、
同じevidenceとgateのまま0.053 m / 0.118°へ改善した。これはtransformの変更ではなく
stability estimatorの修正である。

### 次に有効な実験

fingerprint済みtransform/gateを変更せず、隔離済みgroups `{2,6,10,14}` を一度だけ
推論してacceptanceを判定する。
