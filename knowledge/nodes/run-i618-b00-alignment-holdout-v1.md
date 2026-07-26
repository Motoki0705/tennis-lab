---
id: run-i618-b00-alignment-holdout-v1
type: run
title: B00 alignment one-shot holdout（棄却）
issue: 618
provider: codex
date: '2026-07-25'
status: failed
config:
  model: ckpt/court_detection/line/court-detection-epoch19.ckpt
  loss: frozen metric ITF line-template acceptance
  data: untouched B00 holdout groups 2,6,10,14
  calibration: b00-court-alignment-calibration-v2-c953326ffb0825fb
metrics:
  accepted_view_fraction: 0.9609375
  weighted_inlier_fraction: 0.8316577575874766
  distance_weighted_q95_m: 1.2398550966338835
  template_coverage_fraction: 0.8069815195071869
  minimum_group_template_coverage_fraction: 0.22347707049965776
  minimum_group_weighted_inlier_fraction: 0.6864175367475401
  camera_height_min_m: 1.4417053264493893
  camera_height_max_m: 2.8223590693411467
artifacts:
  validation: data/tennis/3dgs_alignment/b00-default-v1/holdout_validation/b00-court-alignment-holdout-v1-009eae8ad21dc827.json
  report: .codex-loop/C05_ALIGNMENT_ACCEPTANCE.md
parents:
- run-i618-b00-alignment-calibration-local-v2
relations: []
tags:
- 3dgs
- alignment
- holdout
- negative
---

## 考察 / Findings

### 要約

one-shot holdoutはaggregate inlier/coverageとcamera-height gateを通したが、distance
q95とevery-group coverageを破り、正式にrejectedとなった。scene contractは作成されて
いない。

### アーキテクチャ詳細

親calibrationのtransform、0.25 m inlier band、全gate、model hashをimmutableに読み、
holdout 128 viewsだけをfitと同じdecode/inference/ray-plane adapterで処理した。
holdout結果による再最適化・candidate再選択はない。

### メトリクスの解釈

123/128 viewsがprojection最低数を満たし、aggregate weighted inlier 0.8317とcoverage
0.8070はpassした。一方q95 1.2399 mは上限0.4 mを破った。group 2 coverage 0.2235、
group 6 coverage 0.2690は下限0.35を破った。camera height 1.44–2.82 mは全て正だった。

### アーキテクチャ⇄メトリクスの因果考察

観測として、多くの近距離line evidenceはselected courtへ一致するが、holdoutの一部
groupは対象courtの部分視野しか持たず、ROI内の偽line/outlierがq95を押し上げた。
これはpost-hoc gate緩和を正当化しない。

### 既存実験との比較

親 `run-i618-b00-alignment-calibration-local-v2` のfit q95 0.2097 m / coverage 1.0に
対し、holdoutは1.2399 m / 0.8070へ悪化した。frozen acceptanceにより過適合が顕在化した。

### 次に有効な実験

同じholdoutを再利用せず、新しいcalibrated camera poses/imagesを追加して新しい
untouched acceptance splitを作る。現B00/B02は同一491-camera COLMAP setで、未登録の
`frame_000491.jpg`にはposeがない。
