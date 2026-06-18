---
id: run-i520-canon-none
type: run
title: canon_none
issue: 520
provider: claude
status: done
config:
  model: multiview_axial_canon_split_none
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 17.788857
  angle_accuracy: 0.548351
  angle_accuracy_10deg: 0.393971
  angle_accuracy_15deg: 0.548351
  angle_accuracy_30deg: 0.839181
  angular_error_deg: 17.517141
  angular_error_median_deg: 13.305781
  angular_error_std_deg: 16.778427
  loss: 0.158601
  loss_canonical_pose: 0.004325
  pos_error_m: 0.273591
  position_accuracy: 0.900633
  position_accuracy_0.5m: 0.900633
  position_accuracy_1m: 0.971411
  position_accuracy_2m: 0.999945
  position_error_m: 0.272502
  position_error_median_m: 0.1895
  position_error_std_m: 0.27175
  x_error_m: 0.107655
  y_error_m: 0.22129
  z_error_m: 0.044461
artifacts:
  log: .training_queue/logs/1781739570303919093_6452_canon_none.log
  job: .training_queue/done/1781739570303919093_6452_canon_none.job
  output_dir: ''
parents: []
relations: []
tags: [plcs, canonical, split-trunk, baseline]
---

## 考察 / Findings

#518 で「パスの分離が効く」という結論が出たため、canonical pose の**パスを分離しない**構成を
基準として測定。`ang_error 17.79°`, `position_error 0.273m`。#520 の canonical 分離アブレーション
（rot / pos / both）の baseline であり、他 3 ノードはここからの派生として比較する。

