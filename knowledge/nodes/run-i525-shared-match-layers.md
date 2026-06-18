---
id: run-i525-shared-match-layers
type: run
title: i525_shared_match_layers
issue: 525
provider: claude
status: done
config:
  model: multiview_axial_shared_match_layers
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 29.80065
  angle_accuracy: 0.34992
  angle_accuracy_10deg: 0.213015
  angle_accuracy_15deg: 0.34992
  angle_accuracy_30deg: 0.642657
  angular_error_deg: 29.416918
  angular_error_median_deg: 21.524656
  angular_error_std_deg: 28.282793
  loss: 0.371826
  loss_canonical_pose: 0.007538
  pos_error_m: 1.617211
  position_accuracy: 0.098582
  position_accuracy_0.5m: 0.098582
  position_accuracy_1m: 0.368513
  position_accuracy_2m: 0.769663
  position_error_m: 1.580652
  position_error_median_m: 1.282007
  position_error_std_m: 1.142458
  x_error_m: 0.744425
  y_error_m: 1.228535
  z_error_m: 0.089119
artifacts:
  log: .training_queue/logs/1781750120028938555_67076_i525_shared_match_layers.log
  job: .training_queue/done/1781750120028938555_67076_i525_shared_match_layers.job
  output_dir: ''
parents: []
relations:
  - {to: run-i521-ex10-vel, rel: compares}
tags: [plcs, canonical, shared-trunk]
---

## 考察 / Findings

#525 の対照のもう一方。共有 trunk を**層数増でパラメータ数を合わせた**構成。学習が崩壊し
`ang_error 29.80°`, `position_error 1.58m` と大幅悪化。

→ **同じパラメータ予算でも増やし方（次元拡張 vs 層数増）で安定性が激変する。** 単純な「パラメータ数」
だけでは説明できず、アーキテクチャ/最適化の寄与を示す**負の対照**。`shared_match_dim` と合わせると、
分離 trunk の優位は一部パラメータ数で説明できるが層を深くした共有 trunk は不安定、と読める。

