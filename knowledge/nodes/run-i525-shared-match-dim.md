---
id: run-i525-shared-match-dim
type: run
title: i525_shared_match_dim
issue: 525
provider: claude
status: done
config:
  model: multiview_axial_shared_match_dim
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 12.223938
  angle_accuracy: 0.694056
  angle_accuracy_10deg: 0.510524
  angle_accuracy_15deg: 0.694056
  angle_accuracy_30deg: 0.93843
  angular_error_deg: 12.205566
  angular_error_median_deg: 9.707562
  angular_error_std_deg: 11.089599
  loss: 0.097907
  loss_canonical_pose: 0.006897
  pos_error_m: 0.847612
  position_accuracy: 0.285907
  position_accuracy_0.5m: 0.285907
  position_accuracy_1m: 0.74384
  position_accuracy_2m: 0.958635
  position_error_m: 0.841916
  position_error_median_m: 0.703325
  position_error_std_m: 0.620205
  x_error_m: 0.407429
  y_error_m: 0.643985
  z_error_m: 0.079004
artifacts:
  log: .training_queue/logs/1781750120049888219_67088_i525_shared_match_dim.log
  job: .training_queue/done/1781750120049888219_67088_i525_shared_match_dim.job
  output_dir: ''
parents:
- run-i521-ex10-vel
relations:
  - {to: run-i521-ex10-vel, rel: compares}
tags: [plcs, canonical, shared-trunk]
---

## 考察 / Findings

#525 の問い「分離 trunk の優位は単にパラメータ数が多いだけでは？」への対照。共有 trunk を
**次元拡張でパラメータ数を分離型 EX10 に合わせた**構成。`ang_error 12.22°` と角度は分離型に匹敵する
一方、`position_error 0.842m` と位置は大きく崩れる。

→ **角度面ではパラメータ数を合わせた共有 trunk でも分離型に並ぶ**（「パラメータ数」説を一部支持）。
ただし位置は崩れ、分離型と等価ではない。

