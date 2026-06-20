---
id: run-i521-ex10-vel
type: run
title: i521_ex10_vel
issue: 521
provider: claude
status: done
config:
  model: multiview_axial_base_split_auxpos
  loss: canonical_rot_vel
  data: multiview_sequence
metrics:
  ang_error_deg: 13.217234
  angle_accuracy: 0.700764
  angle_accuracy_10deg: 0.533277
  angle_accuracy_15deg: 0.700764
  angle_accuracy_30deg: 0.903535
  angular_error_deg: 13.215156
  angular_error_median_deg: 9.211813
  angular_error_std_deg: 12.821545
  loss: 0.108247
  loss_canonical_pose: 0.005932
  pos_error_m: 0.269793
  position_accuracy: 0.885019
  position_accuracy_0.5m: 0.885019
  position_accuracy_1m: 0.968665
  position_accuracy_2m: 1.0
  position_error_m: 0.268576
  position_error_median_m: 0.199097
  position_error_std_m: 0.244776
  x_error_m: 0.099376
  y_error_m: 0.224301
  z_error_m: 0.04266
artifacts:
  log: .training_queue/logs/1781786377943542652_240865_i521_ex10_vel.log
  job: .training_queue/done/1781786377943542652_240865_i521_ex10_vel.job
  output_dir: ''
  curves: knowledge/runs/run-i521-ex10-vel/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_30
parents:
- run-i521-base-vel
relations: []
tags:
- plcs
- canonical
- velocity
- split-trunk
---

## 考察 / Findings

### 要約
velocity 損失 + 位置補助パス分離。良好な角度を保ちつつ位置も維持し、#521 内でバランス最良。

### アーキテクチャ詳細
`model=multiview_axial_base_split_auxpos`（EX10 の分離 + 位置補助）+ `canonical_rot_vel`。

### メトリクスの解釈
`ang_error 13.22°` と良好な角度を保ちつつ、`position_error 0.269m` を維持。#521 内で角度 / 位置のバランスが最良。

### アーキテクチャ⇄メトリクスの因果考察
EX10 の補助位置パスが、velocity 損失で生じた位置悪化を回収する。角度を担う回転 trunk と位置を担うパスが競合せず両立する。

### 既存実験との比較
親 [[run-i521-base-vel]]（`11.34° / 0.786m`）の位置悪化を回収。EX10（[[run-i518-exp10]]）に velocity を足した系で、canonical-trunk + velocity の [[run-i521-canonboth-vel]] を角度・位置とも上回る。

### 次に有効な実験
velocity 損失 + auxpos 分離が現状の有力構成。以後の容量スケール（#540/#541）でも本系を基準にする。
