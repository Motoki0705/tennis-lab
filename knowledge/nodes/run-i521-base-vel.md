---
id: run-i521-base-vel
type: run
title: i521_base_vel
issue: 521
provider: claude
status: done
config:
  model: multiview_axial_base
  loss: canonical_rot_vel
  data: multiview_sequence
metrics:
  ang_error_deg: 11.338284
  angle_accuracy: 0.762629
  angle_accuracy_10deg: 0.600636
  angle_accuracy_15deg: 0.762629
  angle_accuracy_30deg: 0.941707
  angular_error_deg: 11.354784
  angular_error_median_deg: 7.975065
  angular_error_std_deg: 12.375816
  loss: 0.092704
  loss_canonical_pose: 0.005976
  pos_error_m: 0.790302
  position_accuracy: 0.389374
  position_accuracy_0.5m: 0.389374
  position_accuracy_1m: 0.77733
  position_accuracy_2m: 0.948462
  position_error_m: 0.786041
  position_error_median_m: 0.629319
  position_error_std_m: 0.661727
  x_error_m: 0.359513
  y_error_m: 0.607698
  z_error_m: 0.065605
artifacts:
  log: .training_queue/logs/1781786377929126009_240853_i521_base_vel.log
  job: .training_queue/done/1781786377929126009_240853_i521_base_vel.job
  output_dir: ''
  curves: knowledge/runs/run-i521-base-vel/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_29
parents:
- run-i520-canon-none
relations:
- to: run-i520-canon-none
  rel: improves
tags:
- plcs
- canonical
- velocity
---

## 考察 / Findings

### 要約
関節角度の角速度（時間）損失を導入。角度は大幅改善するが位置は大きく悪化。

### アーキテクチャ詳細
`model=multiview_axial_base`（共有 trunk）+ `canonical_rot_vel`（角速度損失を追加）。

### メトリクスの解釈
`ang_error 11.34°` と #520 baseline（[[run-i520-canon-none]]: `17.79°`）から大幅改善。一方 `position_error 0.786m` と位置は大きく悪化。

### アーキテクチャ⇄メトリクスの因果考察
時間損失が「ポーズが固まって平均に収束する」問題を解き角度精度を強く押し上げる。ただし共有 trunk のため角度に容量を取られ位置が悪化する。

### 既存実験との比較
親 [[run-i520-canon-none]] に対し角度を大きく改善（`improves`）。位置悪化は分離 auxpos で回収できるか（[[run-i521-ex10-vel]]）を次に検証。

### 次に有効な実験
velocity 損失は角度を強く押し上げるが位置を犠牲にする。位置回収のため補助パス分離を組み合わせる（[[run-i521-ex10-vel]]）。
