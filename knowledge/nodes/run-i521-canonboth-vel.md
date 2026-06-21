---
id: run-i521-canonboth-vel
type: run
title: i521_canonboth_vel
issue: 521
provider: claude
status: done
config:
  model: multiview_axial_canon_split_both
  loss: canonical_rot_vel
  data: multiview_sequence
metrics:
  ang_error_deg: 17.019215
  angle_accuracy: 0.545412
  angle_accuracy_10deg: 0.388172
  angle_accuracy_15deg: 0.545412
  angle_accuracy_30deg: 0.849904
  angular_error_deg: 17.091824
  angular_error_median_deg: 13.406963
  angular_error_std_deg: 15.41909
  loss: 0.182384
  loss_canonical_pose: 0.005258
  pos_error_m: 0.297601
  position_accuracy: 0.834689
  position_accuracy_0.5m: 0.834689
  position_accuracy_1m: 0.984851
  position_accuracy_2m: 1.0
  position_error_m: 0.296807
  position_error_median_m: 0.235366
  position_error_std_m: 0.235928
  x_error_m: 0.103921
  y_error_m: 0.251609
  z_error_m: 0.043049
artifacts:
  log: .training_queue/logs/1781786377957743176_240877_i521_canonboth_vel.log
  job: .training_queue/done/1781786377957743176_240877_i521_canonboth_vel.job
  output_dir: ''
  curves: knowledge/runs/run-i521-canonboth-vel/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_31
parents:
- run-i520-canon-both
relations:
- to: run-i521-ex10-vel
  rel: compares
tags:
- plcs
- canonical
- velocity
- canonical-trunk
---

## 考察 / Findings

### 要約
#520 最良の canonical-trunk 構成に velocity 損失を導入。角度はわずかに悪化、位置は改善するが、EX10 系には届かない。

### アーキテクチャ詳細
`model=multiview_axial_canon_split_both`（#520 の canon_both）+ `canonical_rot_vel`。#521 の 3 本目。

### メトリクスの解釈
`17.02° / 0.298m`。親 [[run-i520-canon-both]]（velocity なし, `15.91° / 0.364m`）比で角度はわずかに悪化、位置は改善。

### アーキテクチャ⇄メトリクスの因果考察
velocity 損失は canonical-trunk 構成では位置側に効いた。ただし canonical-trunk 自体が EX10（分離 auxpos）に劣るため、velocity を足しても EX10 系には届かない。

### 既存実験との比較
親 [[run-i520-canon-both]] に velocity を追加した対。#521 内では [[run-i521-ex10-vel]]（`13.22° / 0.270m`）に角度・位置とも劣る（`compares`）。

### 次に有効な実験
velocity 損失は静的精度を一変させはしないが canonical-trunk では位置を改善する方向。本命は引き続き EX10（分離 auxpos）+ velocity（[[run-i521-ex10-vel]]）。
