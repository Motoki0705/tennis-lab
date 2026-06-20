---
id: run-i520-canon-rot
type: run
title: canon_rot
issue: 520
provider: claude
status: done
config:
  model: multiview_axial_canon_split_rot
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 17.599337
  angle_accuracy: 0.552671
  angle_accuracy_10deg: 0.387384
  angle_accuracy_15deg: 0.552671
  angle_accuracy_30deg: 0.837973
  angular_error_deg: 17.284174
  angular_error_median_deg: 13.259541
  angular_error_std_deg: 16.824196
  loss: 0.325704
  loss_canonical_pose: 0.006695
  pos_error_m: 0.265129
  position_accuracy: 0.865626
  position_accuracy_0.5m: 0.865626
  position_accuracy_1m: 0.973163
  position_accuracy_2m: 0.991974
  position_error_m: 0.274716
  position_error_median_m: 0.184736
  position_error_std_m: 0.285623
  x_error_m: 0.100457
  y_error_m: 0.225406
  z_error_m: 0.043585
artifacts:
  log: .training_queue/logs/1781739570264889499_6416_canon_rot.log
  job: .training_queue/done/1781739570264889499_6416_canon_rot.job
  output_dir: ''
  curves: knowledge/runs/run-i520-canon-rot/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_23
parents:
- run-i520-canon-none
relations: []
tags:
- plcs
- canonical
- split-trunk
---

## 考察 / Findings

### 要約
canonical の**回転パスのみ**を分離。baseline とほぼ同等で、回転パス単独の分離では改善が出ない。

### アーキテクチャ詳細
`multiview_axial_canon_split_rot` + `canonical_rot`。canonical の回転側だけ専用パスに分離。

### メトリクスの解釈
`ang_error 17.60°`, `position_error 0.275m`。baseline（[[run-i520-canon-none]]: `17.79° / 0.273m`）とほぼ同等。

### アーキテクチャ⇄メトリクスの因果考察
回転パス単独の分離では容量配分がほぼ変わらず、角度も位置も動かない。

### 既存実験との比較
親 [[run-i520-canon-none]] と実質差なし。両分離 [[run-i520-canon-both]] が角度を伸ばすのと対照的。

### 次に有効な実験
回転単独では不足。両パス分離（[[run-i520-canon-both]]）が角度に効くかを比較。
