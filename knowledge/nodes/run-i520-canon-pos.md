---
id: run-i520-canon-pos
type: run
title: canon_pos
issue: 520
provider: claude
status: failed
config:
  model: multiview_axial_canon_split_pos
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 20.903999
  angle_accuracy: 0.491325
  angle_accuracy_10deg: 0.32456
  angle_accuracy_15deg: 0.491325
  angle_accuracy_30deg: 0.764369
  angular_error_deg: 20.667231
  angular_error_median_deg: 15.308434
  angular_error_std_deg: 18.430874
  loss: 0.209659
  loss_canonical_pose: 0.004551
  pos_error_m: 0.284956
  position_accuracy: 0.854125
  position_accuracy_0.5m: 0.854125
  position_accuracy_1m: 0.974271
  position_accuracy_2m: 1.0
  position_error_m: 0.2884
  position_error_median_m: 0.212009
  position_error_std_m: 0.262608
  x_error_m: 0.106768
  y_error_m: 0.236256
  z_error_m: 0.044481
artifacts:
  log: .training_queue/logs/1781739570277990308_6428_canon_pos.log
  job: .training_queue/failed/1781739570277990308_6428_canon_pos.job
  output_dir: ''
  curves: knowledge/runs/run-i520-canon-pos/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_24
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
canonical の**位置パスのみ**を分離。記録値は baseline より悪化で、位置パス単独分離は逆効果の可能性。

### アーキテクチャ詳細
`multiview_axial_canon_split_pos` + `canonical_rot`。canonical の位置側だけ専用パスに分離。キューでは `failed` 終了（test 評価値はログに残存。最終的な成否は要再検証）。

### メトリクスの解釈
`ang_error 20.90°`, `position_error 0.288m` で baseline（[[run-i520-canon-none]]）より悪化。

### アーキテクチャ⇄メトリクスの因果考察
位置パスのみを切り出すと、回転と位置の協調が崩れて角度がむしろ悪化したと見られる（要再検証）。

### 既存実験との比較
親 [[run-i520-canon-none]] より悪化。両分離 [[run-i520-canon-both]]（角度最良）と逆方向の結果。

### 次に有効な実験
`failed` 終了のため再走で確認しつつ、有望なのは両パス分離（[[run-i520-canon-both]]）方向。
