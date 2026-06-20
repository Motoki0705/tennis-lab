---
id: run-i525-shared-6l
type: run
title: i525_shared_6l
issue: 525
provider: claude
status: done
config:
  model: multiview_axial_shared_6l
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 15.272086
  angle_accuracy: 0.645513
  angle_accuracy_10deg: 0.47168
  angle_accuracy_15deg: 0.645513
  angle_accuracy_30deg: 0.873525
  angular_error_deg: 15.592398
  angular_error_median_deg: 10.638618
  angular_error_std_deg: 16.42556
  loss: 0.140613
  loss_canonical_pose: 0.006464
  pos_error_m: 0.835967
  position_accuracy: 0.331842
  position_accuracy_0.5m: 0.331842
  position_accuracy_1m: 0.75007
  position_accuracy_2m: 0.947442
  position_error_m: 0.81912
  position_error_median_m: 0.614192
  position_error_std_m: 0.77607
  x_error_m: 0.348879
  y_error_m: 0.658233
  z_error_m: 0.079557
artifacts:
  log: .training_queue/logs/1781794459012091481_270284_i525_shared_6l.log
  job: .training_queue/done/1781794459012091481_270284_i525_shared_6l.job
  output_dir: ''
  curves: knowledge/runs/run-i525-shared-6l/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_32
parents:
- run-i521-ex10-vel
relations:
- to: run-i525-shared-match-dim
  rel: compares
- to: run-i525-shared-match-layers
  rel: compares
tags:
- plcs
- canonical
- shared-trunk
---

## 考察 / Findings

### 要約
#525 の基準（native パラメータ）共有 trunk。半分のパラメータでも位置は ~0.84m で破綻し、分離型優位はアーキ起因という結論の基準点。

### アーキテクチャ詳細
`multiview_axial_shared_6l` + `canonical_rot`：単一 6 層共有 trunk（39.3M＝分離 EX10 78.06M の約半分）。param-matched 共有（[[run-i525-shared-match-dim]] / [[run-i525-shared-match-layers]]）はここからパラメータを増やした子。

### メトリクスの解釈
`15.27° / 0.836m`。半分のパラメータでも位置は既に `0.84m` で破綻し、EX10 の `0.238m` に遠い。パラメータ増でも位置は不変（幅倍化 match-dim 79M = `0.848m`）か悪化（深さ倍化 match-layers 78M = `1.617m`）。

### アーキテクチャ⇄メトリクスの因果考察
位置は共有 trunk ではパラメータ容量によらず約 0.84m で頭打ち。層を積むと共有 trunk で特徴が混ざり両タスク崩壊する（match-layers）。

### 既存実験との比較
[[run-i525-shared-match-dim]]（角度は EX10 に肉薄・位置不変）と [[run-i525-shared-match-layers]]（悪化）の基準点（`compares`）。分離 EX10（[[run-i518-exp10]]）と対照。

### 次に有効な実験
分離 trunk の優位はパラメータ数ではなくアーキ（タスク分離）に起因、という #525 結論の基準。以後は split 構造での容量スイープ（#540/#541）へ。
