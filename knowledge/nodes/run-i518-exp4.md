---
id: run-i518-exp4
type: run
title: exp4 branched + pos30
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_branched
  loss: canonical_rot_v3
  data: multiview_sequence
metrics:
  ang_error_deg: 54.1
  angular_error_median_deg: 26.9
  angle_accuracy_30deg: 0.529
  position_error_m: 0.4
  position_error_median_m: 0.29
artifacts:
  log: experiments/logs/
  output_dir: ''
  curves: knowledge/runs/run-i518-exp4/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_7
parents:
- run-i518-exp3
relations:
- to: run-i518-exp3
  rel: compares
tags:
- plcs
- rotation
- branched
- reweight
---

## 考察 / Findings

### 要約
分岐モデルで位置重みを強く上げると、位置は下がるが回転が崩壊。分岐だけでは共有 trunk の競合を断てない。

### アーキテクチャ詳細
exp3 と同じ分岐モデル `multiview_axial_base_branched`。損失を `canonical_rot_v3`（position 30）に変更。

### メトリクスの解釈
位置は `0.82→0.40m` に下がるが、回転は `13.6→54.1°` に崩壊。

### アーキテクチャ⇄メトリクスの因果考察
分岐 readout があっても、共有 trunk の勾配は position を上げると回転を潰す。exp2 と同じ轍。

### 既存実験との比較
親 [[run-i518-exp3]] と対（`compares`）。分岐だけでは位置を上げると回転が崩れることを確認。

### 次に有効な実験
分岐だけでは不十分。trunk そのものの勾配共有を断つ（分離 / detach）方向を exp5 / exp6 で試す。
