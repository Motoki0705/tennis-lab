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
parents: []
relations:
  - {to: run-i525-shared-match-dim, rel: compares}
  - {to: run-i525-shared-match-layers, rel: compares}
tags: [plcs, canonical, shared-trunk]
---

## 考察 / Findings

#525 の **基準（native パラメータ）共有 trunk**。単一 6 層共有 trunk（39.3M＝分離 EX10 78.06M の約半分）。
param-matched 共有 trunk（`shared-match-dim` / `shared-match-layers`）はこのノードからパラメータを
増やした子に当たる。

- 結果 `15.27° / 0.836m`。**半分のパラメータでも位置は既に 0.84m で破綻**しており、EX10 の 0.238m に遠い。
- パラメータを増やしても位置は救えない：
  - 幅で倍化 `shared-match-dim`(79M) = `12.22° / 0.848m` … 角度は EX10 に肉薄するが位置は不変。
  - 深さで倍化 `shared-match-layers`(78M) = `29.80° / 1.617m` … **この 6L 基準より悪化**（層を積むと共有
    trunk で特徴が混ざり両タスク崩壊）。

→ **位置は共有 trunk ではパラメータ容量によらず約 0.84m で頭打ち**。分離 trunk の優位はパラメータ数では
なくアーキテクチャ（タスク分離）に起因する、という #525 結論を基準点として裏づける。
