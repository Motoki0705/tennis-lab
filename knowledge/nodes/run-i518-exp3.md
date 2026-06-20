---
id: run-i518-exp3
type: run
title: exp3 branched 8+2+2
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_branched
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 13.6
  angular_error_median_deg: 9.7
  angle_accuracy_30deg: 0.886
  position_error_m: 0.82
  position_error_median_m: 0.68
artifacts:
  log: experiments/logs/
  output_dir: ''
  curves: knowledge/runs/run-i518-exp3/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_6
parents:
- run-i518-exp1
relations: []
tags:
- plcs
- rotation
- branched
---

## 考察 / Findings

### 要約
回転に専用の分岐 readout を与えると、回転 `13.6°` と位置 `0.82m` の両方向に前進する有望構成。

### アーキテクチャ詳細
分岐モデル `multiview_axial_base_branched`（共有 8 層 + タスク別 2 層）。損失は exp1 と同じ `canonical_rot`。

### メトリクスの解釈
回転 `13.6°`（exp1 の 20.4° からさらに改善）、位置 `1.10→0.82m` と緩和。両方向に前進したが、位置は `0.82m` と高止まり。

### アーキテクチャ⇄メトリクスの因果考察
分岐で回転に専用容量を与えると、共有 trunk 上の競合が緩和され回転が改善。ただし共有 trunk が依然回転支配で、位置は高止まり。

### 既存実験との比較
親 [[run-i518-exp1]] に対し両指標改善（`compares`）。「分岐で回転に容量を与えると効く」を示した枝分かれの基点。

### 次に有効な実験
ここから幅 / 深さ / 分離 / 重みを枝分かれ探索（exp4=重み, exp5=完全分離, exp8=深さ3）。
