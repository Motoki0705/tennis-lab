---
id: run-i518-exp9
type: run
title: exp9 branched3 + pos12
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_branched3
  loss: canonical_rot_v6
  data: multiview_sequence
metrics:
  ang_error_deg: 49.5
  angular_error_median_deg: 25.5
  angle_accuracy_30deg: 0.549
  position_error_m: 0.59
  position_error_median_m: 0.51
artifacts:
  log: experiments/logs/
  output_dir: ''
  curves: knowledge/runs/run-i518-exp9/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_13
parents:
- run-i518-exp8
relations:
- to: run-i518-exp8
  rel: compares
tags:
- plcs
- rotation
- branched
- reweight
---

## 考察 / Findings

### 要約
exp8 からさらに位置重みを上げると回転が崖のように崩壊（12.5→49.5°）し、位置は改善せず。共有 trunk の位置下限 ~0.5–0.6m を確定。

### アーキテクチャ詳細
exp8 と同じ `multiview_axial_base_branched3`。損失 `canonical_rot_v6`（位置 pos12 にさらに増）。

### メトリクスの解釈
回転が `12.5→49.5°` に崩壊、しかも位置は `0.59m` で exp8 (0.60m) と同じく改善せず。

### アーキテクチャ⇄メトリクスの因果考察
位置重み 6 を超えると回転が急落する一方、位置はこれ以上下がらない。明確な「崖」で、共有 trunk の位置下限 ~0.5–0.6m を確定させる。

### 既存実験との比較
親 [[run-i518-exp8]] に対し位置重みだけを上げた対照（`compares`）。9 実験でフロンティアを精密マッピング完了。

### 次に有効な実験
この直後、claude が [[run-i518-exp7]] の失敗分析から [[run-i518-exp10]] を自律的に着想・実装した。
