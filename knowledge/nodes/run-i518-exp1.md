---
id: run-i518-exp1
type: run
title: exp1 angle loss (shared)
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base
  loss: canonical_rot
  data: multiview_sequence
metrics:
  ang_error_deg: 20.4
  angular_error_median_deg: 16.4
  angle_accuracy_30deg: 0.789
  position_error_m: 1.1
  position_error_median_m: 0.99
artifacts:
  log: experiments/logs/
  output_dir: ''
  curves: knowledge/runs/run-i518-exp1/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_4
parents:
- run-i518-baseline
relations: []
tags:
- plcs
- rotation
- angle-loss
---

## 考察 / Findings

### 要約
wrapped-angle smooth-L1 の `angle` 損失で反転サドルを潰すと回転は `61.6→20.4°` と大幅改善するが、共有 trunk のため位置が崩壊。

### アーキテクチャ詳細
baseline と同じ共有 trunk (`multiview_axial_base`)。損失のみ `canonical_rot`（`angle_weight=1.0`, `rotation_weight=0.5`）に変更。`angle` 損失は 180° でも勾配が消えない（grad@180 = [0, 1.0]）。

### メトリクスの解釈
回転 `61.6→20.4°` と大幅改善し、反転問題に angle 損失が効くことを実証。一方で位置が `0.26→1.10m` に崩壊。

### アーキテクチャ⇄メトリクスの因果考察
angle 損失で回転 trunk が強く学習される結果、共有 trunk の容量を回転が奪い、位置が劣化する。回転と位置が trunk 容量を奪い合う競合構図が露呈した。

### 既存実験との比較
親 [[run-i518-baseline]] に対し回転は改善・位置は悪化（`compares`）。「angle 損失は必須だが共有 trunk では位置と競合する」というトレードオフの起点。

### 次に有効な実験
この競合を、損失リバランス（exp2）と分岐/分離アーキ（exp3 以降）の両面から解く。
