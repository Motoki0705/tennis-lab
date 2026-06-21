---
id: run-i518-baseline
type: run
title: baseline (shared trunk, canonical)
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base
  loss: canonical
  data: multiview_sequence
metrics:
  ang_error_deg: 61.6
  angular_error_median_deg: 45.4
  angle_accuracy_15deg: 0.162
  angle_accuracy_30deg: 0.336
  position_error_m: 0.26
  position_error_median_m: 0.213
artifacts:
  log: experiments/logs/
  output_dir: logs/version_2
parents: []
relations: []
tags:
- plcs
- rotation
- baseline
---

## 考察 / Findings

### 要約
#518「回転誤差を下げる」の出発点。位置は良好だが回転が壊滅 (`61.6°`) で、上位誤差はほぼ 180° 前後反転に集中。

### アーキテクチャ詳細
共有 trunk (`multiview_axial_base`) + `canonical` 損失（`rotation_weight=0.02`）, `data=multiview_sequence`。回転とポーズが同一 trunk を共有する素の構成。

### メトリクスの解釈
位置 `0.260m` と既に良好な一方、回転 `ang_error 61.6°` / 中央値 `45.4°`, `acc@15 16.2%` と壊滅的。上位誤差は `pred_rotation = -gt_rotation` のほぼ 180° 前後反転に集中。

### アーキテクチャ⇄メトリクスの因果考察
原因は 2 点。(1) `rotation_weight=0.02` が小さすぎ回転ヘッドが学習不足。(2) `1-cos` 損失が 180° で平坦なサドル（grad = sinθ → 反対点で 0）を持ち、反転が安定な局所最適になっていた。

### 既存実験との比較
フロンティア探索の基準点（`parents` なし）。最終的に [[run-i518-exp10]] が両指標同時改善でこの壊滅を解決する。

### 次に有効な実験
exp1〜exp10 で「反転サドル対策（angle 損失）」と「容量競合の解消（trunk 設計）」を枝分かれ探索する。
