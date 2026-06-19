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
  position_error_m: 0.260
  position_error_median_m: 0.213
artifacts:
  log: experiments/logs/
  output_dir: logs/version_2
parents: []
relations: []
tags: [plcs, rotation, baseline]
---

## 考察 / Findings

#518「回転誤差を下げる」の出発点。共有 trunk + `canonical` 損失（`rotation_weight=0.02`）。

- 位置は既に良好 (`0.260m`) だが回転が `61.6°` と壊滅的。上位誤差は**ちょうど ~180° の前後反転**
  (`pred_rotation = -gt_rotation`) に集中。
- 原因2点: (1) `rotation_weight` が小さすぎて回転ヘッドが学習不足、(2) `1-cos` 損失が **180° に
  平坦なサドル**（grad = sinθ → 反対点で 0）を持ち、反転が安定な局所最適になっていた。

→ ここから exp1〜exp10 のフロンティア探索が始まる。最終的に [[run-i518-exp10]] が両指標同時改善で解決。
