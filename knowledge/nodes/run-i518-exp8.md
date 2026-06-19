---
id: run-i518-exp8
type: run
title: exp8 branched3 + pos6 (Pareto)
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_branched3
  loss: canonical_rot_v5
  data: multiview_sequence
metrics:
  ang_error_deg: 12.5
  angular_error_median_deg: 9.9
  angle_accuracy_30deg: 0.937
  position_error_m: 0.60
  position_error_median_m: 0.47
artifacts:
  log: experiments/logs/
  output_dir: ''
parents: [run-i518-exp3]
relations: []
tags: [plcs, rotation, branched, pareto]
---

## 考察 / Findings

分岐3層(8+3+3)で強回転(angle1.0/rot0.5) + 控えめ位置上げ(`canonical_rot_v5`: pos6)。

- 回転 `12.5°`・位置 `0.60m`。**回転を崩さず位置を回収した共有 trunk の Pareto 最良**。
- ただし位置は `0.60m` 止まり。後続 exp9(pos12) は回転崩壊(49.5°)し、**共有 trunk の位置下限は
  ~0.5–0.6m** とフロンティアが確定。

→ 共有/分岐路線の到達点。これ以上は分離 trunk + 補助位置（exp10）でしか越えられない。
