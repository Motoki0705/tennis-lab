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
parents: [run-i518-exp8]
relations:
  - {to: run-i518-exp8, rel: compares}
tags: [plcs, rotation, branched, reweight]
---

## 考察 / Findings

exp8 から位置重みをさらに上げ(`canonical_rot_v6`: pos12)、位置を baseline(0.26m)へ近づけられるか。

- **回転が `12.5 → 49.5°` に崩壊、しかも位置は `0.59m` で改善せず**（exp8 の 0.60m と同じ）。
- 明確な「崖」: 位置重み6を超えると回転が急落する一方、位置はこれ以上下がらない。**共有 trunk の
  位置下限 ~0.5–0.6m を確定**。

→ 9実験でフロンティアを精密マッピング完了。この直後、claude が exp7 の失敗分析から
  [[run-i518-exp10]] を自律的に着想・実装した。
