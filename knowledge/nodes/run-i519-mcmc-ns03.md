---
id: run-i519-mcmc-ns03
type: run
title: MCMC/SGLD noise_scale=0.3
issue: 519
provider: claude
date: '2026-06-19'
status: done
config:
  model: multiview_axial_base
  loss: canonical
  data: multiview_sequence
  mcmc: enabled=true target=all decay=cosine temperature=1.0 noise_scale=0.3
metrics:
  ang_error_deg: 76.11
  angular_error_median_deg: 66.07
  angle_accuracy_15deg: 0.128
  angle_accuracy_30deg: 0.258
  position_error_m: 0.754
  position_error_median_m: 0.627
  position_accuracy_0.5m: 0.357
artifacts:
  log: experiments_mcmc/logs/
  output_dir: ''
parents: [run-i519-mcmc-ns01]
relations: []
tags: [plcs, rotation, mcmc]
---

## 考察 / Findings

#519 の MCMC/SGLD をより強いノイズ (`noise_scale=0.3`) で適用した版。

- 結果 `76.11° / 0.754m`。`ns=0.1`（`73.58° / 0.512m`）より**さらに悪化**。ノイズを強めるほど劣化が
  進行する単調な傾向。
- これは「探索ノイズが AdamW の勾配信号を上書きする」という ns=0.1 の解釈を裏づける。

→ noise_scale を振っても MCMC は 180° 反転脱出に寄与せず、両指標を悪化させるのみ。#519 の結論は
**ネガティブ（MCMC は不採用、損失ベースの #518 が正解）**。
