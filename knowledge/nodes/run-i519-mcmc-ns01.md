---
id: run-i519-mcmc-ns01
type: run
title: MCMC/SGLD noise_scale=0.1
issue: 519
provider: claude
date: '2026-06-19'
status: done
config:
  model: multiview_axial_base
  loss: canonical
  data: multiview_sequence
  mcmc: enabled=true target=all decay=cosine temperature=1.0 noise_scale=0.1
metrics:
  ang_error_deg: 73.58
  angular_error_median_deg: 61.34
  angle_accuracy_15deg: 0.132
  angle_accuracy_30deg: 0.279
  position_error_m: 0.512
  position_error_median_m: 0.430
  position_accuracy_0.5m: 0.610
artifacts:
  log: experiments_mcmc/logs/
  output_dir: ''
parents: []
relations:
  - {to: run-i519-mcmc-ns03, rel: compares}
tags: [plcs, rotation, mcmc]
---

## 考察 / Findings

#519「MCMC を学習戦略として導入し 180° 反転の局所最適を脱出できるか」の検証。`canonical` ベースライン
（rotation 弱）に SGLD ノイズ注入 (`theta <- theta - lr*grad + N(0, 2*lr*T)`) を全パラメータへ適用。

- 結果 `73.58° / 0.512m`。**ノイズなしベースライン（約 61.6° / 0.260m）より角度・位置とも悪化**。
- 原因: AdamW は勾配を正規化して約 lr のステップを踏むのに対し、SGLD ノイズ std `sqrt(2*lr)≈0.014` が
  信号を上書きしてしまう。noise_scale=0.1 でも探索ノイズが学習を阻害。

→ MCMC（全パラ SGLD）は 180° 反転脱出に効かず、むしろ劣化。**正解は損失ベースの #518（angle 損失
＋分離 trunk）**。MCMC 機構自体は任意オプション（`mcmc.yaml`）として残置。
