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
  position_error_median_m: 0.43
  position_accuracy_0.5m: 0.61
artifacts:
  log: experiments_mcmc/logs/
  output_dir: ''
  curves: knowledge/runs/run-i519-mcmc-ns01/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_20
parents: []
relations:
- to: run-i519-mcmc-ns03
  rel: compares
tags:
- plcs
- rotation
- mcmc
---

## 考察 / Findings

### 要約
MCMC を学習戦略として導入し 180° 反転の局所最適を脱出できるか検証。結果はノイズなし baseline より角度・位置とも悪化。

### アーキテクチャ詳細
`canonical` ベースライン（rotation 弱）に SGLD ノイズ注入（`theta <- theta - lr*grad + N(0, 2*lr*T)`）を全パラメータへ適用。`mcmc: enabled=true target=all decay=cosine temperature=1.0 noise_scale=0.1`。

### メトリクスの解釈
`73.58° / 0.512m`。ノイズなしベースライン（約 `61.6° / 0.260m`）より角度・位置とも悪化。

### アーキテクチャ⇄メトリクスの因果考察
AdamW は勾配を正規化して約 lr のステップを踏むのに対し、SGLD ノイズ std `sqrt(2*lr)≈0.014` が信号を上書きしてしまう。noise_scale=0.1 でも探索ノイズが学習を阻害する。

### 既存実験との比較
より強いノイズの [[run-i519-mcmc-ns03]] と対（`compares`）。損失ベースで反転を解いた #518（[[run-i518-exp10]]）と対照的に、機構的な探索は効かなかった。

### 次に有効な実験
MCMC（全パラ SGLD）は 180° 反転脱出に効かずむしろ劣化。正解は損失ベースの #518（angle 損失 + 分離 trunk）。MCMC 機構自体は任意オプション（`mcmc.yaml`）として残置。
