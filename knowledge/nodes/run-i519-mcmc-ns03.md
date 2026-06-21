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
  curves: knowledge/runs/run-i519-mcmc-ns03/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_21
parents:
- run-i519-mcmc-ns01
relations: []
tags:
- plcs
- rotation
- mcmc
---

## 考察 / Findings

### 要約
#519 の MCMC/SGLD をより強いノイズで適用すると、`ns=0.1` よりさらに悪化。ノイズを強めるほど劣化する単調傾向。

### アーキテクチャ詳細
[[run-i519-mcmc-ns01]] と同構成で `noise_scale=0.3` に増強（`mcmc: target=all decay=cosine temperature=1.0`）。

### メトリクスの解釈
`76.11° / 0.754m`。`ns=0.1`（`73.58° / 0.512m`）よりさらに悪化。

### アーキテクチャ⇄メトリクスの因果考察
ノイズを強めるほど劣化が進行する単調傾向は、「探索ノイズが AdamW の勾配信号を上書きする」という ns=0.1 の解釈を裏づける。

### 既存実験との比較
親 [[run-i519-mcmc-ns01]] にノイズ強度だけを変えた対照。noise_scale を振っても MCMC は寄与せず両指標を悪化させるのみ。

### 次に有効な実験
#519 の結論はネガティブ（MCMC は不採用、損失ベースの #518 が正解）。これ以上 noise_scale を振る価値はない。
