---
id: run-i518-exp6
type: run
title: exp6 branched + detach
issue: 518
provider: claude
date: '2026-06-17'
status: done
config:
  model: multiview_axial_base_branched_detach
  loss: canonical_rot_v4
  data: multiview_sequence
metrics:
  ang_error_deg: 73.6
  angular_error_median_deg: 62.6
  angle_accuracy_30deg: 0.268
  position_error_m: 2.77
  position_error_median_m: 2.26
artifacts:
  log: experiments/logs/
  output_dir: ''
  curves: knowledge/runs/run-i518-exp6/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial/logs/version_10
parents:
- run-i518-exp5
relations: []
tags:
- plcs
- rotation
- detach
---

## 考察 / Findings

### 要約
ポーズ分岐を detach して回転と位置を両立させようとしたが、回転 `73.6°`・位置 `2.77m` と両方崩壊する最悪の結果。

### アーキテクチャ詳細
`multiview_axial_base_branched_detach`：ポーズ分岐を共有 trunk から detach。損失 `canonical_rot_v4`（強回転 angle1.0/rot0.5 + 強位置 pos30）。

### メトリクスの解釈
回転 `73.6°`・位置 `2.77m` と両方崩壊。位置すら破綻した。

### アーキテクチャ⇄メトリクスの因果考察
detach で勾配を切ると三角測量特徴が trunk に乗らず、位置すら破綻する。exp5 の「位置勾配を切ると壊れる」をさらに強く裏づけ。

### 既存実験との比較
親 [[run-i518-exp5]] の知見を補強。勾配を切る方向は全滅という結論を確定させる。

### 次に有効な実験
勾配を切る方向は全滅。位置信号は切らずに回転 trunk へ流す必要がある（exp10 の核心）。
