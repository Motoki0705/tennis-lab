---
id: run-i786-plcs-norm-v1
type: run
title: PLCS court座標正規化 v1 baseline
issue: 786
provider: codex
session: 01a028df-6133-79c2-9266-f9b6a343765f
date: '2026-08-23'
status: done
config:
  model: multiview_axial_base
  loss: canonical
  data: multiview_sequence_norm_v1
metrics:
  loss: 0.0379030295
  position_error_m: 0.4698834717
  x_error_m: 0.1916563213
  y_error_m: 0.3854689598
  z_error_m: 0.0446767062
  angular_error_deg: 52.0611343384
repro:
  commit: b356bbeb6bedeb43d4dd07c0b574f4eddc8c4f8c
  branch: feat/issue-786-normalization-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_norm_v1 paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/data run.output_dir=plcs/i786/norm-v1/multiview_axial_base
artifacts:
  run_dir: knowledge/runs/run-i786-plcs-norm-v1
  log: .training_queue/logs/1787463605227286681_468781_i786-plcs-norm-v1.log
  tb_logdir: outputs/plcs/i786/norm-v1/multiview_axial_base/logs/version_0
  curves: knowledge/runs/run-i786-plcs-norm-v1/curves.png
parents: []
relations: []
tags:
- plcs
- normalization-v1
- baseline
---

## 考察 / Findings

### 要約

legacy軸別scaleと既存v1 datasetを用いたPLCS baseline。epoch 89まで学習し、test `position_error_m=0.4699m`、`angular_error_deg=52.06°`だった。

### アーキテクチャ詳細

`multiview_axial_base`、canonical loss、batch size 4、bf16 mixed precision、`torch.compile`を使用した。v1互換のposition lossとartifact契約を維持している。

### メトリクスの解釈

軸別MAEはX `0.1917m`、Y `0.3855m`、Z `0.0447m`。position誤差はYが支配的で、Zが非常に小さい。

### アーキテクチャ⇄メトリクスの因果考察

v1ではZ scaleが`1.07m`でnormalized lossの物理勾配係数が大きい。この契約と小さいZ誤差は整合するが、姿勢・dataset分布も影響するため本runだけでは因果を分離できない。

### 既存実験との比較

同Issueのv2継続run `run-i786-plcs-v2-resume-b24-r2`のbaseline。ただしv2は途中からbatch 24へ変更されており、完全な一変数比較ではない。

### 次に有効な実験

同一batch、同一更新回数、複数seedでv1/v2を再学習し、positionとrotationのtrade-offを検証する。
