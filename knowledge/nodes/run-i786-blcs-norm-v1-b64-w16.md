---
id: run-i786-blcs-norm-v1-b64-w16
type: run
title: BLCS court座標正規化 v1 baseline（batch 64）
issue: 786
provider: codex
session: 01a028df-6133-79c2-9266-f9b6a343765f
date: '2026-08-23'
status: done
config:
  model: multiview_axial_base
  loss: default
  data: broadcast_norm_v1
metrics:
  loss: 0.0919633508
  position_error_m: 2.4052329063
  x_error_m: 0.5272797942
  y_error_m: 2.1393527985
  z_error_m: 0.4554491341
  endpoint_error_m: 4.6605615616
repro:
  commit: 7aff92cb59eb6c4abfa844fcf19a9452ee7e8000
  branch: feat/issue-786-normalization-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.blcs.scripts.train
    --config-name train_normalization_v1 paths.data_root=/home/kamimura/projects/tennis-lab/data
    run.output_dir=blcs/i786/norm-v1/multiview_axial_base run.gpus=1 data.batch_size=64
    data.num_workers=16
artifacts:
  run_dir: knowledge/runs/run-i786-blcs-norm-v1-b64-w16
  log: .training_queue/logs/1787472877702221163_605399_i786-blcs-norm-v1-b64-w16.log
  tb_logdir: outputs/blcs/i786/norm-v1/multiview_axial_base/logs/version_1
  curves: knowledge/runs/run-i786-blcs-norm-v1-b64-w16/curves.png
parents: []
relations: []
tags:
- blcs
- normalization-v1
- baseline
- batch-64
---

## 考察 / Findings

### 要約

legacyの軸別scale `(5.485, 11.885, 1.07)m` とmetadataなし既存datasetを明示的v1として学習した比較baseline。100 epoch完走し、test `position_error_m=2.4052m`だった。

### アーキテクチャ詳細

`multiview_axial_base`をsingle-view broadcast入力で学習した。GPU 0、batch size 64、DataLoader worker 16、bf16 mixed precision、`torch.compile`を使用した。position lossはv1互換のnormalized Smooth L1契約である。

### メトリクスの解釈

軸別MAEはX `0.5273m`、Y `2.1394m`、Z `0.4554m`で、誤差の大部分はY方向だった。endpoint errorは`4.6606m`である。

### アーキテクチャ⇄メトリクスの因果考察

観測としてY誤差が支配的だが、単一seedの本runだけでは、軸別normalization scaleがその原因だとは断定できない。このrunはv2と同じmodel、batch、worker、epoch条件を持つため、BLCS内のscale変更比較のbaselineとして使用できる。

### 既存実験との比較

同Issueのv2 run `run-i786-blcs-norm-v2-b64-w16`が直接比較対象である。

### 次に有効な実験

複数seedでv1/v2を再実行し、trajectory長・可視率別にY誤差とendpoint errorを層別化する。
