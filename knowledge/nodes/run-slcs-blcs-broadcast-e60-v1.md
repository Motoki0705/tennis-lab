---
id: run-slcs-blcs-broadcast-e60-v1
type: run
title: 'BLCS broadcast単眼: 60 epoch探索学習'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: multiview_axial_small
  loss: default
  data: single_object_broadcast
  epochs: 60
metrics:
  position_error_m: 2.52331
  position_accuracy_0.3m: 0.015809
  endpoint_error_m: 4.241859
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_broadcast_real_rgb run.output_dir=blcs/train/broadcast_real_rgb/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-blcs-broadcast-e60-v1
  predictions: knowledge/runs/run-slcs-blcs-broadcast-e60-v1/pred_test.npz
  output_dir: outputs/blcs/train/broadcast_real_rgb/s42-001
  curves: knowledge/runs/run-slcs-blcs-broadcast-e60-v1/curves.png
  tb_logdir: outputs/blcs/train/broadcast_real_rgb/s42-001/logs/version_0
parents: []
relations: []
tags:
- slcs
- real-rgb
---

## 考察 / Findings

### 要約
broadcast単眼モデルの60 epoch学習を完了した。合成held-out評価のため、実動画の採用には別途観測整合性検査を必要とする。

### アーキテクチャ詳細
256幅・8層axial model、単眼、physical_v1。128フレーム、検出欠損・座標ノイズ・連続欠損を学習時に付加。

### メトリクスの解釈
保存されたtest指標は最終epoch59に対する値。実動画生成に使用するcheckpointはvalidation位置誤差最小（epoch53）で選択し、testでは選択しない。指標: {"position_error_m": 2.52331, "position_accuracy_0.3m": 0.015809, "endpoint_error_m": 4.241859}

### アーキテクチャ⇄メトリクスの因果考察
仮説: 単眼では奥行きが曖昧で、synthetic held-outのメートル級誤差は実動画teacherの不確かさにもつながる。2Dの観測支持・再投影でラベル重みを抑える必要がある。

### 既存実験との比較
このrun自体に同条件の対照実験はない。Meiji用3–4視点モデルとの数値比較はデータ・視点数が異なるため性能差の断定に用いない。

### 次に有効な実験
保存済みbroadcast ball UVを検査し、新たな人物・Court観測と組み合わせて実動画の再投影を評価する。必要なら追加fine-tuningを実施する。
