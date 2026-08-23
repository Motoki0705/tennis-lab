---
id: run-i786-plcs-v2-resume-b24-r2
type: run
title: PLCS court座標正規化 v2 baseline（batch 24継続）
issue: 786
provider: codex
session: 01a028df-6133-79c2-9266-f9b6a343765f
date: '2026-08-23'
status: done
config:
  model: multiview_axial_base
  loss: canonical
  data: multiview_sequence_norm_v2
metrics:
  loss: 0.0448162854
  position_error_m: 0.3137337267
  x_error_m: 0.1715996265
  y_error_m: 0.2071425468
  z_error_m: 0.0663859546
  angular_error_deg: 63.6289901733
repro:
  commit: b356bbeb6bedeb43d4dd07c0b574f4eddc8c4f8c
  branch: feat/issue-786-normalization-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.plcs.scripts.train
    --config-name train_norm_v2 paths.data_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-786-normalization-v2/data
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/data run.output_dir=plcs/i786/norm-v2/multiview_axial_base
    run.gpus=1 data.batch_size=24 training.steps_per_epoch=200 run.resume=plcs/i786/norm-v2/multiview_axial_base/logs/version_0/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-i786-plcs-v2-resume-b24-r2
  log: .training_queue/logs/1787470103957374667_558507_i786-plcs-v2-resume-b24-r2.log
  tb_logdir: outputs/plcs/i786/norm-v2/multiview_axial_base/logs/version_1
  curves: knowledge/runs/run-i786-plcs-v2-resume-b24-r2/curves.png
parents:
- run-i786-plcs-norm-v1
relations:
- to: run-i786-plcs-norm-v1
  rel: compares
tags:
- plcs
- normalization-v2
- baseline
- resumed
- batch-24
---

## 考察 / Findings

### 要約

v2学習をepoch 62 checkpointからbatch size 24で継続し、epoch 101で終了した。test `position_error_m=0.3137m`はv1より`0.1561m`（33.2%）低い一方、`angular_error_deg=63.63°`は悪化した。

### アーキテクチャ詳細

`multiview_axial_base`、canonical loss、v2 datasetを使用。epoch 62まではbatch 4、その後はoptimizer/schedulerをcheckpointから復元し、batch 24・`steps_per_epoch=200`で継続した。GPU 0単独で1 epochは概ね12–14秒だった。

### メトリクスの解釈

v1比でXは`0.1917→0.1716m`、Yは`0.3855→0.2071m`へ低下し、Zは`0.0447→0.0664m`へ増加した。position改善と引き換えに角度誤差は`52.06→63.63°`へ増加している。

### アーキテクチャ⇄メトリクスの因果考察

Y改善とZ悪化は等方scale化による軸間の勾配再配分と整合する。しかしbatch変更、継続checkpoint、停止epochがv1と異なるため、改善量をnormalization v2だけへ帰属させることはできない。

### 既存実験との比較

baselineは`run-i786-plcs-norm-v1`。物理m metricではpositionが改善したが、比較条件差とrotation悪化を併記する必要がある。

### 次に有効な実験

v1/v2を同じbatch 24、同じseed、同じoptimizer step数で最初から再実行し、positionとrotationを主評価にしたpaired comparisonを行う。
