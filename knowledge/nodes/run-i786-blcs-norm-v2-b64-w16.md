---
id: run-i786-blcs-norm-v2-b64-w16
type: run
title: BLCS court座標正規化 v2 baseline（batch 64）
issue: 786
provider: codex
session: 01a028df-6133-79c2-9266-f9b6a343765f
date: '2026-08-23'
status: done
config:
  model: multiview_axial_base
  loss: default
  data: broadcast_norm_v2
metrics:
  loss: 0.0601557530
  mean_position_error_m: 2.33845
  mean_x_error_m: 0.434171
  mean_y_error_m: 2.122155
  mean_z_error_m: 0.50461
  mean_endpoint_error_m: 4.364187
  position_accuracy_0_3m: 0.069732
  position_accuracy_0_6m: 0.213814
  position_accuracy_1_2m: 0.434886
  endpoint_accuracy_0_5m: 0.04
  endpoint_accuracy_1m: 0.11
repro:
  commit: 7aff92cb59eb6c4abfa844fcf19a9452ee7e8000
  branch: feat/issue-786-normalization-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.blcs.scripts.train
    --config-name train_normalization_v2 paths.data_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-786-normalization-v2/data
    run.output_dir=blcs/i786/norm-v2/multiview_axial_base run.gpus=1 data.batch_size=64
    data.num_workers=16
artifacts:
  run_dir: knowledge/runs/run-i786-blcs-norm-v2-b64-w16
  predictions: knowledge/runs/run-i786-blcs-norm-v2-b64-w16/pred_test.npz
  log: .training_queue/logs/1787472877773340690_605419_i786-blcs-norm-v2-b64-w16.log
  curves: knowledge/runs/run-i786-blcs-norm-v2-b64-w16/curves.png
  tb_logdir: outputs/blcs/i786/norm-v2/multiview_axial_base/logs/version_1
parents: [run-i786-blcs-norm-v1-b64-w16]
relations:
  - {to: run-i786-blcs-norm-v1-b64-w16, rel: compares}
tags: [blcs, normalization-v2, baseline, batch-64]
---

## 考察 / Findings

### 要約

XYZ全軸へ`11.885m`を適用するv2 datasetで100 epochを完走し、test `mean_position_error_m=2.3385m`を得た。同条件v1比で`0.0668m`（2.8%）低い。

### アーキテクチャ詳細

v1と同じ`multiview_axial_base`、GPU 0、batch size 64、DataLoader worker 16、bf16 mixed precision、`torch.compile`を使用した。datasetとcheckpointはv2 metadata・artifact名でv1から分離されている。

### メトリクスの解釈

v1比でXは`0.5273→0.4342m`、Yは`2.1394→2.1222m`、endpointは`4.6606→4.3642m`へ低下した。一方Zは`0.4554→0.5046m`へ増加した。統合lossは`0.0920→0.0602`だが、normalized lossはscale契約が異なるため物理m metricほど直接比較可能ではない。

### アーキテクチャ⇄メトリクスの因果考察

v2でZのnormalized勾配倍率が除かれたことと、Z誤差増加・X誤差低下は整合的だが、単一seedなので因果確定ではない。物理mの統合誤差とendpoint errorは悪化せず、等方scaleへの変更で学習が破綻しないことは観測できた。

### 既存実験との比較

直接baselineは`run-i786-blcs-norm-v1-b64-w16`。model、batch、worker、epochが一致するため、今回の4 run中では最も制御されたv1/v2比較である。

### 次に有効な実験

3 seeds以上でX改善とZ悪化の再現性を確認し、物理m単位の軸等方lossを併用するかを判断する。
