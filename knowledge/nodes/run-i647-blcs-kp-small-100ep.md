---
id: run-i647-blcs-kp-small-100ep
type: run
title: i647_blcs_kp_small_100ep
issue: 647
provider: codex
session: 019f6013-0a29-7183-87ae-e4c221387139
date: '2026-07-14'
status: done
config:
  model: multiview_axial_small
  data: chunked_multiview_sequence_bs4
metrics:
  mean_position_error_m: 1.828465
  mean_x_error_m: 0.481019
  mean_y_error_m: 1.588772
  mean_z_error_m: 0.360708
  mean_endpoint_error_m: 3.046367
  position_accuracy_0_3m: 0.095205
  position_accuracy_0_6m: 0.280973
  position_accuracy_1_2m: 0.568345
  endpoint_accuracy_0_5m: 0.04
  endpoint_accuracy_1m: 0.22
repro:
  commit: f26addb77c3b6c3657f10caa8917328f22ab0274
  branch: feat/issue-647-court-line-token
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train model=multiview_axial_small data=chunked_multiview_sequence_bs4
    data.scene_dir=data/blcs_broadcast data.chunk.chunks_dir=data/blcs_broadcast/chunks
    camera=broadcast data.batch_size=8 data.num_workers=4 data.num_views_range=[1,1]
    data.seq_len_range=[64,256] data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20
    training.position_axis_weights=[1.0,4.0,1.0] training.reprojection_loss_weight=0.1
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false run.output_dir=outputs/blcs/issue647_kp_small_100ep
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i647-blcs-kp-small-100ep
  predictions: knowledge/runs/run-i647-blcs-kp-small-100ep/pred_test.npz
  log: .training_queue/logs/1784028363231758827_1965012_i647_blcs_kp_small_100ep.log
  output_dir: outputs/blcs/issue647_kp_small_100ep/logs/version_0
  curves: knowledge/runs/run-i647-blcs-kp-small-100ep/curves.png
  tb_logdir: outputs/blcs/issue647_kp_small_100ep/logs/version_0
parents: []
relations:
- to: run-i647-blcs-line-small-100ep
  rel: compares
- to: run-i647-blcs-line-moderate-v2-100ep
  rel: compares
tags:
- blcs
- court-kp
- baseline
---

## 考察 / Findings

### 要約

line版と同一broadcast split、small axial規模、seed、100 epochのBLCS KP対照。test位置 `1.828m`、endpoint `3.046m` まで学習した。

### 比較

line strong [[run-i647-blcs-line-small-100ep]] の位置 `8.144m` / endpoint `9.905m`、line moderate [[run-i647-blcs-line-moderate-v2-100ep]] の `8.427m` / `9.555m` に対して大幅に良い。`1.2m`以内精度もKP `56.83%`、lineは約`2%`である。

### 解釈

KP経路の後方互換と学習可能性は維持されている。line版の差はtrainerやsplitではなく、部分線分集合を固定順flattenする表現・増強耐性に起因する可能性が高い。
